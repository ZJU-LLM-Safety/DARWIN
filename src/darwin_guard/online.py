from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from .config import fingerprint, pairs_per_round


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def copy_database(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)


def _worker(stage: str, config_path: Path, round_dir: Path, model_path: str, round_index: int):
    subprocess.run([
        sys.executable, "-m", "darwin_guard", "_worker", "--stage", stage,
        "--resolved-config", str(config_path), "--round-dir", str(round_dir),
        "--model", model_path, "--round-index", str(round_index),
    ], check=True)


def run_online(config: dict, *, resume=False, worker=None) -> dict:
    if worker is None:
        from .attack_bridge import validate_attack_profile
        validate_attack_profile(config["attack"]["config_path"])
    worker = worker or _worker
    root = Path(config["runtime"]["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".run.lock"
    import fcntl
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is already using this output directory") from exc
        state_path = root / "state.json"
        config_hash = fingerprint(config)
        if state_path.exists():
            if not resume:
                raise RuntimeError("Run state exists; use --resume or a new output directory")
            state = json.loads(state_path.read_text())
            if state["config_sha256"] != config_hash:
                raise RuntimeError("Resume configuration differs from the saved run")
            if state["completed_rounds"] > 0:
                if not (Path(state["guard_checkpoint"]) / "config.json").is_file():
                    raise RuntimeError("The completed guard checkpoint is missing")
                if not Path(state["strategy_database"]).is_file():
                    raise RuntimeError("The completed strategy database is missing")
        else:
            if any(root.glob("round_*")):
                raise RuntimeError("Round artifacts exist without run state; select a new output directory")
            initial = Path(config["attack"]["initial_database"])
            if not initial.is_file():
                raise FileNotFoundError("The initial admitted strategy database is missing")
            with sqlite3.connect(initial.resolve().as_uri() + "?mode=ro", uri=True) as db:
                active = db.execute("SELECT COUNT(*) FROM strategies WHERE status='active'").fetchone()[0]
                total = db.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
            if total != active:
                raise ValueError("The initial strategy database must contain only active admitted strategies")
            if active != 50:
                raise ValueError("A paper run must start from 50 admitted seeds, not the released final pool")
            state = {
                "config_sha256": config_hash, "completed_rounds": 0,
                "guard_checkpoint": config["models"]["guard"]["path"],
                "strategy_database": str(initial), "used_source_ids": [],
            }
            atomic_json(state_path, state)
        config_path = root / "resolved_config.json"
        atomic_json(config_path, config)
        for index in range(state["completed_rounds"] + 1, config["online"]["rounds"] + 1):
            round_dir = root / f"round_{index:04d}"
            round_dir.mkdir(exist_ok=True)
            if not (round_dir / "collection_complete.json").exists():
                copy_database(Path(state["strategy_database"]), round_dir / "strategies.sqlite")
                atomic_json(round_dir / "excluded_source_ids.json", state["used_source_ids"])
                previous = root / f"round_{index - 1:04d}" / "failures.json"
                atomic_json(round_dir / "previous_failures.json", json.loads(previous.read_text()) if previous.exists() else [])
                worker("collect", config_path, round_dir, state["guard_checkpoint"], index)
            collection = json.loads((round_dir / "collection_complete.json").read_text())
            if collection["pair_count"] != pairs_per_round(config):
                raise RuntimeError("Collection budget does not match the configured pair count")
            with sqlite3.connect((round_dir / "strategies.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
                active = db.execute("SELECT COUNT(*) FROM strategies WHERE status='active'").fetchone()[0]
            if index == config["online"]["rounds"] and active != 200:
                raise RuntimeError(f"The final paper round requires 200 admitted strategies; found {active}")
            if not (round_dir / "training_complete.json").exists():
                worker("train", config_path, round_dir, state["guard_checkpoint"], index)
            training = json.loads((round_dir / "training_complete.json").read_text())
            checkpoint = Path(training["checkpoint"])
            if not (checkpoint / "config.json").is_file():
                raise RuntimeError("Round did not produce a loadable model configuration")
            state = {
                "config_sha256": config_hash, "completed_rounds": index,
                "guard_checkpoint": str(checkpoint),
                "strategy_database": str(round_dir / "strategies.sqlite"),
                "strategy_pool_size": active,
                "used_source_ids": sorted(set(state["used_source_ids"]) | set(collection["used_source_ids"])),
            }
            atomic_json(state_path, state)
            print(json.dumps({"round": index, "pairs": collection["pair_count"], "status": "complete"}), flush=True)
        with sqlite3.connect(Path(state["strategy_database"]).resolve().as_uri() + "?mode=ro", uri=True) as db:
            final_active = db.execute("SELECT COUNT(*) FROM strategies WHERE status='active'").fetchone()[0]
        if final_active != 200:
            raise RuntimeError(f"The final paper checkpoint requires 200 admitted strategies; found {final_active}")
        return state


def collect_worker(config: dict, round_dir: Path, model_path: str, round_index: int):
    from .attack_bridge import DARWINAttackAdapter, grow_pool_for_round, validate_attack_profile
    from .collection import collect_pair
    from .data import deduplicate_sources, load_sources, write_jsonl
    from .filtering import IntentPreservationFilter
    from .inference import HFGuard, build_filter_model
    import random

    validate_attack_profile(config["attack"]["config_path"])
    harmful = load_sources(config["data"]["harmful_path"], expected_label=1)
    benign = load_sources(config["data"]["benign_path"], expected_label=0)
    deduplicate_sources(harmful + benign)
    excluded = set(json.loads((round_dir / "excluded_source_ids.json").read_text()))
    used = set(excluded)
    seed = config["runtime"]["seed"] + round_index
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    count = pairs_per_round(config)
    harmful_count = round(count * config["online"]["harmful_fraction"])
    quotas = {1: harmful_count, 0: count - harmful_count}
    proposer = DARWINAttackAdapter(
        config["attack"]["config_path"], str(round_dir / "strategies.sqlite"), round_index, seed,
        generator_settings={key: value for key, value in config["models"]["generator"].items() if key != "identity"},
    )
    pairs = []
    try:
        proposer.source_records.update({row["id"]: row["text"] for row in harmful + benign})
        proposer.validate_source_isolation()
        proposer.failure_records = json.loads((round_dir / "previous_failures.json").read_text())
        try:
            evolution = grow_pool_for_round(proposer, config["attack"], config["online"]["rounds"])
        except RuntimeError as exc:
            if hasattr(exc, "evolution_report"):
                atomic_json(round_dir / "evolution_failed.json", exc.evolution_report)
            raise
        atomic_json(round_dir / "evolution.json", evolution)
        proposer.release_evolution_models()
        proposer.failure_records = []
        guard = HFGuard(model_path, config["training"], config["inference"])
        judge = IntentPreservationFilter.from_model(build_filter_model(config["models"]["filter"], config["training"]["dtype"]))
        for label, sources in ((1, harmful), (0, benign)):
            sources = [x for x in sources if x["id"] not in used]
            rng.shuffle(sources)
            kept = 0
            for source in sources:
                if kept == quotas[label]:
                    break
                used.add(source["id"])
                pair = collect_pair(source, proposer, guard, judge, config["online"]["max_attempts"])
                if pair is not None:
                    pairs.append(pair)
                    kept += 1
            if kept != quotas[label]:
                raise RuntimeError(f"Source pool exhausted after filtering: label={label}, retained={kept}, required={quotas[label]}")
        rng.shuffle(pairs)
        write_jsonl(round_dir / "pairs.jsonl", pairs)
        atomic_json(round_dir / "failures.json", proposer.failure_records)
        atomic_json(round_dir / "collection_complete.json", {
            "pair_count": len(pairs), "row_count": len(pairs) * 2,
            "used_source_ids": sorted(used - excluded), "quotas": quotas,
            "evolution": evolution,
        })
    finally:
        proposer.close()


def train_worker(config: dict, round_dir: Path, model_path: str, round_index: int):
    from .training import train_round
    from datetime import datetime, timezone
    pairs = [json.loads(line) for line in (round_dir / "pairs.jsonl").read_text().splitlines() if line.strip()]
    destination = round_dir / "model"
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        destination.rename(round_dir / f"incomplete_model_{stamp}")
    settings = dict(config["training"], seed=config["runtime"]["seed"] + round_index)
    checkpoint = train_round(pairs, model_path, destination, settings)
    atomic_json(round_dir / "training_complete.json", {"checkpoint": str(checkpoint)})
