from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config, pairs_per_round


def parser(*, internal=False):
    p = argparse.ArgumentParser(prog="darwin-guard", description="DARWIN-Guard online adversarial training")
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-config", help="Check paper settings and required runtime inputs")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--allow-unset", action="store_true", help="Audit a template; does not approve a training run")
    prepare = sub.add_parser("prepare-data", help="Prepare user-owned training datasets offline")
    prepare.add_argument("--wildjailbreak", type=Path, required=True, help="WildJailbreak training-split JSON/JSONL/TSV/CSV")
    prepare.add_argument("--orbench", type=Path, required=True, help="OR-Bench benign JSON/JSONL/TSV/CSV")
    prepare.add_argument("--output-dir", type=Path, required=True)
    train = sub.add_parser("train", help="Run the 20-round paper training profile")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    evaluation = sub.add_parser("evaluate", help="Evaluate one checkpoint on labeled benchmark prompts")
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--input", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    summary = sub.add_parser("summarize", help="Compute binary metrics from recorded predictions")
    summary.add_argument("--input", type=Path, required=True)
    if internal:
        worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
        worker.add_argument("--stage", choices=("collect", "train"), required=True)
        worker.add_argument("--resolved-config", type=Path, required=True)
        worker.add_argument("--round-dir", type=Path, required=True)
        worker.add_argument("--model", required=True)
        worker.add_argument("--round-index", type=int, required=True)
    return p


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    args = parser(internal=bool(arguments and arguments[0] == "_worker")).parse_args(arguments)
    if args.command == "validate-config":
        from .config import missing_settings
        config = load_config(args.config, allow_unset=args.allow_unset)
        missing = missing_settings(config)
        attack_profile = None
        if config.get("attack", {}).get("config_path"):
            from .attack_bridge import validate_attack_profile
            attack_profile = validate_attack_profile(config["attack"]["config_path"])
        print(json.dumps({"paper_settings": "valid", "runtime_ready": not missing,
                          "unset": missing, "attack_profile": attack_profile}, indent=2))
    elif args.command == "prepare-data":
        from .data import deduplicate_sources, load_sources, write_jsonl
        harmful = load_sources(args.wildjailbreak, source_kind="wildjailbreak")
        benign = load_sources(args.orbench, source_kind="orbench")
        deduplicate_sources(harmful + benign)
        if not harmful or not benign:
            raise ValueError("Both training routes must contain valid source records")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in (("harmful.jsonl", harmful), ("benign.jsonl", benign)):
            if (args.output_dir / name).exists():
                raise FileExistsError("Prepared data already exist; select a new output directory")
        write_jsonl(args.output_dir / "harmful.jsonl", harmful)
        write_jsonl(args.output_dir / "benign.jsonl", benign)
        print(json.dumps({"harmful_sources": len(harmful), "benign_sources": len(benign)}))
    elif args.command == "train":
        from .online import run_online
        state = run_online(load_config(args.config), resume=args.resume)
        print(json.dumps({"completed_rounds": state["completed_rounds"], "checkpoint": state["guard_checkpoint"]}))
    elif args.command == "_worker":
        from .online import collect_worker, train_worker
        config = json.loads(args.resolved_config.read_text())
        run = collect_worker if args.stage == "collect" else train_worker
        run(config, args.round_dir, args.model, args.round_index)
    elif args.command == "evaluate":
        from .inference import HFGuard
        from .data import write_jsonl
        from .evaluation import check_evaluation_isolation, evaluation_input_settings, summarize
        config = load_config(args.config, allow_unset=True)
        records = read_jsonl(args.input)
        summarize([dict(row, decision=None) for row in records])
        input_settings = [evaluation_input_settings(row) for row in records]
        metadata_path = args.output.with_name(args.output.name + ".metadata.json")
        if args.output.exists() or metadata_path.exists():
            raise FileExistsError("Evaluation output already exists")
        isolation = check_evaluation_isolation(records, config)
        guard = HFGuard(str(args.checkpoint), config["training"], config["inference"])
        predictions = []
        for row, settings in zip(records, input_settings):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Each benchmark input must contain nonempty text")
            predictions.append({
                "benchmark": row["benchmark"], "id": row.get("id"),
                "label": row["label"],
                "decision": guard.predict(text, max_length=settings["max_input_tokens"]),
                "benchmark_group": settings["benchmark_group"],
            })
        write_jsonl(args.output, predictions)
        result = summarize(predictions)
        result["isolation_check"] = isolation
        benchmark_limits = {
            (settings["benchmark"], settings["benchmark_group"]): settings
            for settings in input_settings
        }
        metadata = {
            "checkpoint": str(args.checkpoint.resolve()),
            "input_limits_by_group": {
                settings["benchmark_group"]: settings["max_input_tokens"]
                for settings in input_settings
            },
            "benchmark_input_limits": [
                benchmark_limits[key] for key in sorted(benchmark_limits)
            ],
            "max_new_tokens": config["inference"]["max_new_tokens"],
            "isolation_check": isolation,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.command == "summarize":
        from .evaluation import summarize
        print(json.dumps(summarize(read_jsonl(args.input)), indent=2))


if __name__ == "__main__":
    main()
