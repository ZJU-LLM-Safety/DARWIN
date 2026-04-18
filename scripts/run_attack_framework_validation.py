"""Run end-to-end DARWIN attack validation on sampled HarmBench questions."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
NEWCODE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = NEWCODE_ROOT.parent
if str(NEWCODE_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWCODE_ROOT))
from config.settings import DATA_DIR, FUSED_STRATEGY_SANDBOX_ENABLED, REFLECTIVE_SANDBOX_ENABLED, SANDBOX_DATASET_PATH


DEFAULT_OUTPUT_ROOT = Path(DATA_DIR) / "attack_framework_runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the runtime DARWIN attack pipeline.")
    parser.add_argument("--dataset", default=SANDBOX_DATASET_PATH, help="Path to HarmBench-style JSON dataset.")
    parser.add_argument("--sample-size", type=int, default=10, help="How many questions to sample.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for question sampling.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--run-id", default=None, help="Optional stable run id.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for run artifacts.")
    parser.add_argument(
        "--reflective-sandbox",
        action=argparse.BooleanOptionalAction,
        default=REFLECTIVE_SANDBOX_ENABLED,
        help="Whether reflective strategies should be sandbox-validated before pool insertion.",
    )
    parser.add_argument(
        "--fused-sandbox",
        action=argparse.BooleanOptionalAction,
        default=FUSED_STRATEGY_SANDBOX_ENABLED,
        help="Whether successful fused strategies should be sandbox-validated before pool insertion.",
    )
    return parser.parse_args(argv)


def ensure_cuda_visible_devices(value: str | None) -> None:
    if value:
        os.environ["CUDA_VISIBLE_DEVICES"] = value


def load_sampled_questions(dataset_path: Path, sample_size: int, sample_seed: int) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if sample_size <= 0 or sample_size >= len(payload):
        return list(payload)
    rng = random.Random(sample_seed)
    return rng.sample(payload, sample_size)


def ensure_run_dir(output_root: Path, run_id: str | None) -> tuple[str, Path]:
    resolved_run_id = run_id or datetime.now().strftime("attack_validation_%Y%m%d_%H%M%S")
    run_dir = output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return resolved_run_id, run_dir


def save_results(run_dir: Path, run_summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    summary_path = run_dir / "summary.json"
    jsonl_path = run_dir / "results.jsonl"
    summary_path.write_text(
        json.dumps({**run_summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_cuda_visible_devices(args.cuda_visible_devices)

    from attack.attack_pipeline import AttackPipeline
    from database.chroma_db import ChromaDB
    from database.sqlite_db import SQLiteDB

    dataset_path = Path(args.dataset)
    sampled_questions = load_sampled_questions(dataset_path, args.sample_size, args.sample_seed)
    run_id, run_dir = ensure_run_dir(Path(args.output_root), args.run_id)

    sqlite = SQLiteDB()
    chroma = ChromaDB()
    pipeline = AttackPipeline(
        sqlite,
        chroma,
        reflective_sandbox_enabled=args.reflective_sandbox,
        fused_strategy_sandbox_enabled=args.fused_sandbox,
    )

    manifest = {
        "run_id": run_id,
        "dataset": str(dataset_path),
        "sample_size": len(sampled_questions),
        "sample_seed": args.sample_seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "reflective_sandbox_enabled": args.reflective_sandbox,
        "fused_strategy_sandbox_enabled": args.fused_sandbox,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    success_count = 0
    for index, item in enumerate(sampled_questions, start=1):
        question = item["goal"]
        print(f"[{index}/{len(sampled_questions)}] goal_id={item.get('id')} {question[:120]}", flush=True)
        result = pipeline.attack_question(question, verbose=True)
        row = {
            "question_index": index,
            "goal_id": item.get("id"),
            "question": question,
            "functional_category": item.get("FunctionalCategory", ""),
            "semantic_category": item.get("SemanticCategory", ""),
            "success": bool(result.get("success", False)),
            "score": float(result.get("score", 0.0)),
            "chain": result.get("chain"),
            "step": result.get("step"),
            "strategy_id": result.get("strategy_id"),
            "strategy_name": result.get("strategy_name", ""),
            "disguised_prompt": result.get("disguised_prompt", ""),
            "target_response": result.get("response", ""),
            "judge_output": result.get("judge_output", ""),
            "used_history": bool(result.get("used_history", False)),
        }
        success_count += int(row["success"])
        results.append(row)
        save_results(
            run_dir,
            {
                **manifest,
                "completed": len(results),
                "success_count": success_count,
                "success_rate": success_count / max(len(results), 1),
            },
            results,
        )
        print(
            f"  -> {'SUCCESS' if row['success'] else 'FAILED'} "
            f"score={row['score']:.2f} strategy={row['strategy_name']}",
            flush=True,
        )

    final_summary = {
        **manifest,
        "completed": len(results),
        "success_count": success_count,
        "success_rate": success_count / max(len(results), 1),
    }
    save_results(run_dir, final_summary, results)
    print(json.dumps(final_summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
