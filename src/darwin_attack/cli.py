from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .composer import FailureReflector, PromptComposer
from .config import DarwinConfig, load_config
from .datasets import assert_disjoint_goals, load_goals
from .embeddings import SentenceTransformerEmbedder
from .engine import AttackEngine
from .evolution import (
    GeneticStrategyEvolution,
    ReflectionDrivenEvolution,
    load_mutation_operators,
)
from .extraction import ExternalKnowledgeEvolution
from .judge import HarmfulnessJudge, IntentJudge, ResponseJudge
from .metrics import summarize
from .pool import StrategyPool
from .providers import build_model
from .reevaluation import rejudge_record, summarize_rejudged
from .sandbox import SandboxValidator
from .schemas import StrategyCandidate
from .selector import FeedbackGuidedEvolution
from .storage import Repository
from .targets import GuardrailTarget, LLMTarget
from .utils import read_jsonl, write_jsonl


def _pool(config: DarwinConfig, repository: Repository) -> StrategyPool:
    embedder = SentenceTransformerEmbedder(config.embedding)
    return StrategyPool(
        repository,
        embedder,
        config.embedding.similarity_threshold,
        config.pool.admission_threshold,
        config.pool.target_size,
    )


def _checked_datasets(config: DarwinConfig) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    sandbox_goals = load_goals(config.sandbox.dataset_path)
    evaluation_goals = load_goals(config.attack.dataset_path)
    assert_disjoint_goals(
        [(key, " ".join(text.split())) for key, text in sandbox_goals],
        [(key, " ".join(text.split())) for key, text in evaluation_goals],
    )
    return sandbox_goals, evaluation_goals


def _sandbox(config: DarwinConfig) -> SandboxValidator:
    sandbox_goals, _ = _checked_datasets(config)
    generator = build_model(config.model("strategy_generator"))
    sandbox_target = LLMTarget(build_model(config.model("sandbox_target")))
    judge = ResponseJudge(build_model(config.model("judge")), config.attack.success_score)
    goals = [goal for _, goal in sandbox_goals]
    return SandboxValidator(
        PromptComposer(generator),
        sandbox_target,
        judge,
        goals,
        config.sandbox.goals_per_candidate,
        config.sandbox.trials_per_goal,
        config.runtime.random_seed,
    )


def _admit_candidates(
    config: DarwinConfig,
    candidates: list[StrategyCandidate],
    output: Path,
) -> None:
    _checked_datasets(config)
    repository = Repository(config.runtime.database_path)
    try:
        pool = _pool(config, repository)
        sandbox = None
        records = []
        for candidate in candidates:
            decision = pool.precheck(candidate)
            report = None
            if decision is None:
                if sandbox is None:
                    sandbox = _sandbox(config)
                report = sandbox.validate(candidate)
                decision = pool.consider(candidate, report)
            records.append(
                {
                    "strategy_key": candidate.key,
                    "sandbox_success_rate": report.success_rate if report is not None else None,
                    "sandbox_average_score": report.average_score if report is not None else None,
                    "sandbox_skipped": report is None,
                    "admitted": decision.admitted,
                    "reason": decision.reason,
                    "maximum_similarity": decision.maximum_similarity,
                    "strategy_id": decision.strategy_id,
                }
            )
        write_jsonl(output, records)
        admitted = sum(int(record["admitted"]) for record in records)
        print(json.dumps({"processed": len(records), "admitted": admitted}))
    finally:
        repository.close()


def cmd_validate_config(args: argparse.Namespace) -> None:
    load_config(args.config)
    print("configuration: valid")


def cmd_extract(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    extractor = ExternalKnowledgeEvolution(build_model(config.model("strategy_generator")))
    candidates: list[dict[str, Any]] = []
    for record in read_jsonl(args.input):
        material = str(record.get("text") or record.get("content") or "").strip()
        if not material:
            continue
        source_id = str(record.get("source_id", ""))
        candidates.extend(item.to_dict() for item in extractor.extract(material, source_id))
    write_jsonl(args.output, candidates)
    print(json.dumps({"extracted": len(candidates), "output": str(args.output)}))


def cmd_admit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    candidates = [StrategyCandidate.from_dict(item) for item in read_jsonl(args.input)]
    _admit_candidates(config, candidates, args.output)


def cmd_load_released_pool(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    candidates = [StrategyCandidate.from_dict(item) for item in read_jsonl(args.input)]
    if len(candidates) != config.pool.target_size:
        raise ValueError(
            f"released pool has {len(candidates)} strategies; "
            f"expected pool.target_size={config.pool.target_size}"
        )
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("released pool contains duplicate strategy keys")

    repository = Repository(config.runtime.database_path)
    try:
        if repository.count_strategies() != 0:
            raise RuntimeError("load-released-pool requires an empty strategy database")
        embedder = SentenceTransformerEmbedder(config.embedding)
        vectors = embedder.encode([candidate.instruction for candidate in candidates])
        for candidate, vector in zip(candidates, vectors, strict=True):
            repository.add_released_strategy(candidate, vector)
        print(json.dumps({"loaded": len(candidates), "active": len(candidates)}))
    finally:
        repository.close()


def cmd_evolve(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _checked_datasets(config)
    repository = Repository(config.runtime.database_path)
    try:
        operators = load_mutation_operators(config.pool.mutation_operators_file)
        if len(operators) != config.pool.mutation_operator_count:
            raise ValueError("mutation operator count does not match pool.mutation_operator_count")
        evolution = GeneticStrategyEvolution(
            repository=repository,
            model=build_model(config.model("strategy_generator")),
            top_k=config.pool.crossover_top_k,
            crossover_probability=config.pool.crossover_probability,
            mutation_probability=config.pool.mutation_probability,
            mutation_operators=operators,
            random_seed=config.runtime.random_seed,
        )
        candidates = evolution.generate(args.count)
    finally:
        repository.close()
    _admit_candidates(config, candidates, args.output)


def cmd_reflect(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _checked_datasets(config)
    repository = Repository(config.runtime.database_path)
    try:
        reflection = ReflectionDrivenEvolution(
            repository, build_model(config.model("reflection"))
        )
        candidates = [
            reflection.generate(
                int(record["strategy_id"]),
                str(record["failed_prompt"]),
                str(record["feedback"]),
            )
            for record in read_jsonl(args.input)
        ]
    finally:
        repository.close()
    _admit_candidates(config, candidates, args.output)


def cmd_pool_status(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    repository = Repository(config.runtime.database_path)
    try:
        active = repository.active_strategies()
        print(json.dumps({"total": repository.count_strategies(), "active": len(active)}))
    finally:
        repository.close()


def cmd_attack(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, goals = _checked_datasets(config)
    repository = Repository(config.runtime.database_path)
    try:
        embedder = SentenceTransformerEmbedder(config.embedding)
        pool = StrategyPool(
            repository,
            embedder,
            config.embedding.similarity_threshold,
            config.pool.admission_threshold,
            config.pool.target_size,
        )
        scope = f"{config.attack.target_id}::{config.attack.dataset_id}"
        selector = FeedbackGuidedEvolution(
            repository,
            embedder,
            scope,
            config.embedding.history_threshold,
            config.selection.alpha,
            config.selection.gamma,
            config.runtime.random_seed,
        )
        target_model = build_model(config.model("target"))
        if config.attack.target_kind == "llm":
            target = LLMTarget(target_model)
        else:
            target = GuardrailTarget(
                target_model,
                config.attack.guardrail_safe_pattern,
                config.attack.guardrail_unsafe_pattern,
                template=config.attack.guardrail_template,
                template_text=(
                    config.attack.guardrail_template_file.read_text(encoding="utf-8")
                    if config.attack.guardrail_template_file else None
                ),
                template_id=config.attack.guardrail_template_id,
            )
        judge_model = build_model(config.model("judge"))
        engine = AttackEngine(
            config=config.attack,
            repository=repository,
            pool=pool,
            selector=selector,
            composer=PromptComposer(build_model(config.model("strategy_generator"))),
            reflector=FailureReflector(build_model(config.model("reflection"))),
            target=target,
            response_judge=ResponseJudge(judge_model, config.attack.success_score),
            intent_judge=IntentJudge(build_model(config.model("intent_judge"))),
            harmfulness_judge=(
                HarmfulnessJudge(judge_model) if config.attack.target_kind == "llm" else None
            ),
        )
        if args.limit is not None:
            goals = goals[: args.limit]
        results = [engine.attack(goal, instance_id) for instance_id, goal in goals]
        write_jsonl(args.output, (item.to_dict() for item in results))
        if config.attack.target_kind == "guardrail":
            metadata = {
                "target_id": config.attack.target_id,
                "model_identity": config.model("target").identity,
                "provider": config.model("target").provider,
                "temperature": config.model("target").temperature,
                "guardrail": target.metadata,
            }
            Path(str(args.output) + ".metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
            )
        print(json.dumps(summarize(results, config.attack.max_target_queries), indent=2))
    finally:
        repository.close()


def cmd_summarize(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.input))
    count = len(rows)
    successes = sum(bool(row.get("success")) for row in rows)
    query_counts = [
        int(row.get("query_count", 0)) if row.get("success") else args.query_budget for row in rows
    ]
    harmfulness = [
        float(row["harmfulness_score"]) for row in rows if row.get("harmfulness_score") is not None
    ]
    report = {
        "instances": count,
        "successes": successes,
        "asr": successes / count if count else 0.0,
        "harmfulness_score": sum(harmfulness) / len(harmfulness) if harmfulness else 0.0,
        "average_query_count": sum(query_counts) / count if count else 0.0,
    }
    print(json.dumps(report, indent=2))


def cmd_rejudge(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model = build_model(config.model("alternative_judge"))
    response_judge = ResponseJudge(model, config.attack.success_score)
    intent_judge = IntentJudge(model)
    harmfulness_judge = HarmfulnessJudge(model) if config.attack.target_kind == "llm" else None
    records = [
        rejudge_record(
            record,
            config.attack.target_kind,
            config.attack.max_target_queries,
            response_judge,
            intent_judge,
            harmfulness_judge,
        )
        for record in read_jsonl(args.input)
    ]
    write_jsonl(args.output, records)
    print(json.dumps(summarize_rejudged(records), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="darwin-attack")
    parser.add_argument("--config", type=Path, help="Path to a completed YAML configuration")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("validate-config", help="Validate configuration and exit")

    extract = commands.add_parser("extract", help="External Knowledge Evolution: extract candidate strategies")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)

    admit = commands.add_parser("admit", help="Deduplicate, sandbox, and admit candidates")
    admit.add_argument("--input", type=Path, required=True)
    admit.add_argument("--output", type=Path, required=True)

    load_pool = commands.add_parser(
        "load-released-pool",
        help="Load the published, prevalidated final strategy pool",
    )
    load_pool.add_argument("--input", type=Path, required=True)

    evolve = commands.add_parser("evolve", help="Genetic Strategy Evolution: generate and validate candidates")
    evolve.add_argument("--count", type=int, required=True)
    evolve.add_argument("--output", type=Path, required=True)

    reflect = commands.add_parser(
        "reflect", help="Reflection Driven Evolution: refine and validate failed strategies"
    )
    reflect.add_argument("--input", type=Path, required=True)
    reflect.add_argument("--output", type=Path, required=True)

    commands.add_parser("pool-status", help="Print strategy counts without strategy content")

    attack = commands.add_parser("attack", help="Evaluate DARWIN-Attack with Feedback Guided Evolution")
    attack.add_argument("--output", type=Path, required=True)
    attack.add_argument("--limit", type=int)

    summary = commands.add_parser("summarize", help="Aggregate an existing result JSONL file")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--query-budget", type=int, required=True)

    rejudge = commands.add_parser(
        "rejudge",
        help="Re-evaluate recorded outputs with the configured alternative judge",
    )
    rejudge.add_argument("--input", type=Path, required=True)
    rejudge.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "summarize" and args.config is None:
        parser.error("--config is required for this command")
    handlers = {
        "validate-config": cmd_validate_config,
        "extract": cmd_extract,
        "admit": cmd_admit,
        "load-released-pool": cmd_load_released_pool,
        "evolve": cmd_evolve,
        "reflect": cmd_reflect,
        "pool-status": cmd_pool_status,
        "attack": cmd_attack,
        "summarize": cmd_summarize,
        "rejudge": cmd_rejudge,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
