"""DARWIN: Self-evolving jailbreak strategy framework — CLI entry point."""
import sys
import os
import json
import argparse
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    DATA_DIR,
    DEFAULT_EVOLUTION_INTERVAL_SECONDS,
    DEFAULT_SELECTED_STRATEGY_CATALOG,
    EXTERNAL_SANDBOX_ENABLED,
    FUSED_STRATEGY_SANDBOX_ENABLED,
    GENETIC_SANDBOX_ENABLED,
    REFLECTIVE_SANDBOX_ENABLED,
)


def get_components():
    """Initialize all shared components."""
    from database.sqlite_db import SQLiteDB
    from database.chroma_db import ChromaDB

    os.makedirs(DATA_DIR, exist_ok=True)
    sqlite = SQLiteDB()
    chroma = ChromaDB()
    return sqlite, chroma


def cmd_seed(args):
    """Import the default DARWIN seed strategies."""
    from strategy.strategy_pool import StrategyPool

    sqlite, chroma = get_components()
    pool = StrategyPool(sqlite, chroma)
    added = pool.seed_from_defaults()
    print(f"Seeded {added} strategies.")
    status = pool.status()
    print(f"Pool status: {status['active']} active, {status['total']} total")


def cmd_status(args):
    """Show strategy pool and system status."""
    from strategy.strategy_pool import StrategyPool
    from evolution.gan_evolution import GANEvolution

    sqlite, chroma = get_components()
    pool = StrategyPool(sqlite, chroma)
    s = pool.status()
    print("=== Strategy Pool Status ===")
    print(f"  Total strategies: {s['total']}")
    print(f"  Active: {s['active']}")
    print(f"  Silent: {s['silent']}")
    print(f"  Total attempts: {s['total_attempts']}")
    print(f"  Total successes: {s['total_successes']}")
    print(f"  Overall win rate: {s['overall_win_rate']:.2%}")
    print(f"  Sources: {s['sources']}")

    # Show individual strategies
    print("\n=== Strategies ===")
    for strat in pool.get_all():
        wr = (strat['total_successes'] / max(strat['total_attempts'], 1))
        print(f"  [{strat['status']:6s}] id={strat['id']:3d} "
              f"| {strat['name'][:30]:30s} "
              f"| {strat['total_successes']}/{strat['total_attempts']} "
              f"({wr:.0%}) | sandbox={strat.get('sandbox_success_rate', 0.0):.0%} "
              f"| fail_streak={strat.get('consecutive_failures', 0)} "
              f"| src={strat['source']}")

    # GAN progression
    gan = GANEvolution(sqlite)
    print(f"\n=== GAN Target Model ===")
    print(f"  Current: {gan.get_current_model()}")
    prog = gan.get_progression_status()
    for p in prog:
        wr = p['total_successes'] / max(p['total_attacks'], 1)
        print(f"  {p['model_name']}: {p['total_successes']}/{p['total_attacks']} ({wr:.0%})")


def cmd_collect(args):
    """Run external data collection and strategy extraction."""
    from evolution.external_evolution import ExternalEvolution

    sqlite, chroma = get_components()
    ext = ExternalEvolution(sqlite, chroma, sandbox_enabled=args.sandbox)
    stats = ext.run(max_items_per_source=args.max_items, verbose=True)
    print(f"\nCollection complete: {stats}")


def cmd_review_external(args):
    """Collect external sources and export sanitized strategy review cards."""
    from evolution.external_evolution import ExternalEvolution

    ext = ExternalEvolution()
    stats = ext.review(
        max_items_per_source=args.max_items,
        verbose=True,
        output_path=args.out,
    )
    print(f"\nExternal review complete: {stats}")


def cmd_evolve(args):
    """Run genetic evolution (crossover + mutation)."""
    from evolution.genetic_evolution import GeneticEvolution
    from strategy.strategy_pool import StrategyPool

    sqlite, chroma = get_components()
    gen = GeneticEvolution(sqlite, chroma, sandbox_enabled=args.sandbox)
    pool = StrategyPool(sqlite, chroma)

    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== Internal Evolution Cycle {cycle} ===")
        pruned_before = pool.prune()
        if pruned_before:
            print(f"Pruned before evolve: {pruned_before}")

        stats = gen.evolve(n_offspring=args.offspring, verbose=True)
        print(f"\nEvolution complete: {stats}")

        pruned_after = pool.prune()
        if pruned_after:
            print(f"Pruned after evolve: {pruned_after}")

        if not args.loop:
            break
        print(f"Sleeping {args.interval_seconds}s before next cycle...")
        time.sleep(args.interval_seconds)


def cmd_bootstrap_selected(args):
    """Bootstrap the self-evolving pool from sandbox-filtered strategies."""
    from strategy.strategy_pool import StrategyPool

    sqlite, chroma = get_components()
    pool = StrategyPool(sqlite, chroma)
    stats = pool.bootstrap_from_selected_catalog(args.catalog, activate=not args.silent)
    print(f"Bootstrap complete: {stats}")
    summary = pool.status()
    print(
        f"Pool status: {summary['active']} active, "
        f"{summary['silent']} silent, {summary['total']} total"
    )


def cmd_attack(args):
    """Run attack on a single question or a dataset."""
    from attack.attack_pipeline import AttackPipeline

    sqlite, chroma = get_components()
    pipeline = AttackPipeline(
        sqlite,
        chroma,
        reflective_sandbox_enabled=args.reflective_sandbox,
        fused_strategy_sandbox_enabled=args.fused_sandbox,
    )

    if args.question:
        questions = [args.question]
    elif args.dataset:
        with open(args.dataset) as f:
            data = json.load(f)
        questions = [item["goal"] for item in data]
        if args.limit:
            questions = questions[:args.limit]
    else:
        print("Provide --question or --dataset")
        return

    if args.target:
        pipeline.target_model = args.target

    results = []
    total_success = 0
    for i, q in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(questions)}] {q[:80]}...")
        print('='*60)
        result = pipeline.attack_question(q, verbose=True)
        results.append({"question": q, **result})
        if result["success"]:
            total_success += 1
        print(f"  → {'SUCCESS' if result['success'] else 'FAILED'} "
              f"(score={result['score']:.2f})")

    print(f"\n{'='*60}")
    print(f"Attack Summary: {total_success}/{len(questions)} "
          f"({total_success/max(len(questions),1):.0%})")

    # Save results
    out_path = os.path.join(DATA_DIR, "attack_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


def cmd_prune(args):
    """Prune underperforming strategies."""
    from strategy.strategy_pool import StrategyPool

    sqlite, chroma = get_components()
    pool = StrategyPool(sqlite, chroma)
    pruned = pool.prune()
    print(f"Pruned {len(pruned)} strategies: {pruned}")


def main():
    parser = argparse.ArgumentParser(
        description="DARWIN: self-evolving jailbreak strategy framework"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # seed
    sub.add_parser("seed", help="Import the default DARWIN seed strategies")

    # status
    sub.add_parser("status", help="Show strategy pool status")

    # collect
    p_collect = sub.add_parser("collect", help="Run external data collection")
    p_collect.add_argument("--max-items", type=int, default=20,
                           help="Max items per source")
    p_collect.add_argument(
        "--sandbox",
        action=argparse.BooleanOptionalAction,
        default=EXTERNAL_SANDBOX_ENABLED,
        help="Whether to run sandbox validation before adding external strategies to the pool.",
    )

    # review-external
    p_review = sub.add_parser(
        "review-external",
        help="Collect external sources and export sanitized strategy cards",
    )
    p_review.add_argument("--max-items", type=int, default=20,
                          help="Max items per source")
    p_review.add_argument("--out", type=str,
                          help="Optional output JSON path")

    # evolve
    p_evolve = sub.add_parser("evolve", help="Run genetic evolution")
    p_evolve.add_argument("--offspring", type=int, default=5,
                          help="Number of offspring per generation")
    p_evolve.add_argument("--loop", action="store_true",
                          help="Run internal evolution periodically")
    p_evolve.add_argument("--interval-seconds", type=int,
                          default=DEFAULT_EVOLUTION_INTERVAL_SECONDS,
                          help="Sleep interval between evolution cycles when --loop is set")
    p_evolve.add_argument(
        "--sandbox",
        action=argparse.BooleanOptionalAction,
        default=GENETIC_SANDBOX_ENABLED,
        help="Whether to sandbox-validate internal crossover/mutation strategies before pool insertion.",
    )

    # bootstrap-selected
    p_bootstrap = sub.add_parser(
        "bootstrap-selected",
        help="Initialize the self-evolving strategy pool from selected sandbox strategies",
    )
    p_bootstrap.add_argument(
        "--catalog",
        type=str,
        default=DEFAULT_SELECTED_STRATEGY_CATALOG,
        help="Path to selected_strategy_catalog.json",
    )
    p_bootstrap.add_argument(
        "--silent",
        action="store_true",
        help="Import strategies in silent state instead of active",
    )

    # attack
    p_attack = sub.add_parser("attack", help="Run attack")
    p_attack.add_argument("--question", type=str, help="Single question to attack")
    p_attack.add_argument("--dataset", type=str, help="Path to JSON dataset")
    p_attack.add_argument("--target", type=str, help="Target model name")
    p_attack.add_argument("--limit", type=int, help="Limit number of questions")
    p_attack.add_argument(
        "--reflective-sandbox",
        action=argparse.BooleanOptionalAction,
        default=REFLECTIVE_SANDBOX_ENABLED,
        help="Whether reflective strategies should be sandbox-validated before pool insertion.",
    )
    p_attack.add_argument(
        "--fused-sandbox",
        action=argparse.BooleanOptionalAction,
        default=FUSED_STRATEGY_SANDBOX_ENABLED,
        help="Whether successful fused strategies should be sandbox-validated before pool insertion.",
    )

    # prune
    sub.add_parser("prune", help="Prune underperforming strategies")

    args = parser.parse_args()

    commands = {
        "seed": cmd_seed,
        "status": cmd_status,
        "collect": cmd_collect,
        "review-external": cmd_review_external,
        "bootstrap-selected": cmd_bootstrap_selected,
        "evolve": cmd_evolve,
        "attack": cmd_attack,
        "prune": cmd_prune,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
