"""Genetic evolution — crossover and mutation of active strategy templates."""
from __future__ import annotations

import random
import re
import time
from typing import Any

from config.prompts import CROSSOVER_PROMPT, MUTATION_PROMPT
from config.settings import CROSSOVER_TOP_K, GENETIC_SANDBOX_ENABLED, MUTATION_RATE
from database.chroma_db import ChromaDB
from database.sqlite_db import SQLiteDB
from models.llm_manager import LLMManager
from sandbox.validator import SandboxValidator
from strategy.strategy_pool import StrategyPool

from evolution.mutation_operators import get_random_operator


class GeneticEvolution:
    """Genetic algorithm-inspired strategy evolution: crossover + mutation."""

    def __init__(self, sqlite: SQLiteDB, chroma: ChromaDB, *, sandbox_enabled: bool = GENETIC_SANDBOX_ENABLED):
        self.pool = StrategyPool(sqlite, chroma)
        self.mgr = LLMManager()
        self.sandbox_enabled = sandbox_enabled
        self.sandbox = SandboxValidator() if sandbox_enabled else None

    def evolve(self, n_offspring: int = 5, verbose: bool = True) -> dict[str, Any]:
        """Run one generation of internal heuristic evolution."""
        stats: dict[str, Any] = {
            "active_candidates": 0,
            "crossover_attempts": 0,
            "crossover_added": 0,
            "mutation_attempts": 0,
            "mutation_added": 0,
            "duplicates": 0,
            "sandbox_failed": 0,
            "added_strategy_ids": [],
        }

        active = self.pool.get_ranked_active(top_k=max(CROSSOVER_TOP_K, 2))
        stats["active_candidates"] = len(active)
        if len(active) < 2:
            if verbose:
                print("[Genetic] Need at least 2 active strategies.")
            return stats

        top_k = active[: min(CROSSOVER_TOP_K, len(active))]
        rng = random.Random()

        for child_idx in range(n_offspring):
            do_mutation = random.random() < MUTATION_RATE
            if do_mutation:
                stats["mutation_attempts"] += 1
                parent = rng.choice(top_k)
                candidate_text = self._mutate(parent)
                candidate_name = self._derive_name(candidate_text, "Mutated Strategy")
                source = "genetic_mutation"
                parents = [parent["id"]]
            else:
                stats["crossover_attempts"] += 1
                parent_a, parent_b = rng.sample(top_k, 2)
                candidate_text = self._crossover(parent_a, parent_b)
                candidate_name = self._derive_name(candidate_text, "Crossover Strategy")
                source = "genetic_crossover"
                parents = [parent_a["id"], parent_b["id"]]

            if not candidate_text:
                continue

            added_id = self._try_add(
                text=candidate_text,
                name=candidate_name,
                source=source,
                parent_strategy_ids=parents,
                verbose=verbose,
                child_idx=child_idx,
                stats=stats,
            )
            if added_id is None:
                continue
            stats["added_strategy_ids"].append(added_id)
            if do_mutation:
                stats["mutation_added"] += 1
            else:
                stats["crossover_added"] += 1

        return stats

    def _crossover(self, a: dict[str, Any], b: dict[str, Any]) -> str | None:
        win_a = a["total_successes"] / max(a["total_attempts"], 1)
        win_b = b["total_successes"] / max(b["total_attempts"], 1)
        prompt = CROSSOVER_PROMPT.format(
            win_rate_a=win_a,
            strategy_a=a["text"],
            win_rate_b=win_b,
            strategy_b=b["text"],
        )
        result = self.mgr.gemma_generate(prompt).strip()
        return result if result else None

    def _mutate(self, parent: dict[str, Any]) -> str | None:
        op_name, op_dimension, op_desc = get_random_operator()
        prompt = MUTATION_PROMPT.format(
            original_strategy=parent["text"],
            mutation_name=op_name,
            mutation_description=op_desc,
        )
        result = self.mgr.gemma_generate(prompt).strip()
        return result if result else None

    def _try_add(
        self,
        *,
        text: str,
        name: str,
        source: str,
        parent_strategy_ids: list[int],
        verbose: bool,
        child_idx: int,
        stats: dict[str, Any],
    ) -> int | None:
        if self.pool.vec.is_duplicate_strategy(text):
            stats["duplicates"] += 1
            if verbose:
                print(f"  [Genetic] Duplicate candidate skipped: {name}")
            return None

        validation = None
        if self.sandbox_enabled and self.sandbox is not None:
            validation = self.sandbox.validate_detailed(
                text,
                strategy_name=name,
                strategy_id=f"{source}_{int(time.time())}_{child_idx}",
                source_group=source,
                source_path="",
                verbose=verbose,
            )
            if not validation["passed"]:
                stats["sandbox_failed"] += 1
                if verbose:
                    print(
                        f"  [Genetic] Sandbox failed: {name} "
                        f"ASR={validation['success_rate']:.2%}",
                    )
                return None

        sid = self.pool.add_strategy(
            name=name,
            text=text,
            tags=source,
            source=source,
            status="active",
            strategy_key=f"{source}_{int(time.time())}_{child_idx}",
            source_group=source,
            source_path="",
            metadata={
                "validation": (
                    {
                        "success_count": validation["success_count"],
                        "trial_count": validation["trial_count"],
                        "avg_score": validation["avg_score"],
                    }
                    if validation is not None else {"sandbox_skipped": True}
                )
            },
            sandbox_success_rate=validation["success_rate"] if validation is not None else 0.0,
            sandbox_avg_score=validation["avg_score"] if validation is not None else 0.0,
            generation=1 + max((self.pool.get(pid) or {}).get("generation", 0) for pid in parent_strategy_ids),
            parent_strategy_ids=parent_strategy_ids,
        )
        if sid is None:
            stats["duplicates"] += 1
            return None
        if verbose:
            print(
                f"  [Genetic] Added strategy id={sid} "
                f"ASR={(validation['success_rate'] if validation is not None else 0.0):.2%} "
                f"avg_score={(validation['avg_score'] if validation is not None else 0.0):.3f}",
            )
        return sid

    @staticmethod
    def _derive_name(strategy_text: str, fallback: str) -> str:
        match = re.search(r"\[\[(.*?)\]\]", strategy_text)
        if match:
            return match.group(1).strip()
        return fallback
