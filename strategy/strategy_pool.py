"""Strategy pool manager — init, dedup, prune, and selection."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from config.prompts import SEED_STRATEGIES
from config.settings import (
    DEFAULT_SELECTED_STRATEGY_CATALOG,
    PRUNE_MAX_CONSECUTIVE_FAILURES,
)
from database.chroma_db import ChromaDB
from database.sqlite_db import SQLiteDB


class StrategyPool:
    """Unified interface over SQLite + ChromaDB for strategy management."""

    def __init__(self, sqlite: SQLiteDB, chroma: ChromaDB):
        self.db = sqlite
        self.vec = chroma

    # ── Seed / Bootstrap ──────────────────────────────────────────
    def seed_from_defaults(self) -> int:
        """Import the default DARWIN seed strategies. Returns count added."""
        existing = self.db.get_all_strategies()
        existing_names = {s["name"] for s in existing}
        added = 0
        for idx, seed in enumerate(SEED_STRATEGIES):
            if seed["name"] == "Reverse Attack Strategy":
                continue
            if seed["name"] in existing_names:
                continue
            sid = self.add_strategy(
                name=seed["name"],
                text=seed["text"],
                tags=seed["tags"],
                source="seed",
                source_group="legacy_seed",
                strategy_key=f"seed_{idx:03d}",
                metadata={"origin": "legacy_seed"},
            )
            if sid is not None:
                added += 1
        return added

    def bootstrap_from_selected_catalog(
        self,
        catalog_path: str | None = None,
        *,
        activate: bool = True,
    ) -> dict[str, int]:
        """Import sandbox-filtered strategies as the initial evolving pool."""
        path = Path(catalog_path or DEFAULT_SELECTED_STRATEGY_CATALOG)
        if not path.exists():
            raise RuntimeError(f"Selected strategy catalog not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats = {"read": len(payload), "added": 0, "duplicates": 0, "existing": 0}
        for item in payload:
            strategy_key = item.get("strategy_id", "")
            if strategy_key and self.db.get_strategy_by_key(strategy_key) is not None:
                stats["existing"] += 1
                continue
            sid = self.add_strategy(
                name=item["strategy_name"],
                text=item["template_text"],
                tags=item.get("tags", ""),
                source="sandbox_selected",
                status="active" if activate else "silent",
                strategy_key=strategy_key,
                source_group=item.get("source_group", ""),
                source_path=item.get("source_path", ""),
                metadata={
                    "bootstrap": True,
                    "success_count": item.get("success_count", 0),
                    "trial_count": item.get("trial_count", 0),
                    "avg_score": item.get("avg_score", 0.0),
                },
                sandbox_success_rate=float(item.get("success_rate", 0.0)),
                sandbox_avg_score=float(item.get("avg_score", 0.0)),
                last_sandbox_run_id=path.parent.name,
                generation=0,
            )
            if sid is None:
                stats["duplicates"] += 1
            else:
                stats["added"] += 1
        return stats

    # ── Add New Strategy (with dedup) ─────────────────────────────
    def add_strategy(
        self,
        *,
        name: str,
        text: str,
        tags: str = "",
        source: str = "external",
        status: str = "active",
        strategy_key: str = "",
        source_group: str = "",
        source_path: str = "",
        metadata: dict[str, Any] | None = None,
        sandbox_success_rate: float = 0.0,
        sandbox_avg_score: float = 0.0,
        last_sandbox_run_id: str = "",
        generation: int = 0,
        parent_strategy_ids: list[int] | None = None,
    ) -> int | None:
        """Add a strategy if it's not a semantic duplicate. Returns ID or None."""
        if self.vec.is_duplicate_strategy(text):
            return None
        sid = self.db.add_strategy(
            name=name,
            text=text,
            tags=tags,
            source=source,
            status=status,
            strategy_key=strategy_key,
            source_group=source_group,
            source_path=source_path,
            metadata=metadata,
            sandbox_success_rate=sandbox_success_rate,
            sandbox_avg_score=sandbox_avg_score,
            last_sandbox_run_id=last_sandbox_run_id,
            generation=generation,
            parent_strategy_ids=parent_strategy_ids,
        )
        self.vec.add_strategy(
            sid,
            text,
            {
                "name": name,
                "source": source,
                "status": status,
                "strategy_key": strategy_key,
                "source_group": source_group,
            },
        )
        return sid

    # ── Get Strategies ────────────────────────────────────────────
    def get_active(self) -> list[dict[str, Any]]:
        return self.db.get_active_strategies()

    def get_all(self) -> list[dict[str, Any]]:
        return self.db.get_all_strategies()

    def get(self, strategy_id: int) -> dict[str, Any] | None:
        return self.db.get_strategy(strategy_id)

    def get_ranked_active(self, top_k: int | None = None) -> list[dict[str, Any]]:
        ranked = sorted(
            self.get_active(),
            key=lambda s: (
                s["total_successes"] / max(s["total_attempts"], 1),
                s.get("sandbox_success_rate", 0.0),
                s.get("sandbox_avg_score", 0.0),
            ),
            reverse=True,
        )
        if top_k is not None:
            return ranked[:top_k]
        return ranked

    # ── UCB-Weighted Selection ────────────────────────────────────
    def get_ucb_scores(self, c: float = 1.414) -> list[tuple[dict[str, Any], float]]:
        strategies = self.get_active()
        if not strategies:
            return []
        total_all = sum(s["total_attempts"] for s in strategies) or 1
        scored = []
        for s in strategies:
            n = s["total_attempts"] or 1
            w = s["total_successes"] / n
            exploration = c * math.sqrt(math.log(total_all) / n)
            ucb = w + exploration
            scored.append((s, ucb))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── Record Attempt ────────────────────────────────────────────
    def record_attempt(
        self,
        strategy_id: int,
        success: bool,
        score: float = 0.0,
        question: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        self.db.record_attempt(strategy_id, success, score, question, metadata)

    # ── Pruning ───────────────────────────────────────────────────
    def prune(self) -> list[int]:
        """Mark strategies with too many consecutive failures as silent."""
        pruned: list[int] = []
        for s in self.get_active():
            if s.get("consecutive_failures", 0) < PRUNE_MAX_CONSECUTIVE_FAILURES:
                continue
            self.db.update_strategy_status(s["id"], "silent")
            pruned.append(s["id"])
        return pruned

    # ── Status Summary ────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        all_strats = self.get_all()
        active = [s for s in all_strats if s["status"] == "active"]
        silent = [s for s in all_strats if s["status"] == "silent"]
        total_attempts = sum(s["total_attempts"] for s in all_strats)
        total_successes = sum(s["total_successes"] for s in all_strats)
        source_counts: dict[str, int] = {}
        for item in all_strats:
            source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
        return {
            "total": len(all_strats),
            "active": len(active),
            "silent": len(silent),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "overall_win_rate": (
                total_successes / total_attempts if total_attempts > 0 else 0.0
            ),
            "sources": source_counts,
        }
