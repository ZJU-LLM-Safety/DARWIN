from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schemas import AttackAttempt, SandboxReport, StrategyCandidate, StrategyRecord


class Repository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_key TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                instruction TEXT NOT NULL,
                mode TEXT NOT NULL CHECK(mode IN ('instruction', 'template')),
                tags_json TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'inactive')),
                total_attempts INTEGER NOT NULL DEFAULT 0,
                total_successes INTEGER NOT NULL DEFAULT 0,
                sandbox_success_rate REAL,
                sandbox_average_score REAL,
                validation_status TEXT NOT NULL DEFAULT 'measured'
                    CHECK(validation_status IN ('measured', 'released_prevalidated')),
                generation INTEGER NOT NULL DEFAULT 0,
                parent_ids_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS strategy_embeddings (
                strategy_id INTEGER PRIMARY KEY,
                vector_json TEXT NOT NULL,
                FOREIGN KEY(strategy_id) REFERENCES strategies(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                attempt_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS success_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                goal TEXT NOT NULL,
                goal_embedding_json TEXT NOT NULL,
                strategy_sequence_json TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS transitions (
                scope TEXT NOT NULL,
                from_strategy_id INTEGER NOT NULL,
                to_strategy_id INTEGER NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY(scope, from_strategy_id, to_strategy_id)
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def _record(row: sqlite3.Row) -> StrategyRecord:
        return StrategyRecord(
            id=int(row["id"]),
            key=str(row["strategy_key"]),
            name=str(row["name"]),
            instruction=str(row["instruction"]),
            mode=str(row["mode"]),
            status=str(row["status"]),
            total_attempts=int(row["total_attempts"]),
            total_successes=int(row["total_successes"]),
            sandbox_success_rate=(
                float(row["sandbox_success_rate"])
                if row["sandbox_success_rate"] is not None
                else None
            ),
            sandbox_average_score=(
                float(row["sandbox_average_score"])
                if row["sandbox_average_score"] is not None
                else None
            ),
            validation_status=str(row["validation_status"]),
            generation=int(row["generation"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def add_strategy(
        self,
        candidate: StrategyCandidate,
        embedding: np.ndarray,
        report: SandboxReport,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO strategies (
                strategy_key, name, instruction, mode, tags_json, source_hash,
                sandbox_success_rate, sandbox_average_score, generation,
                parent_ids_json, metadata_json, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'measured')
            """,
            (
                candidate.key,
                candidate.name,
                candidate.instruction,
                candidate.mode,
                json.dumps(candidate.tags),
                candidate.source_hash,
                report.success_rate,
                report.average_score,
                candidate.generation,
                json.dumps(candidate.parent_ids),
                json.dumps(candidate.metadata, sort_keys=True),
            ),
        )
        strategy_id = int(cursor.lastrowid)
        self.connection.execute(
            "INSERT INTO strategy_embeddings(strategy_id, vector_json) VALUES (?, ?)",
            (strategy_id, json.dumps(np.asarray(embedding, dtype=float).tolist())),
        )
        self.connection.commit()
        return strategy_id

    def add_released_strategy(
        self,
        candidate: StrategyCandidate,
        embedding: np.ndarray,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO strategies (
                strategy_key, name, instruction, mode, tags_json, source_hash,
                sandbox_success_rate, sandbox_average_score, generation,
                parent_ids_json, metadata_json, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, 'released_prevalidated')
            """,
            (
                candidate.key,
                candidate.name,
                candidate.instruction,
                candidate.mode,
                json.dumps(candidate.tags),
                candidate.source_hash,
                candidate.generation,
                json.dumps(candidate.parent_ids),
                json.dumps(candidate.metadata, sort_keys=True),
            ),
        )
        strategy_id = int(cursor.lastrowid)
        self.connection.execute(
            "INSERT INTO strategy_embeddings(strategy_id, vector_json) VALUES (?, ?)",
            (strategy_id, json.dumps(np.asarray(embedding, dtype=float).tolist())),
        )
        self.connection.commit()
        return strategy_id

    def strategy_by_key(self, key: str) -> StrategyRecord | None:
        row = self.connection.execute(
            "SELECT * FROM strategies WHERE strategy_key = ?", (key,)
        ).fetchone()
        return self._record(row) if row else None

    def strategy(self, strategy_id: int) -> StrategyRecord | None:
        row = self.connection.execute(
            "SELECT * FROM strategies WHERE id = ?", (strategy_id,)
        ).fetchone()
        return self._record(row) if row else None

    def active_strategies(self) -> list[StrategyRecord]:
        rows = self.connection.execute(
            "SELECT * FROM strategies WHERE status = 'active' ORDER BY id"
        ).fetchall()
        return [self._record(row) for row in rows]

    def ranked_active(self, limit: int) -> list[StrategyRecord]:
        rows = self.connection.execute(
            """
            SELECT * FROM strategies
            WHERE status = 'active'
            ORDER BY
                CASE WHEN total_attempts = 0 THEN COALESCE(sandbox_success_rate, 0.0)
                     ELSE CAST(total_successes AS REAL) / total_attempts END DESC,
                COALESCE(sandbox_success_rate, 0.0) DESC,
                COALESCE(sandbox_average_score, 0.0) DESC,
                id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._record(row) for row in rows]

    def embeddings(self) -> list[tuple[int, np.ndarray]]:
        rows = self.connection.execute(
            "SELECT strategy_id, vector_json FROM strategy_embeddings ORDER BY strategy_id"
        ).fetchall()
        return [
            (int(row["strategy_id"]), np.asarray(json.loads(row["vector_json"]), dtype=np.float32))
            for row in rows
        ]

    def record_attempt(
        self,
        scope: str,
        instance_id: str,
        goal: str,
        attempt: AttackAttempt,
    ) -> None:
        self.connection.execute(
            "INSERT INTO attempts(scope, instance_id, goal, attempt_json) VALUES (?, ?, ?, ?)",
            (scope, instance_id, goal, json.dumps(attempt.to_dict(), ensure_ascii=False)),
        )
        self.connection.execute(
            """
            UPDATE strategies
            SET total_attempts = total_attempts + 1,
                total_successes = total_successes + ?
            WHERE id = ?
            """,
            (int(attempt.success), attempt.strategy_id),
        )
        self.connection.commit()

    def store_success(
        self,
        scope: str,
        goal: str,
        goal_embedding: np.ndarray,
        sequence: Iterable[int],
        score: float,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO success_memory(
                scope, goal, goal_embedding_json, strategy_sequence_json, score
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                scope,
                goal,
                json.dumps(np.asarray(goal_embedding, dtype=float).tolist()),
                json.dumps(list(sequence)),
                float(score),
            ),
        )
        self.connection.commit()

    def success_memories(self, scope: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM success_memory WHERE scope = ? ORDER BY id", (scope,)
        ).fetchall()
        return [
            {
                "goal": str(row["goal"]),
                "embedding": np.asarray(json.loads(row["goal_embedding_json"]), dtype=np.float32),
                "sequence": tuple(json.loads(row["strategy_sequence_json"])),
                "score": float(row["score"]),
            }
            for row in rows
        ]

    def ensure_transitions(self, scope: str, strategy_ids: Iterable[int]) -> None:
        ids = sorted(set(int(item) for item in strategy_ids))
        if not ids:
            return
        initial = 1.0 / len(ids)
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO transitions(
                scope, from_strategy_id, to_strategy_id, value
            ) VALUES (?, ?, ?, ?)
            """,
            ((scope, left, right, initial) for left in ids for right in ids),
        )
        placeholders = ",".join("?" for _ in ids)
        self.connection.execute(
            f"""
            DELETE FROM transitions
            WHERE scope = ? AND (
                from_strategy_id NOT IN ({placeholders})
                OR to_strategy_id NOT IN ({placeholders})
            )
            """,
            (scope, *ids, *ids),
        )
        for strategy_id in ids:
            self._normalize_transition_row(scope, strategy_id)
        self.connection.commit()

    def _normalize_transition_row(self, scope: str, from_strategy_id: int) -> None:
        row = self.transition_row(scope, from_strategy_id)
        if not row:
            return
        values = {key: max(float(value), 0.0) for key, value in row.items()}
        total = sum(values.values())
        if total <= 0.0:
            normalized = {key: 1.0 / len(values) for key in values}
        else:
            normalized = {key: value / total for key, value in values.items()}
        self.connection.executemany(
            """
            UPDATE transitions SET value = ?
            WHERE scope = ? AND from_strategy_id = ? AND to_strategy_id = ?
            """,
            (
                (value, scope, from_strategy_id, to_strategy_id)
                for to_strategy_id, value in normalized.items()
            ),
        )

    def transition_row(self, scope: str, from_strategy_id: int) -> dict[int, float]:
        rows = self.connection.execute(
            """
            SELECT to_strategy_id, value FROM transitions
            WHERE scope = ? AND from_strategy_id = ?
            ORDER BY to_strategy_id
            """,
            (scope, from_strategy_id),
        ).fetchall()
        return {int(row["to_strategy_id"]): float(row["value"]) for row in rows}

    def max_transition(self, scope: str, from_strategy_id: int) -> float:
        row = self.connection.execute(
            """
            SELECT MAX(value) AS maximum FROM transitions
            WHERE scope = ? AND from_strategy_id = ?
            """,
            (scope, from_strategy_id),
        ).fetchone()
        return float(row["maximum"] or 0.0)

    def update_transition(
        self,
        scope: str,
        from_strategy_id: int,
        to_strategy_id: int,
        value: float,
    ) -> None:
        cursor = self.connection.execute(
            """
            UPDATE transitions SET value = ?
            WHERE scope = ? AND from_strategy_id = ? AND to_strategy_id = ?
            """,
            (max(float(value), 0.0), scope, from_strategy_id, to_strategy_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Cannot update a transition outside the active strategy matrix")
        self._normalize_transition_row(scope, from_strategy_id)
        self.connection.commit()

    def count_strategies(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM strategies").fetchone()
        return int(row["count"])
