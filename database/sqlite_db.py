"""SQLite database for strategies, attack logs, and pool metadata."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from config.settings import SQLITE_DB_PATH


STRATEGY_EXTRA_COLUMNS: dict[str, str] = {
    "strategy_key": "TEXT",
    "source_group": "TEXT DEFAULT ''",
    "source_path": "TEXT DEFAULT ''",
    "metadata_json": "TEXT DEFAULT '{}'",
    "consecutive_failures": "INTEGER DEFAULT 0",
    "sandbox_success_rate": "REAL DEFAULT 0.0",
    "sandbox_avg_score": "REAL DEFAULT 0.0",
    "last_sandbox_run_id": "TEXT DEFAULT ''",
    "generation": "INTEGER DEFAULT 0",
    "parent_strategy_ids_json": "TEXT DEFAULT '[]'",
}

ATTEMPT_EXTRA_COLUMNS: dict[str, str] = {
    "metadata_json": "TEXT DEFAULT '{}'",
}

ATTACK_LOG_EXTRA_COLUMNS: dict[str, str] = {
    "metadata_json": "TEXT DEFAULT '{}'",
}

SUCCESS_HISTORY_EXTRA_COLUMNS: dict[str, str] = {
    "metadata_json": "TEXT DEFAULT '{}'",
}


class SQLiteDB:
    """Manages all structured data: strategies, attempts, logs, and state."""

    def __init__(self, db_path: str = SQLITE_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._ensure_columns("strategies", STRATEGY_EXTRA_COLUMNS)
        self._ensure_columns("strategy_attempts", ATTEMPT_EXTRA_COLUMNS)
        self._ensure_columns("attack_log", ATTACK_LOG_EXTRA_COLUMNS)
        self._ensure_columns("success_history", SUCCESS_HISTORY_EXTRA_COLUMNS)

    def _create_tables(self):
        c = self.conn.cursor()
        c.executescript(
            """
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            text TEXT NOT NULL,
            tags TEXT DEFAULT '',
            source TEXT DEFAULT 'seed',
            status TEXT DEFAULT 'active',
            total_attempts INTEGER DEFAULT 0,
            total_successes INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS strategy_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            success INTEGER NOT NULL,
            score REAL DEFAULT 0.0,
            question TEXT DEFAULT '',
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );

        CREATE TABLE IF NOT EXISTS attack_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            strategy_id INTEGER,
            disguised_prompt TEXT,
            response TEXT,
            score REAL DEFAULT 0.0,
            success INTEGER DEFAULT 0,
            target_model TEXT DEFAULT '',
            chain_idx INTEGER DEFAULT 0,
            step_idx INTEGER DEFAULT 0,
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS success_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            strategy_id INTEGER NOT NULL,
            disguised_prompt TEXT,
            score REAL DEFAULT 0.0,
            target_model TEXT DEFAULT '',
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS markov_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matrix_json TEXT NOT NULL,
            q_table_json TEXT NOT NULL,
            strategy_ids_json TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS gan_progression (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            activated_at TEXT DEFAULT (datetime('now')),
            total_attacks INTEGER DEFAULT 0,
            total_successes INTEGER DEFAULT 0
        );
        """
        )
        self.conn.commit()

    def _ensure_columns(self, table_name: str, columns: dict[str, str]):
        existing = {
            row["name"] for row in self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_sql in columns.items():
            if column_name in existing:
                continue
            self.conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
            )
        self.conn.commit()

    @staticmethod
    def _decode_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        if "metadata_json" in item:
            try:
                item["metadata"] = json.loads(item["metadata_json"] or "{}")
            except json.JSONDecodeError:
                item["metadata"] = {}
        if "parent_strategy_ids_json" in item:
            try:
                item["parent_strategy_ids"] = json.loads(
                    item["parent_strategy_ids_json"] or "[]"
                )
            except json.JSONDecodeError:
                item["parent_strategy_ids"] = []
        return item

    def _decode_rows(self, rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        return [self._decode_row(row) for row in rows if row is not None]

    # ── Strategy CRUD ─────────────────────────────────────────────
    def add_strategy(
        self,
        *,
        name: str,
        text: str,
        tags: str = "",
        source: str = "seed",
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
    ) -> int:
        payload = (
            name,
            text,
            tags,
            source,
            status,
            strategy_key,
            source_group,
            source_path,
            json.dumps(metadata or {}, ensure_ascii=False),
            sandbox_success_rate,
            sandbox_avg_score,
            last_sandbox_run_id,
            generation,
            json.dumps(parent_strategy_ids or []),
        )
        c = self.conn.execute(
            """
            INSERT INTO strategies (
                name, text, tags, source, status, strategy_key, source_group,
                source_path, metadata_json, sandbox_success_rate,
                sandbox_avg_score, last_sandbox_run_id, generation,
                parent_strategy_ids_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            payload,
        )
        self.conn.commit()
        return c.lastrowid

    def get_strategy(self, strategy_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM strategies WHERE id=?", (strategy_id,)
        ).fetchone()
        return self._decode_row(row)

    def get_strategy_by_key(self, strategy_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM strategies WHERE strategy_key=?",
            (strategy_key,),
        ).fetchone()
        return self._decode_row(row)

    def get_active_strategies(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM strategies WHERE status='active' ORDER BY id"
        ).fetchall()
        return self._decode_rows(rows)

    def get_all_strategies(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM strategies ORDER BY id").fetchall()
        return self._decode_rows(rows)

    def update_strategy_status(self, strategy_id: int, status: str):
        self.conn.execute(
            "UPDATE strategies SET status=?, updated_at=datetime('now') WHERE id=?",
            (status, strategy_id),
        )
        self.conn.commit()

    def update_strategy_sandbox_metrics(
        self,
        strategy_id: int,
        *,
        success_rate: float,
        avg_score: float,
        run_id: str = "",
    ):
        self.conn.execute(
            """
            UPDATE strategies
            SET sandbox_success_rate=?, sandbox_avg_score=?, last_sandbox_run_id=?,
                updated_at=datetime('now')
            WHERE id=?
            """,
            (success_rate, avg_score, run_id, strategy_id),
        )
        self.conn.commit()

    def strategy_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM strategies"
        ).fetchone()[0]

    # ── Attempt Tracking ──────────────────────────────────────────
    def record_attempt(
        self,
        strategy_id: int,
        success: bool,
        score: float = 0.0,
        question: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        self.conn.execute(
            """
            INSERT INTO strategy_attempts (strategy_id, success, score, question, metadata_json)
            VALUES (?,?,?,?,?)
            """,
            (strategy_id, int(success), score, question, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        if success:
            self.conn.execute(
                """
                UPDATE strategies
                SET total_attempts = total_attempts + 1,
                    total_successes = total_successes + 1,
                    consecutive_failures = 0,
                    status = 'active',
                    updated_at=datetime('now')
                WHERE id=?
                """,
                (strategy_id,),
            )
        else:
            self.conn.execute(
                """
                UPDATE strategies
                SET total_attempts = total_attempts + 1,
                    consecutive_failures = consecutive_failures + 1,
                    updated_at=datetime('now')
                WHERE id=?
                """,
                (strategy_id,),
            )
        self.conn.commit()

    def get_recent_attempts(self, strategy_id: int, n: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM strategy_attempts
            WHERE strategy_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (strategy_id, n),
        ).fetchall()
        return self._decode_rows(rows)

    def get_recent_success_count(self, strategy_id: int, n: int = 20) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT success FROM strategy_attempts
              WHERE strategy_id=?
              ORDER BY id DESC
              LIMIT ?
            ) WHERE success=1
            """,
            (strategy_id, n),
        ).fetchone()
        return row[0] if row else 0

    # ── Attack Log ────────────────────────────────────────────────
    def log_attack(
        self,
        question: str,
        strategy_id: int,
        disguised_prompt: str,
        response: str,
        score: float,
        success: bool,
        target_model: str = "",
        chain_idx: int = 0,
        step_idx: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        self.conn.execute(
            """
            INSERT INTO attack_log (
                question, strategy_id, disguised_prompt, response, score,
                success, target_model, chain_idx, step_idx, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                question,
                strategy_id,
                disguised_prompt,
                response,
                score,
                int(success),
                target_model,
                chain_idx,
                step_idx,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    # ── Success History ───────────────────────────────────────────
    def add_success_history(
        self,
        question: str,
        strategy_id: int,
        disguised_prompt: str = "",
        score: float = 0.0,
        target_model: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        self.conn.execute(
            """
            INSERT INTO success_history (
                question, strategy_id, disguised_prompt, score, target_model, metadata_json
            ) VALUES (?,?,?,?,?,?)
            """,
            (
                question,
                strategy_id,
                disguised_prompt,
                score,
                target_model,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def get_success_history(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM success_history ORDER BY id").fetchall()
        return self._decode_rows(rows)

    # ── Markov State ──────────────────────────────────────────────
    def save_markov_state(self, matrix: list, q_table: list, strategy_ids: list[int]):
        self.conn.execute(
            """
            INSERT INTO markov_state (matrix_json, q_table_json, strategy_ids_json)
            VALUES (?,?,?)
            """,
            (json.dumps(matrix), json.dumps(q_table), json.dumps(strategy_ids)),
        )
        self.conn.commit()

    def load_markov_state(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM markov_state ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["matrix"] = json.loads(d["matrix_json"])
        d["q_table"] = json.loads(d["q_table_json"])
        d["strategy_ids"] = json.loads(d["strategy_ids_json"])
        return d

    # ── GAN Progression ───────────────────────────────────────────
    def add_gan_model(self, model_name: str) -> int:
        c = self.conn.execute(
            "INSERT INTO gan_progression (model_name) VALUES (?)",
            (model_name,),
        )
        self.conn.commit()
        return c.lastrowid

    def get_current_gan_model(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM gan_progression ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def update_gan_stats(self, model_id: int, attacks: int, successes: int):
        self.conn.execute(
            """
            UPDATE gan_progression
            SET total_attacks=total_attacks+?, total_successes=total_successes+?
            WHERE id=?
            """,
            (attacks, successes, model_id),
        )
        self.conn.commit()

    # ── Utility ───────────────────────────────────────────────────
    def close(self):
        self.conn.close()
