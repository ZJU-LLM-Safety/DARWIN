"""Reflective evolution — learn from failures to create improved strategies."""
from __future__ import annotations

import re
import time
from typing import Any

from config.prompts import REFLECTIVE_EVOLUTION_PROMPT, REFUSAL_REASON_PROMPT
from config.settings import (
    ATTACK_GENERATOR_MODEL_PATH,
    ATTACK_REFLECTION_DEVICE,
    ATTACK_TARGET_DEVICE,
    ATTACK_TARGET_MODEL_PATH,
    REFLECTIVE_SANDBOX_ENABLED,
)
from database.chroma_db import ChromaDB
from database.sqlite_db import SQLiteDB
from models.llm_manager import LLMManager
from models.local_model import LocalModel
from sandbox.validator import SandboxValidator
from strategy.strategy_pool import StrategyPool


class ReflectiveEvolution:
    """Analyze failed attacks and generate improved strategies via reflection."""

    def __init__(self, sqlite: SQLiteDB, chroma: ChromaDB, *, sandbox_enabled: bool = REFLECTIVE_SANDBOX_ENABLED):
        self.pool = StrategyPool(sqlite, chroma)
        self.mgr = LLMManager()
        self.sandbox_enabled = sandbox_enabled
        self.sandbox = SandboxValidator() if sandbox_enabled else None
        self.local_fallback = LocalModel(
            ATTACK_GENERATOR_MODEL_PATH,
            device=ATTACK_REFLECTION_DEVICE,
            max_new_tokens=1200,
        )
        self.target_local = LocalModel(
            ATTACK_TARGET_MODEL_PATH,
            device=ATTACK_TARGET_DEVICE,
            max_new_tokens=256,
        )

    def reflect_and_evolve(
        self,
        *,
        failed_prompt: str,
        refusal_response: str,
        original_strategy: dict[str, Any],
        target_model: str = ATTACK_TARGET_MODEL_PATH,
        target_generator: LocalModel | None = None,
        query_refusal_reason: bool = True,
        verbose: bool = True,
    ) -> dict[str, Any] | None:
        refusal_reason = ""
        if query_refusal_reason:
            refusal_reason = self._query_refusal_reason(
                failed_prompt=failed_prompt,
                refusal_response=refusal_response,
                target_model=target_model,
                target_generator=target_generator,
            )

        prompt = REFLECTIVE_EVOLUTION_PROMPT.format(
            failed_prompt=failed_prompt,
            refusal_response=refusal_response,
            refusal_reason=refusal_reason,
            original_strategy=original_strategy["text"],
        )

        raw_output = self._generate_reflection(prompt)
        reasoning = self._extract_tag(raw_output, "Reasoning")
        new_strategy_text = self._extract_tag(raw_output, "New_Strategy")

        if not new_strategy_text:
            if verbose:
                print("[Reflective] Failed to extract new strategy from reflection output.")
            return None

        if not self.sandbox_enabled or self.sandbox is None:
            if verbose:
                print("[Reflective] Sandbox gate disabled; reflective strategy not inserted into pool.")
            return None

        validation = self.sandbox.validate_detailed(
            new_strategy_text,
            strategy_name=f"Reflective {original_strategy['name']}",
            strategy_id=f"reflective_candidate_{int(time.time())}",
            source_group="reflective",
            verbose=verbose,
        )
        if not validation["passed"]:
            if verbose:
                print(
                    f"[Reflective] New strategy failed sandbox "
                    f"ASR={validation['success_rate']:.2%}",
                )
            return None

        sid = self.pool.add_strategy(
            name=f"Reflective {original_strategy['name']}",
            text=new_strategy_text,
            tags="reflective,evolved",
            source="reflective",
            status="active",
            strategy_key=f"reflective_{original_strategy['id']}_{int(time.time())}",
            source_group="reflective",
            metadata={
                "reasoning": reasoning,
                "refusal_reason": refusal_reason,
                "parent_strategy_id": original_strategy["id"],
                "validation": {
                    "success_count": validation["success_count"],
                    "trial_count": validation["trial_count"],
                    "avg_score": validation["avg_score"],
                },
            },
            sandbox_success_rate=validation["success_rate"],
            sandbox_avg_score=validation["avg_score"],
            generation=int(original_strategy.get("generation", 0)) + 1,
            parent_strategy_ids=[original_strategy["id"]],
        )
        if sid is None:
            if verbose:
                print("[Reflective] New strategy was duplicate, skipped.")
            return None

        result = {
            "id": sid,
            "name": f"Reflective {original_strategy['name']}",
            "text": new_strategy_text,
            "reasoning": reasoning,
            "refusal_reason": refusal_reason,
            "validation": validation,
        }
        if verbose:
            print(
                f"[Reflective] Added evolved strategy id={sid} "
                f"ASR={validation['success_rate']:.2%}",
            )
        return result

    def _generate_reflection(self, prompt: str) -> str:
        try:
            raw = self.mgr.api_generate(prompt, model="gpt-5.4", max_tokens=1200)
            if raw and "<New_Strategy>" in raw:
                return raw
        except Exception:
            pass

        return self.local_fallback.generate([{"role": "user", "content": prompt}], max_new_tokens=1200)

    def _query_refusal_reason(
        self,
        *,
        failed_prompt: str,
        refusal_response: str,
        target_model: str,
        target_generator: LocalModel | None = None,
    ) -> str:
        try:
            follow_up = (
                f"{REFUSAL_REASON_PROMPT}\n\n"
                f"Previous prompt:\n{failed_prompt}\n\n"
                f"Previous refusal:\n{refusal_response}"
            )
            if target_model.startswith("/"):
                local_target = target_generator or self.target_local
                return local_target.generate(
                    [{"role": "user", "content": follow_up}],
                    max_new_tokens=200,
                )
            return self.mgr.api_generate(follow_up, model=target_model, max_tokens=200)
        except Exception:
            return ""

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
