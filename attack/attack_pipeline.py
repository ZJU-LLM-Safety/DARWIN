"""Main attack pipeline — history lookup → pool selection → attack → update."""
from __future__ import annotations

import time
from typing import Any

from config.prompts import CROSSOVER_PROMPT
from config.settings import (
    ATTACK_TARGET_DEVICE,
    ATTACK_TARGET_MODEL_PATH,
    CHAIN_COUNT,
    CHAIN_LENGTH,
    FUSED_STRATEGY_SANDBOX_ENABLED,
    REFLECTIVE_SANDBOX_ENABLED,
)
from database.chroma_db import ChromaDB
from database.sqlite_db import SQLiteDB
from evolution.gan_evolution import GANEvolution
from evolution.reflective_evolution import ReflectiveEvolution
from models.llm_manager import LLMManager
from models.local_model import LocalModel
from sandbox.validator import SandboxValidator
from strategy.strategy_pool import StrategyPool

from attack.history_memory import HistoryMemory
from attack.judge import Judge
from attack.markov_selector import MarkovSelector
from attack.prompt_generator import PromptGenerator


class AttackPipeline:
    """Main orchestrator for the actual attack-time process."""

    def __init__(
        self,
        sqlite: SQLiteDB,
        chroma: ChromaDB,
        *,
        reflective_sandbox_enabled: bool = REFLECTIVE_SANDBOX_ENABLED,
        fused_strategy_sandbox_enabled: bool = FUSED_STRATEGY_SANDBOX_ENABLED,
    ):
        self.db = sqlite
        self.vec = chroma
        self.mgr = LLMManager()
        self.pool = StrategyPool(sqlite, chroma)
        self.judge = Judge()
        self.prompt_gen = PromptGenerator()
        self.history = HistoryMemory(sqlite, chroma)
        self.markov = MarkovSelector(sqlite)
        self.reflective = ReflectiveEvolution(
            sqlite,
            chroma,
            sandbox_enabled=reflective_sandbox_enabled,
        )
        self.gan = GANEvolution(sqlite)
        self.reflective_sandbox_enabled = reflective_sandbox_enabled
        self.fused_strategy_sandbox_enabled = fused_strategy_sandbox_enabled
        self.fused_sandbox = SandboxValidator() if fused_strategy_sandbox_enabled else None
        self.target_model = ATTACK_TARGET_MODEL_PATH
        self.target = LocalModel(
            ATTACK_TARGET_MODEL_PATH,
            device=ATTACK_TARGET_DEVICE,
            max_new_tokens=1024,
        )

    def attack_question(self, question: str, verbose: bool = True) -> dict[str, Any]:
        active = self.pool.get_active()
        if not active:
            return {"success": False, "reason": "no_active_strategies", "score": 0.0}

        active_ids = [s["id"] for s in active]
        self.markov.sync_strategies(active_ids)

        best_result: dict[str, Any] = {
            "success": False,
            "score": 0.0,
            "chain": -1,
            "step": -1,
            "used_history": False,
        }

        for chain_idx in range(CHAIN_COUNT):
            result = self._run_chain(question, chain_idx, verbose)
            if result["success"]:
                best_result = result
                break
            if result["score"] > best_result["score"]:
                best_result = result

        self.markov.save()
        if self.gan.should_upgrade():
            upgraded = self.gan.upgrade()
            if verbose and upgraded:
                print(f"[Pipeline] GAN target upgraded to {upgraded}")
        return best_result

    def _run_chain(self, question: str, chain_idx: int, verbose: bool) -> dict[str, Any]:
        history_match = self.history.get_strategy_for_question(question)
        if history_match:
            strategy = self.pool.get(int(history_match["strategy_id"]))
            if strategy and strategy["status"] == "active":
                current = {
                    "strategy_id": strategy["id"],
                    "strategy_name": strategy["name"],
                    "strategy_text": strategy["text"],
                    "strategy_obj": strategy,
                    "used_history": True,
                    "history_match": history_match,
                    "fusion_parents": [strategy["id"]],
                }
            else:
                current = self._select_initial_candidate(question)
        else:
            current = self._select_initial_candidate(question)

        best: dict[str, Any] = {
            "success": False,
            "score": 0.0,
            "chain": chain_idx,
            "step": -1,
            "used_history": bool(history_match),
        }

        for step_idx in range(CHAIN_LENGTH):
            if verbose:
                print(
                    f"  [Chain {chain_idx}, Step {step_idx}] "
                    f"Strategy: {current['strategy_name']} "
                    f"(base_id={current['strategy_id']})",
                    flush=True,
                )

            result = self._single_attack(
                question=question,
                chain_idx=chain_idx,
                step_idx=step_idx,
                candidate=current,
            )
            result["used_history"] = current.get("used_history", False)
            if history_match:
                result["history_match"] = history_match

            if result["score"] > best["score"]:
                best = result
            if result["success"]:
                return best

            strategy_obj = current["strategy_obj"]
            if self.reflective_sandbox_enabled:
                reflective_result = self.reflective.reflect_and_evolve(
                    failed_prompt=result["disguised_prompt"],
                    refusal_response=result["response"],
                    original_strategy=strategy_obj,
                    target_model=self.target_model,
                    target_generator=self.target,
                    query_refusal_reason=True,
                    verbose=verbose,
                )
                if reflective_result is not None:
                    self.markov.sync_strategies([s["id"] for s in self.pool.get_active()])

            next_strategy_id = self.markov.select_next(strategy_obj["id"])
            next_strategy = self.pool.get(next_strategy_id)
            if next_strategy is None:
                break

            self.markov.update(strategy_obj["id"], next_strategy_id, reward=result["score"])
            fused_text = self._fuse_strategies(strategy_obj, next_strategy)
            current = {
                "strategy_id": next_strategy["id"],
                "strategy_name": f"Fused {strategy_obj['name']} + {next_strategy['name']}",
                "strategy_text": fused_text or next_strategy["text"],
                "strategy_obj": next_strategy,
                "used_history": False,
                "fusion_parents": [strategy_obj["id"], next_strategy["id"]],
            }

        return best

    def _select_initial_candidate(self, question: str) -> dict[str, Any]:
        del question
        ucb = self.pool.get_ucb_scores()
        strategy_id = self.markov.select_initial(ucb)
        strategy = self.pool.get(strategy_id)
        return {
            "strategy_id": strategy["id"],
            "strategy_name": strategy["name"],
            "strategy_text": strategy["text"],
            "strategy_obj": strategy,
            "used_history": False,
            "fusion_parents": [strategy["id"]],
        }

    def _single_attack(
        self,
        *,
        question: str,
        chain_idx: int,
        step_idx: int,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        sid = int(candidate["strategy_id"])
        disguised = self.prompt_gen.generate(candidate["strategy_text"], question)
        if not disguised:
            self.pool.record_attempt(sid, False, 0.0, question, {"failure": "empty_disguised"})
            return {"success": False, "score": 0.0, "strategy_id": sid, "chain": chain_idx, "step": step_idx}

        response = self.target.generate(
            [{"role": "user", "content": disguised}],
            max_new_tokens=1024,
        )
        if not response:
            self.pool.record_attempt(sid, False, 0.0, question, {"failure": "empty_response"})
            return {
                "success": False,
                "score": 0.0,
                "strategy_id": sid,
                "disguised_prompt": disguised,
                "response": "",
                "chain": chain_idx,
                "step": step_idx,
            }

        success, score, judge_output = self.judge.is_success(question, response)
        self.pool.record_attempt(
            sid,
            success,
            score,
            question,
            {
                "chain_idx": chain_idx,
                "step_idx": step_idx,
                "used_fused_variant": len(candidate.get("fusion_parents", [])) > 1,
                "fusion_parents": candidate.get("fusion_parents", []),
                "judge_output": judge_output,
            },
        )
        self.gan.record_attack(success)
        self.db.log_attack(
            question=question,
            strategy_id=sid,
            disguised_prompt=disguised,
            response=response,
            score=score,
            success=success,
            target_model=self.target_model,
            chain_idx=chain_idx,
            step_idx=step_idx,
            metadata={
                "judge_output": judge_output,
                "used_fused_variant": len(candidate.get("fusion_parents", [])) > 1,
                "fusion_parents": candidate.get("fusion_parents", []),
                "strategy_name": candidate["strategy_name"],
            },
            )

        if success:
            self._maybe_store_fused_success(candidate, sid, verbose=False)
            self.history.store_success(
                question,
                sid,
                disguised,
                score,
                self.target_model,
                metadata={
                    "chain_idx": chain_idx,
                    "step_idx": step_idx,
                    "strategy_name": candidate["strategy_name"],
                },
            )

        self.pool.prune()
        return {
            "success": success,
            "score": score,
            "strategy_id": sid,
            "strategy_name": candidate["strategy_name"],
            "disguised_prompt": disguised,
            "response": response,
            "judge_output": judge_output,
            "chain": chain_idx,
            "step": step_idx,
            "fusion_parents": candidate.get("fusion_parents", []),
        }

    def _maybe_store_fused_success(
        self,
        candidate: dict[str, Any],
        base_strategy_id: int,
        *,
        verbose: bool = False,
    ) -> None:
        parents = candidate.get("fusion_parents", [])
        if len(parents) <= 1:
            return
        if not self.fused_strategy_sandbox_enabled or self.fused_sandbox is None:
            return

        fused_text = candidate.get("strategy_text", "").strip()
        if not fused_text:
            return

        fused_name = candidate.get("strategy_name", "Fused Strategy")
        validation = self.fused_sandbox.validate_detailed(
            fused_text,
            strategy_name=fused_name,
            strategy_id=f"fused_candidate_{base_strategy_id}_{int(time.time())}",
            source_group="fused_success",
            source_path="",
            verbose=verbose,
        )
        if not validation["passed"]:
            return

        self.pool.add_strategy(
            name=fused_name,
            text=fused_text,
            tags="fused_success",
            source="fused_success",
            status="active",
            strategy_key=f"fused_success_{parents[0]}_{parents[-1]}_{int(time.time())}",
            source_group="fused_success",
            metadata={
                "origin": "successful_fused_strategy",
                "base_strategy_id": base_strategy_id,
                "validation": {
                    "success_count": validation["success_count"],
                    "trial_count": validation["trial_count"],
                    "avg_score": validation["avg_score"],
                },
            },
            sandbox_success_rate=validation["success_rate"],
            sandbox_avg_score=validation["avg_score"],
            generation=1 + max((self.pool.get(pid) or {}).get("generation", 0) for pid in parents),
            parent_strategy_ids=list(parents),
        )

    def _fuse_strategies(self, strategy_a: dict[str, Any], strategy_b: dict[str, Any]) -> str | None:
        win_a = strategy_a["total_successes"] / max(strategy_a["total_attempts"], 1)
        win_b = strategy_b["total_successes"] / max(strategy_b["total_attempts"], 1)
        prompt = CROSSOVER_PROMPT.format(
            win_rate_a=win_a,
            strategy_a=strategy_a["text"],
            win_rate_b=win_b,
            strategy_b=strategy_b["text"],
        )
        try:
            result = self.mgr.api_generate(prompt, model="gpt-5.4", max_tokens=1200).strip()
        except Exception:
            result = ""
        if not result:
            result = self.prompt_gen.model.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=1200,
            ).strip()
        return result if result else None
