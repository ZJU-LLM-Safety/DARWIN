"""Sandbox validator using the real strategy sandbox runner logic."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from config.settings import (
    JUDGE_SUCCESS_THRESHOLD,
    SANDBOX_DATASET_PATH,
    SANDBOX_DATASET_SEED,
    SANDBOX_GENERATOR_MODEL_PATH,
    SANDBOX_JUDGE_MODEL,
    SANDBOX_JUDGE_TEMPLATE,
    SANDBOX_KEEP_SUCCESS_RATE,
    SANDBOX_QUESTIONS_COUNT,
    SANDBOX_SUFFIX,
    SANDBOX_TARGET_MODEL_PATH,
    SANDBOX_TRIALS_PER_QUESTION,
)
from scripts.run_strategy_sandbox_validation import (
    GoalItem,
    StrategyTemplate,
    build_generator,
    build_judge,
    build_target,
    evaluate_strategy,
    load_dataset_goals,
)


class SandboxValidator:
    """Validate strategy templates with the same stack as the batch sandbox run."""

    def __init__(
        self,
        *,
        dataset_path: str = SANDBOX_DATASET_PATH,
        dataset_limit: int = SANDBOX_QUESTIONS_COUNT,
        dataset_seed: int = SANDBOX_DATASET_SEED,
        trials_per_goal: int = SANDBOX_TRIALS_PER_QUESTION,
        generator_model_path: str = SANDBOX_GENERATOR_MODEL_PATH,
        target_model_path: str = SANDBOX_TARGET_MODEL_PATH,
        judge_model: str = SANDBOX_JUDGE_MODEL,
        judge_template: str = SANDBOX_JUDGE_TEMPLATE,
        success_threshold: float = JUDGE_SUCCESS_THRESHOLD,
        keep_success_rate: float = SANDBOX_KEEP_SUCCESS_RATE,
        suffix: str = SANDBOX_SUFFIX,
    ):
        self.dataset_path = dataset_path
        self.dataset_limit = dataset_limit
        self.dataset_seed = dataset_seed
        self.trials_per_goal = trials_per_goal
        self.generator_model_path = generator_model_path
        self.target_model_path = target_model_path
        self.judge_model = judge_model
        self.judge_template = judge_template
        self.success_threshold = success_threshold
        self.keep_success_rate = keep_success_rate
        self.suffix = suffix

        self._goals: list[GoalItem] | None = None
        self._fallback_goal_pool: list[GoalItem] | None = None
        self._generator = None
        self._target = None
        self._judge = None

    @property
    def goals(self) -> list[GoalItem]:
        if self._goals is None:
            self._goals = load_dataset_goals(
                Path(self.dataset_path),
                self.dataset_limit,
                self.dataset_seed,
            )
        return self._goals

    @property
    def fallback_goal_pool(self) -> list[GoalItem]:
        if self._fallback_goal_pool is None:
            self._fallback_goal_pool = load_dataset_goals(
                Path(self.dataset_path),
                0,
                self.dataset_seed,
            )
        return self._fallback_goal_pool

    @property
    def generator(self):
        if self._generator is None:
            self._generator = build_generator(self.generator_model_path, 384)
        return self._generator

    @property
    def target(self):
        if self._target is None:
            self._target = build_target(self.target_model_path, 384)
        return self._target

    @property
    def judge(self):
        if self._judge is None:
            self._judge = build_judge(self.judge_model)
        return self._judge

    def validate(
        self,
        strategy_text: str,
        *,
        strategy_name: str = "Candidate Strategy",
        strategy_id: str = "candidate",
        source_group: str = "internal_evolution",
        source_path: str = "",
        verbose: bool = False,
    ) -> tuple[bool, int, int]:
        result = self.validate_detailed(
            strategy_text,
            strategy_name=strategy_name,
            strategy_id=strategy_id,
            source_group=source_group,
            source_path=source_path,
            verbose=verbose,
        )
        return result["passed"], result["success_count"], result["trial_count"]

    def validate_detailed(
        self,
        strategy_text: str,
        *,
        strategy_name: str = "Candidate Strategy",
        strategy_id: str = "candidate",
        source_group: str = "internal_evolution",
        source_path: str = "",
        verbose: bool = False,
    ) -> dict[str, Any]:
        strategy = StrategyTemplate(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            source_group=source_group,
            source_path=source_path,
            template_text=strategy_text,
        )
        result = evaluate_strategy(
            strategy=strategy,
            goals=self.goals,
            trials_per_goal=self.trials_per_goal,
            generator=self.generator,
            target=self.target,
            judge=self.judge,
            suffix=self.suffix,
            judge_model=self.judge_model,
            judge_template=self.judge_template,
            success_threshold=self.success_threshold,
            target_label=self.target_model_path,
            fallback_goal_pool=self.fallback_goal_pool,
        )
        result["passed"] = result["success_rate"] >= self.keep_success_rate
        result["keep_success_rate"] = self.keep_success_rate
        result["validation_goals"] = [asdict(goal) for goal in self.goals]
        if verbose:
            print(
                f"[Sandbox] {strategy_name} "
                f"ASR={result['success_rate']:.2%} "
                f"avg_score={result['avg_score']:.3f} "
                f"threshold={self.keep_success_rate:.2%} "
                f"→ {'PASS' if result['passed'] else 'FAIL'}",
                flush=True,
            )
        return result
