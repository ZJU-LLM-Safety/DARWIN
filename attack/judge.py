"""Judge wrapper aligned with the real sandbox scoring setup."""
from __future__ import annotations

from config.settings import (
    ATTACK_JUDGE_MODEL,
    ATTACK_JUDGE_TEMPLATE,
    JUDGE_SUCCESS_THRESHOLD,
)
from models.api_model import APIModel
from config.settings import API_BASE_URL, API_SECRET_KEY
from scripts.run_strategy_sandbox_validation import (
    build_judge_messages,
    parse_judge_score,
)


class Judge:
    """Evaluate whether a target model response is jailbroken."""

    def __init__(
        self,
        model_name: str = ATTACK_JUDGE_MODEL,
        template_name: str = ATTACK_JUDGE_TEMPLATE,
        success_threshold: float = JUDGE_SUCCESS_THRESHOLD,
    ):
        self.model_name = model_name
        self.template_name = template_name
        self.success_threshold = success_threshold
        self.api = APIModel(API_SECRET_KEY, API_BASE_URL, model_name)

    def score(self, goal: str, response: str) -> tuple[float, str]:
        messages = build_judge_messages(self.template_name, goal, response)
        raw = self.api.generate(
            messages=messages,
            model=self.model_name,
            temperature=0.0,
            max_tokens=512 if self.template_name == "markov_policy" else 100,
        ).strip()
        parsed = parse_judge_score(self.template_name, raw)
        return (parsed or 0.0), raw

    def is_success(self, goal: str, response: str) -> tuple[bool, float, str]:
        score, raw = self.score(goal, response)
        return score >= self.success_threshold, score, raw
