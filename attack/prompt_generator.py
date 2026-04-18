"""Prompt generator — strategy template + harmful question → disguised prompt."""
from __future__ import annotations

from config.prompts import PROMPT_GENERATOR_TEMPLATE
from config.settings import (
    ATTACK_GENERATOR_DEVICE,
    ATTACK_GENERATOR_MODEL_PATH,
    ATTACK_PROMPT_SUFFIX,
)
from models.local_model import LocalModel
from scripts.run_strategy_sandbox_validation import (
    append_suffix,
    instantiate_template,
    normalize_generated_text,
)


class PromptGenerator:
    """Generate disguised prompts from either full templates or abstract strategies."""

    def __init__(
        self,
        model_path: str = ATTACK_GENERATOR_MODEL_PATH,
        suffix: str = ATTACK_PROMPT_SUFFIX,
    ):
        self.model = LocalModel(model_path, device=ATTACK_GENERATOR_DEVICE, max_new_tokens=384)
        self.suffix = suffix

    def generate(self, strategy_text: str, harmful_question: str) -> str:
        """Use the local generator to produce a disguised prompt."""
        if "{harmful_prompt}" in strategy_text:
            prompt = instantiate_template(strategy_text, harmful_question)
        else:
            prompt = PROMPT_GENERATOR_TEMPLATE.format(
                strategy_text=strategy_text,
                harmful_question=harmful_question,
            )
        disguised = normalize_generated_text(
            self.model.generate([{"role": "user", "content": prompt}])
        )
        return append_suffix(disguised, self.suffix).strip()
