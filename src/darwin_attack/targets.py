from __future__ import annotations

import re
from typing import Protocol

from .providers import ChatModel
from .schemas import TargetObservation
from .guardrail_templates import (
    QWEN_TEMPLATE, YUFENG_TEMPLATE,
    format_guard_prompt, format_yufeng_prompt, parse_builtin_guard_output,
    validate_custom_template, yufeng_tokenizer_template,
)


class Target(Protocol):
    def query(self, prompt: str) -> TargetObservation:
        pass


class LLMTarget:
    def __init__(self, model: ChatModel):
        self.model = model

    def query(self, prompt: str) -> TargetObservation:
        output = self.model.generate([{"role": "user", "content": prompt}])
        return TargetObservation(output=output)


class GuardrailTarget:
    def __init__(
        self, model: ChatModel, safe_pattern: str | None = None,
        unsafe_pattern: str | None = None, *, template: str,
        template_text: str | None = None, template_id: str | None = None,
    ):
        self.model = model
        self.template = template
        self.template_text = template_text
        if template not in {QWEN_TEMPLATE, YUFENG_TEMPLATE, "custom", "prewrapped_endpoint"}:
            raise ValueError("Select an explicit guardrail input template")
        self.template_id = template if template in {QWEN_TEMPLATE, YUFENG_TEMPLATE} else template_id
        if not self.template_id or not self.template_id.strip():
            raise ValueError("Custom and prewrapped guardrails require a template_id")
        if template != "prewrapped_endpoint" and not callable(getattr(model, "generate_prompt", None)):
            raise ValueError("Rendered guardrail templates require a raw-prompt completion provider")
        self.metadata = {"template": template, "template_id": self.template_id}
        if template == YUFENG_TEMPLATE:
            tokenizer = getattr(model, "tokenizer", None)
            yufeng_tokenizer_template(tokenizer)
            self.metadata.update({
                "policy": "model_default", "reason_first": False,
            })
        elif template == "custom":
            validate_custom_template(template_text)
        elif template == "prewrapped_endpoint":
            self.metadata["template_verified_locally"] = False
        self.metadata["transport"] = (
            "endpoint_chat_wrapper" if template == "prewrapped_endpoint" else "raw_prompt_completion"
        )
        builtin = self.template_id in {QWEN_TEMPLATE, YUFENG_TEMPLATE}
        if not builtin and not (safe_pattern and unsafe_pattern):
            raise ValueError("Custom guardrail output requires safe_pattern and unsafe_pattern")
        if builtin and (safe_pattern or unsafe_pattern):
            raise ValueError("Built-in guardrails use their paper output parser, not regex overrides")
        self.safe_pattern = re.compile(safe_pattern, re.IGNORECASE) if safe_pattern else None
        self.unsafe_pattern = re.compile(unsafe_pattern, re.IGNORECASE) if unsafe_pattern else None

    def query(self, prompt: str) -> TargetObservation:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Guard queries must be nonempty strings")
        if self.template == "prewrapped_endpoint":
            output = self.model.generate([{"role": "user", "content": prompt}]).strip()
        else:
            if self.template == QWEN_TEMPLATE:
                rendered = format_guard_prompt(prompt)
            elif self.template == YUFENG_TEMPLATE:
                rendered = format_yufeng_prompt(self.model.tokenizer, prompt)
            else:
                rendered = self.template_text.replace("{{QUERY}}", prompt)
            output = self.model.generate_prompt(rendered).strip()
        if self.template_id in {QWEN_TEMPLATE, YUFENG_TEMPLATE}:
            decision = parse_builtin_guard_output(output, self.template_id)
        else:
            unsafe = self.unsafe_pattern.search(output)
            safe = self.safe_pattern.search(output)
            decision = "unsafe" if unsafe and not safe else "safe" if safe and not unsafe else "unparseable"
        return TargetObservation(output=output, decision=decision, raw={"guardrail": dict(self.metadata)})
