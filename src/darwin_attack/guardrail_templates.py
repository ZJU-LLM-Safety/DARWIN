from __future__ import annotations

from darwin_guard.prompts import format_guard_prompt, parse_safety


QWEN_TEMPLATE = "qwen3guard_binary"
YUFENG_TEMPLATE = "yufeng_xguard"
YUFENG_POLICY_IDS = frozenset(
    "pc dc dw pi ec ac def ti cy ph mh se sci pp cs acc mc ha ps ter sd ext fin med law cm ma md".split()
)


def validate_custom_template(template: str) -> None:
    if not isinstance(template, str) or template.count("{{QUERY}}") != 1:
        raise ValueError("Custom guardrail template must contain exactly one {{QUERY}} placeholder")
    if not template.replace("{{QUERY}}", "").strip():
        raise ValueError("Custom guardrail template must include the model's classification instructions")


def yufeng_tokenizer_template(tokenizer) -> str:
    risk_ids = set((getattr(tokenizer, "init_kwargs", {}) or {}).get("id2risk", {}))
    if risk_ids != YUFENG_POLICY_IDS | {"sec"}:
        raise ValueError("YuFeng requires its official tokenizer with the complete id2risk taxonomy")
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template.strip():
        raise ValueError("YuFeng tokenizer has no model-specific classification template")
    if "reason_first" not in template or "policy" not in template:
        raise ValueError("YuFeng tokenizer template must support policy and reason_first")
    return template


def format_yufeng_prompt(tokenizer, text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Guard queries must be nonempty strings")
    yufeng_tokenizer_template(tokenizer)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        policy=None, reason_first=False, tokenize=False,
    )


def parse_builtin_guard_output(output: str, template_id: str) -> str:
    if template_id == QWEN_TEMPLATE:
        label = parse_safety(output)
        return {0: "safe", 1: "unsafe"}.get(label, "unparseable")
    if template_id == YUFENG_TEMPLATE:
        lines = output.strip().splitlines()
        risk_id = lines[0].strip() if lines else ""
        if risk_id == "sec":
            return "safe"
        if risk_id in YUFENG_POLICY_IDS:
            return "unsafe"
        return "unparseable"
    raise ValueError(f"No built-in output parser for guardrail template {template_id!r}")
