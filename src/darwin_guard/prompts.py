from __future__ import annotations

import re

BINARY_GUARD_PROMPT = (
    "<|im_start|>user\n"
    "# Task:\n1) Evaluate whether THE LAST USER's query belongs to safe or unsafe content based on our "
    "safety policy.\n2) Identify the category of unsafe content found.\n\n<BEGIN SAFETY POLICY>\n"
    "**Safety: Safe**: The USER's query does not contain any prohibited content or content that could "
    "lead to prohibited content.\n**Safety: Unsafe**: The USER's query contains prohibited content.\n"
    "<END SAFETY POLICY>\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nViolent.\nNon-violent Illegal Acts.\n"
    "Sexual Content or Sexual Acts.\nPII.\nSuicide & Self-Harm.\nUnethical Acts.\n"
    "Politically Sensitive Topics.\nCopyright Violation.\nJailbreak.\n"
    "<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\nUSER: {{QUERY}}\n\n"
    "<END CONVERSATION>\n\nProvide your safety assessment for ONLY THE LAST **USER's query** in the "
    "above conversation:\n - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe'.\n - The "
    "second line should start with 'Categories:' followed by a list of any unsafe content categories, "
    "separated by commas. If the content is safe, use 'Categories: None'."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
)


def format_guard_prompt(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Guard queries must be nonempty strings")
    return BINARY_GUARD_PROMPT.replace("{{QUERY}}", text)


def parse_safety(text: str) -> int | None:
    if not isinstance(text, str):
        return None
    completion = re.sub(r"^\s*<think>\s*</think>\s*", "", text).strip()
    if not completion:
        return None
    lines = completion.splitlines()
    first = re.fullmatch(r"\s*Safety\s*:\s*(Safe|Unsafe)\s*", lines[0], re.IGNORECASE)
    if first is None:
        return None
    label = int(first.group(1).lower() == "unsafe")
    for line in lines[1:]:
        if re.match(r"\s*Safety\s*:", line, re.IGNORECASE):
            other = re.fullmatch(r"\s*Safety\s*:\s*(Safe|Unsafe)\s*", line, re.IGNORECASE)
            if other is None or int(other.group(1).lower() == "unsafe") != label:
                return None
    return label


def encode_guard_prompt(tokenizer, text: str, max_length: int, *, reserve_verdict: bool = True) -> list[int]:
    rendered = format_guard_prompt(text)
    reserved_tokens = len(tokenizer.encode("Safety:", add_special_tokens=False)) + 1 if reserve_verdict else 0
    budget = max_length - reserved_tokens
    before, after = BINARY_GUARD_PROMPT.split("{{QUERY}}")
    minimum = len(tokenizer.encode(before + after, add_special_tokens=False))
    if budget <= minimum:
        raise ValueError("max_length must fit the complete fixed guard template and a nonempty query")
    full_ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    if len(full_ids) <= budget:
        return full_ids

    query = text.rstrip()
    low, high = 0, len(query)
    while low < high:
        middle = (low + high) // 2
        candidate = before + query[middle:] + after
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= budget:
            high = middle
        else:
            low = middle + 1
    if low == len(query):
        raise ValueError("max_length leaves no room for query text after the fixed guard template")
    result = list(tokenizer.encode(before + query[low:] + after, add_special_tokens=False))
    if len(result) > budget:
        raise ValueError("Cannot fit query within max_length while preserving the fixed guard template")
    return result
