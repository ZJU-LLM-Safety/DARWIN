from __future__ import annotations

import os
from typing import Any, Protocol, Sequence

from .config import ConfigurationError, ModelConfig


Message = dict[str, str]


class ChatModel(Protocol):
    def generate(self, messages: Sequence[Message]) -> str:
        pass


class PromptModel(Protocol):
    def generate_prompt(self, prompt: str) -> str:
        pass


class OpenAICompatibleChatModel:

    def __init__(self, config: ModelConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install DARWIN with the 'api' extra") from exc

        if not config.api_key_env:
            raise ConfigurationError("api_key_env is required for openai_compatible models")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ConfigurationError(f"Environment variable {config.api_key_env!r} is not set")
        base_url = None
        if config.base_url_env:
            base_url = os.environ.get(config.base_url_env)
            if not base_url:
                raise ConfigurationError(f"Environment variable {config.base_url_env!r} is not set")
        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages: Sequence[Message]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra,
        )
        message = response.choices[0].message
        content = message.content
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            ).strip()
        elif content is None:
            text = ""
        else:
            raise ValueError("Chat completion content must be text, text parts, or null")
        refusal = getattr(message, "refusal", None)
        if refusal is not None and not isinstance(refusal, str):
            raise ValueError("Chat completion refusal must be text or null")
        return "\n\n".join(part for part in ((refusal or "").strip(), text) if part)

    def generate_prompt(self, prompt: str) -> str:
        response = self.client.completions.create(
            model=self.config.model,
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra,
        )
        return str(response.choices[0].text or "").strip()


class TransformersChatModel:

    def __init__(self, config: ModelConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install DARWIN with the 'local' extra") from exc

        self.config = config
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=bool(config.extra.get("trust_remote_code", False)),
        )
        model_kwargs = dict(config.extra.get("model_kwargs") or {})
        if config.device:
            model_kwargs.setdefault("device_map", config.device)
        self.model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)

    def generate(self, messages: Sequence[Message]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        return self.generate_prompt(prompt)

    def generate_prompt(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        model_device = next(self.model.parameters()).device
        inputs = {name: value.to(model_device) for name, value in inputs.items()}
        do_sample = self.config.temperature > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = self.config.temperature
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()


def build_model(config: ModelConfig) -> ChatModel:
    if config.provider == "openai_compatible":
        return OpenAICompatibleChatModel(config)
    if config.provider == "transformers":
        return TransformersChatModel(config)
    raise ConfigurationError(f"Unsupported model provider: {config.provider}")
