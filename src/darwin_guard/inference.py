from __future__ import annotations


def build_filter_model(settings: dict, dtype: str):
    if settings.get("provider") == "transformers":
        return GemmaIntentModel(settings, dtype)
    from darwin_attack.config import ModelConfig
    from darwin_attack.providers import build_model
    settings = {key: value for key, value in settings.items() if key != "identity"}
    return build_model(ModelConfig.from_dict(settings, "models.filter"))


class GemmaIntentModel:

    def __init__(self, settings: dict, dtype: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.settings = settings
        extra = settings.get("extra") or {}
        kwargs = dict(extra.get("model_kwargs") or {})
        precision = kwargs.pop("torch_dtype", kwargs.pop("dtype", dtype))
        kwargs["dtype"] = getattr(torch, precision) if isinstance(precision, str) else precision
        kwargs.setdefault("device_map", settings.get("device") or "auto")
        self.processor = AutoProcessor.from_pretrained(settings["model"])
        self.model = AutoModelForImageTextToText.from_pretrained(settings["model"], **kwargs)
        self.model.eval()

    def generate(self, messages: list[dict[str, str]]) -> str:
        conversation = [
            {"role": item["role"], "content": [{"type": "text", "text": item["content"]}]}
            for item in messages
        ]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
        ).to(self.model.device)
        prefix_length = inputs["input_ids"].shape[-1]
        kwargs = {
            "max_new_tokens": self.settings["max_tokens"],
            "do_sample": self.settings["temperature"] > 0,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = self.settings["temperature"]
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        return self.processor.decode(generated[0, prefix_length:], skip_special_tokens=True).strip()


class HFGuard:
    def __init__(self, model_path: str, training: dict, inference: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.max_length = training["max_length"]
        self.max_new_tokens = inference["max_new_tokens"]
        self.device = training["device"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, training["dtype"]),
        ).to(self.device)
        self.model.eval()

    def predict(self, text: str, *, max_length: int | None = None) -> int | None:
        from .prompts import encode_guard_prompt, parse_safety
        limit = self.max_length if max_length is None else max_length
        if type(limit) is not int or limit < 2:
            raise ValueError("Guard input length must be an integer of at least 2 tokens")
        tokens = encode_guard_prompt(self.tokenizer, text, limit, reserve_verdict=max_length is None)
        ids = self.torch.tensor([tokens], device=self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids=ids, attention_mask=self.torch.ones_like(ids),
                max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(generated[0, len(tokens):], skip_special_tokens=True)
        return parse_safety(answer)
