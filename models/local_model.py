"""Local model loading and inference (Gemma-4-31B-it)."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LocalModel:
    """Wrapper for local HuggingFace models."""

    def __init__(self, model_path: str, device: str = "auto", max_new_tokens: int = 512):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._pipe = None
        self._device = device

    def _load(self):
        if self._model is not None:
            return
        print(f"[LocalModel] Loading {self.model_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        load_device = self._device
        if load_device == "auto":
            if not torch.cuda.is_available():
                load_device = "cpu"
            elif torch.cuda.device_count() == 1:
                # In our worker setup a single visible GPU is the common case.
                # Pinning 7B models to the visible cuda:0 is more stable there.
                load_device = {"": 0}
            else:
                # When multiple GPUs are intentionally exposed, allow accelerate
                # to shard the model across them.
                load_device = "auto"
        elif isinstance(load_device, str) and load_device.startswith("cuda:"):
            load_device = {"": int(load_device.split(":", 1)[1])}
        elif isinstance(load_device, int):
            load_device = {"": int(load_device)}
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=load_device,
            trust_remote_code=True,
        )
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        print(f"[LocalModel] {self.model_path} loaded.")

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        """Generate a response from a chat-style message list."""
        self._load()
        result = self._pipe(
            messages,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
        # Extract the assistant reply from pipeline output
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            # Chat pipeline returns list of messages
            return generated[-1]["content"]
        # Plain text pipeline
        return generated
