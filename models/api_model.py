"""API model wrapper (OpenAI-compatible endpoints)."""
import time


class APIModel:
    """Wrapper for OpenAI-compatible API models."""

    def __init__(self, api_key: str, base_url: str, default_model: str = "gpt-5.4"):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The `openai` package is required for API-backed review and attack flows."
            ) from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model

    def generate(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> str:
        """Call the API with retry logic."""
        model = model or self.default_model
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"[APIModel] Retry {attempt+1}/{retries} after error: {e}")
                    time.sleep(wait)
                else:
                    print(f"[APIModel] Failed after {retries} attempts: {e}")
                    return ""

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        """Shortcut for judge calls with low temperature."""
        return self.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )
