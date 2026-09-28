from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from darwin_attack.config import ModelConfig
from darwin_attack.providers import OpenAICompatibleChatModel


class OpenAIResponseContractTest(unittest.TestCase):
    def generate(self, content, refusal=None):
        provider = OpenAICompatibleChatModel.__new__(OpenAICompatibleChatModel)
        provider.config = ModelConfig("openai_compatible", "fixture", 0.0, 128)
        provider.client = Mock()
        provider.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=refusal))],
        )
        return provider.generate([{"role": "user", "content": "Neutral protocol fixture."}])

    def test_refusal_only_response_is_retained(self):
        refusal = "I cannot assist with that request."
        self.assertEqual(self.generate(None, refusal), refusal)

    def test_refusal_and_content_are_both_retained_in_a_fixed_order(self):
        self.assertEqual(
            self.generate(" General explanation.\n", " I cannot provide that assistance. "),
            "I cannot provide that assistance.\n\nGeneral explanation.",
        )

    def test_content_only_and_empty_response_behavior_is_unchanged(self):
        self.assertEqual(self.generate(" Full response.\n"), "Full response.")
        self.assertEqual(self.generate(None), "")
        self.assertEqual(self.generate(""), "")

    def test_invalid_response_field_types_are_not_silently_discarded(self):
        for content, refusal in ((123, None), (None, {"text": "refusal"})):
            with self.subTest(content=content, refusal=refusal), self.assertRaises(ValueError):
                self.generate(content, refusal)


if __name__ == "__main__":
    unittest.main()
