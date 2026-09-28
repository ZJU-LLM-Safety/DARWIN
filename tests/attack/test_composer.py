from __future__ import annotations

import unittest

from darwin_attack.composer import PromptComposer
from darwin_attack.schemas import StrategyRecord

from fakes import NeverUsedModel, StaticModel


def strategy(instruction: str, mode: str, metadata: dict) -> StrategyRecord:
    return StrategyRecord(
        id=1,
        key="strategy",
        name="Strategy",
        instruction=instruction,
        mode=mode,
        status="active",
        total_attempts=0,
        total_successes=0,
        sandbox_success_rate=None,
        sandbox_average_score=None,
        validation_status="released_prevalidated",
        generation=0,
        metadata=metadata,
    )


class ComposerTest(unittest.TestCase):
    def test_direct_template_skips_generator_and_appends_contract(self):
        composer = PromptComposer(NeverUsedModel())
        prompt = composer.apply(
            strategy(
                "Wrapper: {input}",
                "template",
                {
                    "special_executor": "direct_template",
                    "target_output_contract": "Return the labeled result.",
                },
            ),
            "original",
            "current",
        )
        self.assertEqual(prompt, "Wrapper: current\n\nReturn the labeled result.")

    def test_reverse_executor_uses_generator_for_inversion(self):
        model = StaticModel("safe opposite")
        prompt = PromptComposer(model).apply(
            strategy("unused", "instruction", {"special_executor": "reverse_method"}),
            "original",
            "current",
        )
        self.assertIn("safe opposite", prompt)
        self.assertEqual(len(model.messages), 1)


if __name__ == "__main__":
    unittest.main()
