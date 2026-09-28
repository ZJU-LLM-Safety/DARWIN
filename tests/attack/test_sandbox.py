from __future__ import annotations

import unittest

from darwin_attack.composer import PromptComposer
from darwin_attack.sandbox import SandboxValidator
from darwin_attack.schemas import StrategyCandidate, TargetObservation

from fakes import KeywordResponseJudge, NeverUsedModel


class RecordingTarget:
    def __init__(self, output: str = "failure"):
        self.output = output
        self.prompts = []

    def query(self, prompt: str) -> TargetObservation:
        self.prompts.append(prompt)
        return TargetObservation(output=self.output)


class SandboxTest(unittest.TestCase):
    def test_all_candidates_share_one_fixed_goal_subset(self):
        target = RecordingTarget()
        validator = SandboxValidator(
            PromptComposer(NeverUsedModel()),
            target,
            KeywordResponseJudge(),
            ["goal-a", "goal-b", "goal-c"],
            goals_per_candidate=2,
            trials_per_goal=1,
            random_seed=7,
        )
        validator.validate(
            StrategyCandidate("alpha", "Alpha", "alpha {input}", "template")
        )
        validator.validate(
            StrategyCandidate("beta", "Beta", "beta {input}", "template")
        )
        first_goals = [prompt.removeprefix("alpha ") for prompt in target.prompts[:2]]
        second_goals = [prompt.removeprefix("beta ") for prompt in target.prompts[2:]]
        self.assertEqual(first_goals, second_goals)

    def test_sandbox_judges_complete_response_including_prefix(self):
        target = RecordingTarget("context\n[[Answer]]: success")
        validator = SandboxValidator(
            PromptComposer(NeverUsedModel()),
            target,
            KeywordResponseJudge(),
            ["goal"],
            goals_per_candidate=1,
            trials_per_goal=1,
            random_seed=7,
        )
        report = validator.validate(
            StrategyCandidate(
                "marked",
                "Marked",
                "wrapped {input}",
                "template",
                metadata={
                    "response_extractor": "after_marker",
                    "response_markers": ["[[Answer]]"],
                },
            )
        )
        self.assertEqual(report.success_rate, 0.0)


    def test_missing_legacy_marker_does_not_bypass_the_judge(self):
        validator = SandboxValidator(
            PromptComposer(NeverUsedModel()),
            RecordingTarget("success"),
            KeywordResponseJudge(),
            ["goal"],
            goals_per_candidate=1,
            trials_per_goal=1,
            random_seed=7,
        )
        report = validator.validate(StrategyCandidate(
            "marked", "Marked", "wrapped {input}", "template",
            metadata={"response_extractor": "after_marker", "response_markers": ["[[Answer]]"]},
        ))
        self.assertEqual(report.success_rate, 1.0)
        self.assertEqual(report.average_score, 5.0)


if __name__ == "__main__":
    unittest.main()
