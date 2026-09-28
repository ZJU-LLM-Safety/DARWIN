from __future__ import annotations

import json
import unittest

from darwin_attack.reevaluation import rejudge_record, summarize_rejudged
from darwin_attack.schemas import Evaluation


class AlternateResponseJudge:
    def evaluate(self, goal: str, response: str) -> Evaluation:
        success = response == "alternative success"
        return Evaluation(success, 5.0 if success else 1.0, "alternate")


class AlternateIntentJudge:
    def evaluate(self, goal: str, rewritten: str) -> Evaluation:
        return Evaluation(True, 1.0, "preserved")


class FixedHarmfulnessJudge:
    def __init__(self):
        self.calls = []
        self.details = {
            "score": 0.75, "refused": False, "specificity": 4,
            "convincingness": 4, "reason": "Fixture judgment.",
            "raw_output": "Fixture rubric output.",
        }

    def evaluate(self, goal: str, response: str) -> dict:
        self.calls.append((goal, response))
        return self.details


class ReevaluationTest(unittest.TestCase):
    def test_rejudges_recorded_attempts_without_target_access(self):
        record = {
            "instance_id": "item-1",
            "goal": "goal",
            "target_id": "target",
            "dataset_id": "set",
            "attempts": [
                {"query_number": 1, "target_response": "failure"},
                {"query_number": 2, "target_response": "alternative success"},
                {"query_number": 3, "target_response": "unused"},
            ],
        }
        hs = FixedHarmfulnessJudge()
        result = rejudge_record(
            record,
            "llm",
            60,
            AlternateResponseJudge(),
            AlternateIntentJudge(),
            hs,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(len(result["judgments"]), 2)
        self.assertEqual(result["harmfulness_score"], 0.75)
        self.assertEqual(hs.calls, [("goal", "alternative success")])
        expected = {**hs.details, "query_number": 2}
        self.assertEqual(result["harmfulness_evaluation"], expected)
        self.assertEqual(json.loads(json.dumps(result))["harmfulness_evaluation"], expected)
        self.assertNotIn("query_number", hs.details)
        self.assertEqual(summarize_rejudged([result])["asr"], 1.0)


    def test_rejudging_prefers_complete_raw_response_for_success_and_hs(self):
        class RecordingHarmfulnessJudge:
            def __init__(self):
                self.responses = []

            def evaluate(self, goal, response):
                self.responses.append(response)
                return {"score": 0.25, "refused": False, "specificity": 2,
                        "convincingness": 2, "reason": "fixture", "raw_output": "fixture"}

        full_reply = "context before answer\n[[Answer]]: alternative success"
        hs = RecordingHarmfulnessJudge()
        result = rejudge_record(
            {"goal": "goal", "attempts": [{
                "query_number": 1,
                "raw_target_response": full_reply,
                "target_response": "alternative success",
                "response_extraction": {"applied": True},
            }]},
            "llm", 60, AlternateResponseJudge(), AlternateIntentJudge(), hs,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["query_count"], 60)
        self.assertEqual(hs.responses, [full_reply])
        self.assertEqual(result["harmfulness_score"], 0.25)
        self.assertEqual(result["harmfulness_evaluation"]["query_number"], 1)

    def test_failed_rejudging_evaluates_only_last_complete_response(self):
        hs = FixedHarmfulnessJudge()
        complete = "context\n[[Answer]]: final failed answer"
        result = rejudge_record(
            {"goal": "goal", "attempts": [
                {"query_number": 3, "raw_target_response": complete,
                 "target_response": "extracted fragment", "response_extraction": {"applied": True}},
                {"query_number": 1, "target_response": "first failure"},
                {"query_number": 2, "target_response": "second failure"},
            ]},
            "llm", 60, AlternateResponseJudge(), AlternateIntentJudge(), hs,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["query_count"], 60)
        self.assertEqual([row["query_number"] for row in result["judgments"]], [1, 2, 3])
        self.assertEqual(hs.calls, [("goal", complete)])
        self.assertEqual(result["harmfulness_evaluation"], {**hs.details, "query_number": 3})

    def test_hs_is_absent_for_guard_or_missing_attempts(self):
        cases = (
            ("guardrail", [{"query_number": 1, "target_decision": "safe", "disguised_prompt": "goal"}]),
            ("llm", []),
        )
        for target_kind, attempts in cases:
            with self.subTest(target_kind=target_kind):
                hs = FixedHarmfulnessJudge()
                result = rejudge_record(
                    {"goal": "goal", "attempts": attempts}, target_kind, 60,
                    AlternateResponseJudge(), AlternateIntentJudge(), hs,
                )
                self.assertEqual(hs.calls, [])
                self.assertIsNone(result["harmfulness_score"])
                self.assertIsNone(result["harmfulness_evaluation"])

    def test_rejudging_rejects_an_extracted_record_without_the_original(self):
        with self.assertRaisesRegex(ValueError, "without raw_target_response"):
            rejudge_record(
                {"goal": "goal", "attempts": [{
                    "query_number": 1, "target_response": "alternative success",
                    "response_extraction": {"applied": True},
                }]},
                "llm", 60, AlternateResponseJudge(), AlternateIntentJudge(), FixedHarmfulnessJudge(),
            )


if __name__ == "__main__":
    unittest.main()
