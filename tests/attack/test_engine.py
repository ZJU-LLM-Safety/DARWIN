from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from darwin_attack.composer import PromptComposer
from darwin_attack.config import AttackConfig
from darwin_attack.engine import AttackEngine
from darwin_attack.pool import StrategyPool
from darwin_attack.schemas import AttackResult, SandboxReport, StrategyCandidate
from darwin_attack.selector import FeedbackGuidedEvolution
from darwin_attack.storage import Repository

from fakes import (
    AlwaysIntentJudge,
    FakeEmbedder,
    IdentityReflector,
    KeywordResponseJudge,
    NeverUsedModel,
    SequenceTarget,
)


class EngineTest(unittest.TestCase):
    def test_three_step_chains_and_score_five_early_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = Repository(root / "state.sqlite3")
            embedder = FakeEmbedder()
            pool = StrategyPool(repository, embedder, 0.95, 0.80, 10)
            report = SandboxReport(1, 1, 5.0)
            pool.consider(
                StrategyCandidate("alpha", "Alpha", "alpha[{input}]", "template"),
                report,
            )
            pool.consider(
                StrategyCandidate("beta", "Beta", "beta[{input}]", "template"),
                report,
            )
            config = AttackConfig(
                dataset_path=root / "unused.jsonl",
                dataset_id="set-a",
                target_id="target-a",
                target_kind="llm",
                max_target_queries=6,
                chains_per_instance=2,
                max_chain_length=3,
                success_score=5,
                guardrail_safe_pattern=None,
                guardrail_unsafe_pattern=None,
            )
            selector = FeedbackGuidedEvolution(repository, embedder, "target-a::set-a", 0.9, 0.1, 0.5, 11)
            rewards = []
            update_transition = selector.update

            def record_update(from_strategy_id, to_strategy_id, reward):
                rewards.append(reward)
                update_transition(from_strategy_id, to_strategy_id, reward)

            selector.update = record_update
            target = SequenceTarget(["failure", "failure", "failure", "failure", "success"])
            engine = AttackEngine(
                config=config,
                repository=repository,
                pool=pool,
                selector=selector,
                composer=PromptComposer(NeverUsedModel()),
                reflector=IdentityReflector(),
                target=target,
                response_judge=KeywordResponseJudge(),
                intent_judge=AlwaysIntentJudge(),
            )

            result = engine.attack("goal example", "item-0")
            self.assertTrue(result.success)
            self.assertEqual(result.query_count, 5)
            self.assertEqual(result.best_score, 5)
            self.assertEqual(
                [len(item.strategy_sequence) for item in result.attempts],
                [1, 2, 3, 1, 2],
            )
            self.assertEqual(repository.count_strategies(), 2)
            self.assertEqual(len(repository.success_memories("target-a::set-a")), 1)
            self.assertEqual(rewards, [0.0, 0.0, 1.0])
            repository.close()


    def test_complete_response_reaches_success_judge_hs_and_stored_record(self):
        class RecordingResponseJudge(KeywordResponseJudge):
            def __init__(self):
                self.responses = []

            def evaluate(self, goal, response):
                self.responses.append(response)
                return super().evaluate(goal, response)

        class RecordingHarmfulnessJudge:
            def __init__(self):
                self.responses = []

            def evaluate(self, goal, response):
                self.responses.append(response)
                return {"score": 0.25, "refused": False, "specificity": 2,
                        "convincingness": 2, "reason": "fixture", "raw_output": "fixture"}

        for raw, success in (("context\n[[Answer]]: success", False), ("success", True)):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repository = Repository(root / "state.sqlite3")
                self.addCleanup(repository.close)
                embedder = FakeEmbedder()
                pool = StrategyPool(repository, embedder, 0.95, 0.80, 10)
                pool.consider(StrategyCandidate(
                    "marked", "Marked", "alpha[{input}]", "template",
                    metadata={"response_extractor": "after_marker", "response_markers": ["[[Answer]]"]},
                ), SandboxReport(1, 1, 5.0))
                config = AttackConfig(
                    dataset_path=root / "unused.jsonl", dataset_id="set-a", target_id="target-a",
                    target_kind="llm", max_target_queries=1, chains_per_instance=1,
                    max_chain_length=1, success_score=5, guardrail_safe_pattern=None,
                    guardrail_unsafe_pattern=None,
                )
                response_judge = RecordingResponseJudge()
                hs = RecordingHarmfulnessJudge()
                engine = AttackEngine(
                    config=config, repository=repository, pool=pool,
                    selector=FeedbackGuidedEvolution(repository, embedder, "target-a::set-a", 0.9, 0.1, 0.5, 11),
                    composer=PromptComposer(NeverUsedModel()), reflector=IdentityReflector(),
                    target=SequenceTarget([raw]), response_judge=response_judge,
                    intent_judge=AlwaysIntentJudge(), harmfulness_judge=hs,
                )
                result = engine.attack("goal example", "item-0")
                self.assertEqual(result.success, success)
                self.assertEqual(result.query_count, 1)
                self.assertEqual(response_judge.responses, [raw])
                self.assertEqual(hs.responses, [raw])
                self.assertEqual(result.harmfulness_score, 0.25)
                self.assertEqual(result.harmfulness_evaluation["query_number"], 1)
                self.assertEqual(result.attempts[0].target_response, raw)
                self.assertEqual(result.attempts[0].raw_target_response, raw)
                self.assertFalse(result.attempts[0].response_extraction["applied"])
                repository.close()

    def test_hs_evaluates_only_terminal_response_and_serializes_scores(self):
        class RecordingHarmfulnessJudge:
            def __init__(self):
                self.calls = []
                self.details = {
                    "score": 0.5, "refused": False, "specificity": 3,
                    "convincingness": 3, "reason": "Fixture judgment.",
                    "raw_output": "Fixture rubric output.",
                }

            def evaluate(self, goal, response):
                self.calls.append((goal, response))
                return self.details

        cases = (
            ("llm", 3, ["first failure", "success", "unused"], "success", 2),
            ("llm", 3, ["first failure", "second failure", "complete\nlast failure"],
             "complete\nlast failure", 3),
            ("guardrail", 1, ["safe"], None, 1),
            ("llm", 0, [], None, 0),
        )
        for target_kind, budget, outputs, terminal, expected_queries in cases:
            with self.subTest(target_kind=target_kind, outputs=outputs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repository = Repository(root / "state.sqlite3")
                self.addCleanup(repository.close)
                embedder = FakeEmbedder()
                pool = StrategyPool(repository, embedder, 0.95, 0.80, 10)
                pool.consider(StrategyCandidate("alpha", "Alpha", "alpha[{input}]", "template"),
                              SandboxReport(1, 1, 5.0))
                target = SequenceTarget(outputs)
                hs = RecordingHarmfulnessJudge()
                config = AttackConfig(
                    dataset_path=root / "unused.jsonl", dataset_id="set-a", target_id="target-a",
                    target_kind=target_kind, max_target_queries=budget, chains_per_instance=3,
                    max_chain_length=1, success_score=5, guardrail_safe_pattern=None,
                    guardrail_unsafe_pattern=None,
                )
                engine = AttackEngine(
                    config=config, repository=repository, pool=pool,
                    selector=FeedbackGuidedEvolution(repository, embedder, "target-a::set-a", 0.9, 0.1, 0.5, 11),
                    composer=PromptComposer(NeverUsedModel()), reflector=IdentityReflector(),
                    target=target, response_judge=KeywordResponseJudge(),
                    intent_judge=AlwaysIntentJudge(), harmfulness_judge=hs,
                )
                result = engine.attack("goal example", "item-0")
                self.assertEqual(result.query_count, expected_queries)
                self.assertEqual(len(target.queries), expected_queries)
                self.assertEqual(result.success, terminal == "success")
                if terminal is None:
                    self.assertEqual(hs.calls, [])
                    self.assertIsNone(result.harmfulness_score)
                    self.assertIsNone(result.harmfulness_evaluation)
                else:
                    self.assertEqual(hs.calls, [("goal example", terminal)])
                    expected = {**hs.details, "query_number": expected_queries}
                    self.assertEqual(result.harmfulness_evaluation, expected)
                    self.assertEqual(result.harmfulness_score, expected["score"])
                    stored = json.loads(json.dumps(result.to_dict()))
                    self.assertEqual(stored["harmfulness_evaluation"], expected)
                    self.assertNotIn("query_number", hs.details)

    def test_attack_result_accepts_previous_positional_fields(self):
        result = AttackResult("id", "goal", "target", "set", False, 0, 0.0, None, ())
        self.assertIsNone(result.harmfulness_evaluation)


if __name__ == "__main__":
    unittest.main()
