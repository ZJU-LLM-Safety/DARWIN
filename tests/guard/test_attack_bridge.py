from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from darwin_guard.attack_bridge import DARWINAttackAdapter, _copy_initial_database, evolve_pool


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "transformers"
    model: str = "test-model"
    temperature: float = 0.0
    max_tokens: int = 32
    api_key_env: str | None = None
    base_url_env: str | None = None
    device: str | None = None
    extra: dict | None = None

    @classmethod
    def from_dict(cls, values, section):
        return cls(**values)


@dataclass(frozen=True)
class Strategy:
    id: int
    key: str
    name: str = "format"
    instruction: str = "format {input}"
    mode: str = "template"
    status: str = "active"
    metadata: dict | None = None


class Repository:
    def __init__(self):
        self.strategies = [
            Strategy(i, f"seed-{i}", metadata={"special_executor": "direct_template"})
            for i in range(1, 51)
        ]
        self.attempts = []
        self.memories = []
        self.closed = False

    def active_strategies(self):
        return self.strategies

    def strategy(self, strategy_id):
        return next((s for s in self.strategies if s.id == strategy_id), None)

    def strategy_by_key(self, key):
        return next((s for s in self.strategies if s.key == key), None)

    def count_strategies(self):
        return len(self.strategies)

    def embeddings(self):
        return [(s.id, ["seed"]) for s in self.strategies]

    def record_attempt(self, scope, instance_id, goal, attempt):
        self.attempts.append((scope, instance_id, goal, attempt))

    def store_success(self, scope, goal, embedding, sequence, score):
        self.memories.append((scope, list(sequence), score))

    def close(self):
        self.closed = True


class Embedder:
    def __init__(self, config):
        pass

    def encode(self, texts):
        return [["seed" if text == "duplicate" else text] for text in texts]


class Selector:
    def __init__(self, repository, embedder, scope, history_threshold, alpha, gamma, seed):
        self.repository, self.embedder, self.scope = repository, embedder, scope
        self.updates = []
        self.synced = 0

    def sync(self):
        self.synced += 1

    def encode_goal(self, goal):
        return self.embedder.encode([goal])[0]

    def select_initial(self, embedding):
        return 1

    def select_next(self, current):
        return current + 1

    def update(self, previous, current, reward):
        self.updates.append((previous, current, reward))


class Composer:
    def __init__(self, generator):
        self.generator = generator
        self.calls = []

    def apply(self, strategy, original, current):
        self.calls.append((strategy, original, current))
        if strategy.mode == "instruction":
            return self.generator.generate([{"role": "user", "content": current}])
        return f"format({current})"


class Reflector:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def refine_prompt(self, original, current, feedback):
        self.calls.append((original, current, feedback))
        return current + " reflected"


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Path(self.directory.name) / "runtime.sqlite3"
        self.db.touch()
        self.repository = Repository()
        self.model_calls = []
        self.builds = []
        self.sandbox_calls = []
        self.reflection_calls = []
        self.candidates = []
        self.config = SimpleNamespace(
            model=lambda role: ModelConfig(model=role),
            attack=SimpleNamespace(max_chain_length=2, success_score=5, max_target_queries=60, dataset_path=Path("evaluation.jsonl")),
            embedding=SimpleNamespace(history_threshold=0.7, similarity_threshold=0.8),
            selection=SimpleNamespace(alpha=0.1, gamma=0.5),
            pool=SimpleNamespace(
                target_size=200, admission_threshold=0.8, crossover_top_k=5,
                crossover_probability=0.5, mutation_probability=0.5,
                mutation_operators_file=Path("operators.jsonl"), mutation_operator_count=1,
            ),
            sandbox=SimpleNamespace(
                dataset_path=Path("sandbox.jsonl"), goals_per_candidate=1, trials_per_goal=1,
            ),
            runtime=SimpleNamespace(random_seed=17),
        )
        owner = self

        def build_model(config):
            owner.builds.append(config)

            def generate(messages):
                owner.model_calls.append(messages)
                return "benign variant"

            return SimpleNamespace(generate=generate)

        class Genetic:
            def __init__(self, **kwargs):
                pass

            def generate(self, count):
                return owner.candidates[:count]

        class Reflection:
            def __init__(self, repository, model):
                pass

            def generate(self, strategy_id, failed_prompt, feedback):
                owner.reflection_calls.append((strategy_id, failed_prompt, feedback))
                return Strategy(0, "reflected", instruction="novel reflection", metadata={})

        class Sandbox:
            def __init__(self, *args):
                owner.sandbox_seed = args[-1]

            def validate(self, candidate):
                owner.sandbox_calls.append(candidate.key)
                return SimpleNamespace(
                    success_rate=0.2 if candidate.key == "weak" else 1.0, average_score=5.0,
                )

        class Pool:
            def __init__(self, repository, embedder, similarity_threshold, admission, target):
                self.repository, self.admission = repository, admission

            def consider(self, candidate, report):
                admitted = report.success_rate >= self.admission
                index = len(self.repository.strategies) + 1 if admitted else None
                if admitted:
                    self.repository.strategies.append(Strategy(
                        index, candidate.key, instruction=candidate.instruction, metadata={},
                    ))
                return SimpleNamespace(
                    admitted=admitted, strategy_id=index, maximum_similarity=0.0,
                    reason="admitted" if admitted else "below_sandbox_threshold",
                )

        self.api = SimpleNamespace(
            load_config=lambda path, *, validate_target=True: self.config, ModelConfig=ModelConfig,
            Repository=lambda path: self.repository, SentenceTransformerEmbedder=Embedder,
            FeedbackGuidedEvolution=Selector, PromptComposer=Composer, FailureReflector=Reflector,
            build_model=build_model, AttackAttempt=lambda **kwargs: SimpleNamespace(**kwargs),
            GeneticStrategyEvolution=Genetic, ReflectionDrivenEvolution=Reflection,
            load_mutation_operators=lambda path: ["format"], StrategyPool=Pool,
            cosine_similarity=lambda left, right: float(left == right),
            SandboxValidator=Sandbox, LLMTarget=lambda model: model,
            ResponseJudge=lambda model, score: model,
            load_goals=lambda path: [("heldout", "separate sandbox goal")],
            assert_disjoint_goals=lambda first, second: None,
        )
        self.patcher = patch("darwin_guard.attack_bridge._load_attack_api", return_value=self.api)
        self.patcher.start()
        self.proposer = DARWINAttackAdapter("attack.yaml", str(self.db), 1, 42)

    def tearDown(self):
        self.proposer.close()
        self.patcher.stop()
        self.directory.cleanup()

    @staticmethod
    def source(label=1):
        return {"id": f"source-{label}", "text": "original request", "label": label}

    def test_initialization_does_not_load_chat_models(self):
        self.assertEqual(self.builds, [])

    def test_feedback_composes_with_reflection_then_restarts_full_chain(self):
        source = self.source()
        self.proposer.start(source)
        first = self.proposer.propose(source, "ignored initial", 1)
        self.proposer.feedback(source, first, 1, False)
        second = self.proposer.propose(source, first["text"], 2)
        self.assertEqual(second["strategy_ids"], [1, 2])
        self.assertIn("reflected", second["text"])
        self.proposer.feedback(source, second, 1, False)
        third = self.proposer.propose(source, second["text"], 3)
        self.assertEqual(third["strategy_ids"], [1])
        self.assertEqual(third["text"], "format(original request)")
        self.proposer.feedback(source, third, 0, True)
        self.assertEqual(self.proposer.selectors[1].updates, [(1, 2, 0.0)])
        self.assertEqual(self.repository.memories[-1][1], [1])
        self.assertEqual(len(self.proposer.failure_records), 2)
        self.assertIn("Source label: unsafe", self.proposer.failure_records[0]["feedback"])

    def test_benign_templates_use_intent_preserving_generator(self):
        source = self.source(0)
        self.proposer.start(source)
        candidate = self.proposer.propose(source, source["text"], 1)
        strategy = self.proposer.composers[0].calls[0][0]
        self.assertEqual(strategy.mode, "instruction")
        self.assertEqual(strategy.metadata, {})
        self.assertIn("benign", strategy.instruction)
        self.assertEqual(self.model_calls[0][0]["role"], "system")
        self.assertIn("Preserve its exact benign task", self.model_calls[0][0]["content"])
        self.proposer.feedback(source, candidate, 1, True)
        self.assertEqual(self.repository.memories[-1][0], "guard-training::benign")

    def test_feedback_state_persists_across_rounds_but_separates_routes(self):
        for label in (1, 0):
            source = self.source(label)
            self.proposer.start(source)
            candidate = self.proposer.propose(source, source["text"], 1)
            self.proposer.feedback(source, candidate, 1 - label, True)
        other = DARWINAttackAdapter("attack.yaml", str(self.db), 2, 43)
        other.start(self.source(1))
        self.assertNotEqual(self.proposer.selectors[0].scope, self.proposer.selectors[1].scope)
        self.assertEqual(self.proposer.selectors[1].scope, other.selectors[1].scope)
        other.close()

    def test_evolution_models_can_be_unloaded_without_discarding_wrappers(self):
        model = self.proposer._model("judge")
        model.generate([])
        self.assertIsNotNone(model.instance)
        self.proposer.release_evolution_models()
        self.assertIsNone(model.instance)
        self.assertIs(self.proposer._model("judge"), model)
        model.generate([])
        self.assertIsNotNone(model.instance)

    def test_collector_owns_attempt_budget(self):
        self.config.attack.max_target_queries = 1
        source = self.source()
        self.proposer.start(source)
        current = source["text"]
        for attempt in range(1, 21):
            candidate = self.proposer.propose(source, current, attempt)
            self.proposer.feedback(source, candidate, 1, False)
            current = candidate["text"]
        self.assertEqual(len(self.repository.attempts), 20)

    def test_invalid_guard_decision_cannot_be_a_success(self):
        source = self.source()
        self.proposer.start(source)
        candidate = self.proposer.propose(source, source["text"], 1)
        with self.assertRaises(ValueError):
            self.proposer.feedback(source, candidate, None, True)
        self.proposer.feedback(source, candidate, None, False)
        self.assertIsNone(self.repository.attempts[-1][3].target_decision)

    def test_missing_feedback_cannot_be_silently_overwritten(self):
        source = self.source()
        self.proposer.start(source)
        self.proposer.propose(source, source["text"], 1)
        with self.assertRaises(RuntimeError):
            self.proposer.propose(source, source["text"], 2)
        with self.assertRaises(RuntimeError):
            self.proposer.start(self.source(0))

    def test_empty_generation_consumes_one_attempt_and_records_failure(self):
        source = self.source()
        self.proposer.start(source)
        with patch.object(
            self.proposer.composers[1], "apply",
            side_effect=RuntimeError("Strategy application produced an empty prompt"),
        ):
            candidate = self.proposer.propose(source, source["text"], 1)
        self.assertEqual(candidate["text"], "")
        self.assertEqual(candidate["strategy_ids"], [1])
        self.proposer.feedback(source, candidate, None, False)
        self.assertEqual(self.repository.attempts[-1][3].query_number, 1)
        self.assertEqual(self.proposer.failure_records[-1]["failed_prompt"], "")
        next_candidate = self.proposer.propose(source, source["text"], 2)
        self.assertEqual(next_candidate["attempt"], 2)
        self.assertEqual(next_candidate["strategy_ids"], [1, 2])

    def test_other_generation_errors_are_not_converted_to_empty_candidates(self):
        source = self.source()
        self.proposer.start(source)
        with patch.object(
            self.proposer.composers[1], "apply", side_effect=RuntimeError("model load failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "model load failure"):
                self.proposer.propose(source, source["text"], 1)
        self.assertEqual(self.repository.attempts, [])

    def test_source_isolation_runs_without_candidates_and_loads_no_models(self):
        self.proposer.source_records = {"train-source": "separate sandbox goal"}
        with patch.object(
            self.api, "assert_disjoint_goals", side_effect=ValueError("overlapping source"),
        ) as disjoint:
            with self.assertRaisesRegex(ValueError, "overlapping source"):
                self.proposer.validate_source_isolation()
        disjoint.assert_called_once_with(
            [("heldout", "separate sandbox goal")],
            [("train-source", "separate sandbox goal")],
        )
        self.assertEqual(self.builds, [])
        self.assertEqual(self.sandbox_calls, [])

    def test_evolution_checks_isolation_before_empty_or_full_pool_return(self):
        self.proposer.source_records = {"train-source": "separate sandbox goal"}
        for target_size in (50, 200):
            with self.subTest(target_size=target_size):
                self.config.pool.target_size = target_size
                with patch.object(
                    self.api, "assert_disjoint_goals", side_effect=ValueError("overlapping source"),
                ) as disjoint:
                    with self.assertRaisesRegex(ValueError, "overlapping source"):
                        evolve_pool(self.proposer, {})
                self.assertEqual(disjoint.call_count, 1)
        self.assertEqual(self.builds, [])
        self.assertEqual(self.sandbox_calls, [])

    def test_variant_override_does_not_change_evolution_generator(self):
        other = DARWINAttackAdapter(
            "attack.yaml", str(self.db), 1, 42,
            {"model": "variant-model", "max_new_tokens": 256},
        )
        self.assertEqual(other.variant_generator.config.model, "variant-model")
        self.assertEqual(other.variant_generator.config.max_tokens, 256)
        self.assertEqual(other._model("strategy_generator").config.model, "strategy_generator")
        other.close()

    def test_requires_valid_active_seed_pool(self):
        self.repository.strategies = self.repository.strategies[:49]
        with self.assertRaisesRegex(ValueError, "50 to 200"):
            DARWINAttackAdapter("attack.yaml", str(self.db), 1, 42)

    def test_evolution_deduplicates_before_sandbox_and_rejects_weak_candidate(self):
        self.candidates = [
            Strategy(0, "same-text", instruction="duplicate", metadata={}),
            Strategy(0, "weak", instruction="weak novel", metadata={}),
            Strategy(0, "novel", instruction="strong novel", metadata={}),
        ]
        summary = evolve_pool(self.proposer, {"genetic_candidates": 3})
        self.assertEqual(self.sandbox_calls, ["weak", "novel"])
        self.assertEqual(summary["admitted"], 1)
        self.assertEqual(summary["after"], 51)
        self.assertEqual(summary["decisions"][0]["reason"], "semantic_duplicate")
        self.assertEqual(summary["decisions"][1]["reason"], "below_sandbox_threshold")
        self.assertEqual(self.sandbox_seed, 17)

    def test_strategy_reflection_uses_recorded_guard_failure(self):
        source = self.source()
        self.proposer.start(source)
        candidate = self.proposer.propose(source, source["text"], 1)
        self.proposer.feedback(source, candidate, 1, False)
        summary = evolve_pool(self.proposer, {"reflection_candidates": 1})
        self.assertEqual(self.reflection_calls[0][0], 1)
        self.assertEqual(self.reflection_calls[0][1], candidate["text"])
        self.assertIn("Guard decision: unsafe", self.reflection_calls[0][2])
        self.assertEqual(summary["admitted"], 1)

    def test_full_pool_is_reported_without_claiming_new_evolution(self):
        self.config.pool.target_size = 50
        summary = evolve_pool(self.proposer, {"genetic_candidates": 5})
        self.assertEqual(summary["status"], "target_size_reached")
        self.assertEqual(summary["admitted"], 0)
        self.assertEqual(self.sandbox_calls, [])

    def test_external_extraction_uses_same_admission_gate(self):
        self.api.read_jsonl = lambda path: [{"text": "abstract format", "source_id": "local"}]
        self.api.ExternalKnowledgeEvolution = lambda model: SimpleNamespace(
            extract=lambda material, source: [
                Strategy(0, "external", instruction="novel external", metadata={}),
            ],
        )
        summary = evolve_pool(self.proposer, {"external_material": "material.jsonl"})
        self.assertEqual(summary["decisions"][0]["kind"], "external")
        self.assertEqual(self.sandbox_calls, ["external"])
        self.assertEqual(summary["admitted"], 1)


class DatabaseCopyTest(unittest.TestCase):
    def test_initial_database_is_copied_without_overwriting_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "initial.sqlite3"
            destination = Path(directory) / "runtime.sqlite3"
            with sqlite3.connect(source) as connection:
                connection.execute("CREATE TABLE example(value INTEGER)")
                connection.execute("INSERT INTO example VALUES(50)")
            _copy_initial_database(source, destination)
            with sqlite3.connect(destination) as connection:
                self.assertEqual(connection.execute("SELECT value FROM example").fetchone()[0], 50)
                connection.execute("UPDATE example SET value=51")
            _copy_initial_database(source, destination)
            with sqlite3.connect(source) as connection:
                self.assertEqual(connection.execute("SELECT value FROM example").fetchone()[0], 50)
            with sqlite3.connect(destination) as connection:
                self.assertEqual(connection.execute("SELECT value FROM example").fetchone()[0], 51)


class AttackIntegrationTest(unittest.TestCase):
    def test_real_attack_apis_with_fake_models_and_temporary_database(self):
        from darwin_guard.attack_bridge import _load_attack_api

        api = _load_attack_api()
        Candidate = import_module("darwin_attack.schemas").StrategyCandidate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "runtime.sqlite3"
            repository = api.Repository(database)
            for index in range(50):
                repository.add_released_strategy(Candidate(
                    key=f"seed-{index}", name="format", instruction="format({input})",
                    mode="template",
                ), [1.0, 0.0])
            repository.close()
            operators = root / "operators.jsonl"
            operators.write_text(json.dumps({
                "name": "format", "dimension": "surface", "instruction": "change format",
            }) + "\n", encoding="utf-8")
            sandbox_data = root / "sandbox.jsonl"
            sandbox_data.write_text(json.dumps({
                "id": "heldout", "goal": "separate sandbox prompt",
            }) + "\n", encoding="utf-8")
            evaluation_data = root / "evaluation.jsonl"
            evaluation_data.write_text(json.dumps({
                "id": "heldout", "goal": "separate final evaluation prompt",
            }) + "\n", encoding="utf-8")
            config = SimpleNamespace(
                model=lambda role: api.ModelConfig(
                    provider="transformers", model=role, temperature=0.0, max_tokens=32,
                ),
                attack=SimpleNamespace(max_chain_length=2, success_score=5, dataset_path=evaluation_data),
                embedding=SimpleNamespace(history_threshold=0.7, similarity_threshold=0.8),
                selection=SimpleNamespace(alpha=0.1, gamma=0.5),
                pool=SimpleNamespace(
                    target_size=200, admission_threshold=0.8, crossover_top_k=5,
                    crossover_probability=0.0, mutation_probability=1.0,
                    mutation_operators_file=operators, mutation_operator_count=1,
                ),
                sandbox=SimpleNamespace(
                    dataset_path=sandbox_data, goals_per_candidate=1, trials_per_goal=1,
                ),
                runtime=SimpleNamespace(random_seed=17),
            )

            class NumericEmbedder:
                def __init__(self, config):
                    pass

                def encode(self, texts):
                    return [[0.0, 1.0] if "beta" in text else [1.0, 0.0] for text in texts]

            def build_model(model_config):
                if model_config.model == "judge":
                    output = json.dumps({"score": 5, "reason": "fake sandbox success"})
                elif model_config.model == "strategy_generator":
                    output = json.dumps({
                        "name": "beta format", "instruction": "beta format {input}",
                        "mode": "template", "tags": [],
                    })
                else:
                    output = "refined prompt"
                return SimpleNamespace(generate=lambda messages: output)

            api.load_config = lambda path, *, validate_target=True: config
            api.SentenceTransformerEmbedder = NumericEmbedder
            api.build_model = build_model
            with patch("darwin_guard.attack_bridge._load_attack_api", return_value=api):
                proposer = DARWINAttackAdapter("unused.yaml", str(database), 1, 42)
                try:
                    source = {"id": "source", "text": "original request", "label": 1}
                    proposer.start(source)
                    first = proposer.propose(source, source["text"], 1)
                    proposer.feedback(source, first, 1, False)
                    second = proposer.propose(source, first["text"], 2)
                    proposer.feedback(source, second, 0, True)
                    memories = proposer.repository.success_memories("guard-training::harmful")
                    self.assertEqual(len(memories), 1)
                    self.assertEqual(len(memories[0]["sequence"]), 2)
                    summary = evolve_pool(proposer, {"genetic_candidates": 1})
                    self.assertEqual(summary["admitted"], 1)
                    self.assertEqual(summary["after"], 51)
                    record = proposer.repository.active_strategies()[-1]
                    self.assertEqual(record.validation_status, "measured")
                    self.assertEqual(record.sandbox_success_rate, 1.0)
                finally:
                    proposer.close()


if __name__ == "__main__":
    unittest.main()
