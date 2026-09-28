from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from darwin_attack.config import ConfigurationError, load_config
from darwin_attack.extraction import ExternalKnowledgeEvolution

from fakes import StaticModel


CONFIG = """
runtime:
  database_path: state.sqlite3
  random_seed: 7
models:
  strategy_generator: &model
    provider: transformers
    model: local-model
    temperature: 0.7
    max_tokens: 128
  reflection: *model
  sandbox_target: *model
  target: *model
  judge: *model
  intent_judge: *model
embedding:
  model: embedding-model
  device: null
  similarity_threshold: 0.8
  history_threshold: 0.9
pool:
  target_size: 200
  admission_threshold: 0.8
  crossover_top_k: 5
  crossover_probability: 0.5
  mutation_probability: 0.5
  mutation_operator_count: 15
  mutation_operators_file: operators.jsonl
selection:
  alpha: 0.1
  gamma: 0.5
sandbox:
  dataset_path: sandbox.jsonl
  goals_per_candidate: 5
  trials_per_goal: 2
attack:
  dataset_path: evaluation.jsonl
  dataset_id: evaluation
  target_id: target
  target_kind: llm
  max_target_queries: 60
  chains_per_instance: 20
  max_chain_length: 3
  success_score: 5
  guardrail_safe_pattern: null
  guardrail_unsafe_pattern: null
"""


class ConfigAndExtractionTest(unittest.TestCase):
    def test_complete_config_loads_without_source_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            path.write_text(CONFIG, encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.attack.max_target_queries, 60)
            self.assertEqual(config.pool.admission_threshold, 0.8)
            self.assertEqual(config.pool.mutation_operator_count, 15)
            self.assertEqual(config.attack.dataset_path.name, "evaluation.jsonl")

    def test_same_sandbox_and_evaluation_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            path.write_text(
                CONFIG.replace("evaluation.jsonl", "sandbox.jsonl"),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_extractor_returns_structured_candidate_without_raw_material(self):
        model = StaticModel(
            '[{"name":"Format","instruction":"Format {input}","mode":"template","tags":["format"]}]'
        )
        candidates = ExternalKnowledgeEvolution(model).extract("research note", "record-1")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].mode, "template")
        self.assertNotIn("research note", candidates[0].to_dict().values())


if __name__ == "__main__":
    unittest.main()
