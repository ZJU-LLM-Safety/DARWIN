from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from darwin_attack.pool import StrategyPool
from darwin_attack.schemas import SandboxReport, StrategyCandidate
from darwin_attack.selector import FeedbackGuidedEvolution
from darwin_attack.storage import Repository

from fakes import FakeEmbedder


def _seed(repository: Repository) -> None:
    pool = StrategyPool(repository, FakeEmbedder(), 0.95, 0.80, 10)
    report = SandboxReport(1, 1, 5.0)
    pool.consider(StrategyCandidate("alpha", "Alpha", "alpha {input}", "template"), report)
    pool.consider(StrategyCandidate("beta", "Beta", "beta {input}", "template"), report)


class SelectorTest(unittest.TestCase):
    def test_q_update_and_history_are_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Repository(Path(directory) / "state.sqlite3")
            _seed(repository)
            embedder = FakeEmbedder()
            selector = FeedbackGuidedEvolution(repository, embedder, "target-a::set-a", 0.9, 0.1, 0.5, 7)
            selector.sync()

            selector.update(1, 2, 1.0)
            updated_row = repository.transition_row("target-a::set-a", 1)
            self.assertAlmostEqual(updated_row[2], 0.575 / 1.075)
            self.assertAlmostEqual(sum(updated_row.values()), 1.0)

            with self.assertRaises(ValueError):
                selector.update(1, 2, 0.8)

            goal_embedding = selector.encode_goal("goal example")
            repository.store_success("target-a::set-a", "goal example", goal_embedding, [2, 1], 5)
            self.assertEqual(selector.select_initial(goal_embedding), 2)

            isolated = FeedbackGuidedEvolution(repository, embedder, "target-b::set-a", 0.9, 0.1, 0.5, 7)
            isolated.sync()
            self.assertEqual(repository.success_memories("target-b::set-a"), [])
            self.assertAlmostEqual(repository.transition_row("target-b::set-a", 1)[2], 0.5)
            repository.close()


if __name__ == "__main__":
    unittest.main()
