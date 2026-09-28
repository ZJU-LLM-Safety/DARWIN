from __future__ import annotations

import random
from typing import Sequence

from .composer import PromptComposer
from .judge import ResponseJudge
from .response_extraction import extract_target_response
from .schemas import SandboxReport, StrategyCandidate, StrategyRecord
from .targets import Target


class SandboxValidator:
    def __init__(
        self,
        composer: PromptComposer,
        target: Target,
        judge: ResponseJudge,
        goals: Sequence[str],
        goals_per_candidate: int,
        trials_per_goal: int,
        random_seed: int,
    ):
        if not goals:
            raise ValueError("Sandbox validation requires at least one goal")
        self.rng = random.Random(random_seed)
        count = min(goals_per_candidate, len(goals))
        self.composer = composer
        self.target = target
        self.judge = judge
        self.goals = self.rng.sample(list(goals), count)
        self.trials_per_goal = trials_per_goal

    def validate(self, candidate: StrategyCandidate) -> SandboxReport:
        temporary = StrategyRecord(
            id=-1,
            key=candidate.key,
            name=candidate.name,
            instruction=candidate.instruction,
            mode=candidate.mode,
            status="active",
            total_attempts=0,
            total_successes=0,
            sandbox_success_rate=0.0,
            sandbox_average_score=0.0,
            validation_status="measured",
            generation=candidate.generation,
            metadata=candidate.metadata,
        )
        successes = 0
        scores: list[float] = []
        for goal in self.goals:
            for _ in range(self.trials_per_goal):
                prompt = self.composer.apply(temporary, goal, goal)
                observation = self.target.query(prompt)
                response, _ = extract_target_response(observation.output, candidate.metadata)
                evaluation = self.judge.evaluate(goal, response)
                success = evaluation.success
                score = evaluation.score
                successes += int(success)
                scores.append(score)
        trials = len(scores)
        return SandboxReport(
            successes=successes,
            trials=trials,
            average_score=sum(scores) / trials if trials else 0.0,
        )
