from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from darwin_attack.schemas import Evaluation, TargetObservation


class FakeEmbedder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "beta" in lowered:
                vectors.append([0.0, 1.0])
            elif "goal" in lowered:
                vectors.append([2**-0.5, 2**-0.5])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


class StaticModel:
    def __init__(self, output: str):
        self.output = output
        self.messages = []

    def generate(self, messages):
        self.messages.append(messages)
        return self.output


class NeverUsedModel:
    def generate(self, messages):
        raise AssertionError("Template strategies must not call the generator")


class SequenceTarget:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.queries: list[str] = []

    def query(self, prompt: str) -> TargetObservation:
        self.queries.append(prompt)
        return TargetObservation(output=next(self.outputs))


class KeywordResponseJudge:
    def evaluate(self, goal: str, response: str) -> Evaluation:
        score = 5.0 if response == "success" else 1.0
        return Evaluation(score == 5.0, score, f"score={score}")


class AlwaysIntentJudge:
    def evaluate(self, goal: str, rewritten: str) -> Evaluation:
        return Evaluation(True, 1.0, "preserved")


class IdentityReflector:
    def refine_prompt(self, original: str, current: str, feedback: str) -> str:
        return current
