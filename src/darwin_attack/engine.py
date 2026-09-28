from __future__ import annotations

from .composer import FailureReflector, PromptComposer
from .config import AttackConfig
from .judge import HarmfulnessJudge, IntentJudge, ResponseJudge
from .pool import StrategyPool
from .response_extraction import extract_target_response
from .schemas import AttackAttempt, AttackResult, Evaluation, TargetObservation
from .selector import FeedbackGuidedEvolution
from .storage import Repository
from .targets import Target


class AttackEngine:

    def __init__(
        self,
        config: AttackConfig,
        repository: Repository,
        pool: StrategyPool,
        selector: FeedbackGuidedEvolution,
        composer: PromptComposer,
        reflector: FailureReflector,
        target: Target,
        response_judge: ResponseJudge,
        intent_judge: IntentJudge,
        harmfulness_judge: HarmfulnessJudge | None = None,
    ):
        self.config = config
        self.repository = repository
        self.pool = pool
        self.selector = selector
        self.composer = composer
        self.reflector = reflector
        self.target = target
        self.response_judge = response_judge
        self.intent_judge = intent_judge
        self.harmfulness_judge = harmfulness_judge
        self.scope = f"{config.target_id}::{config.dataset_id}"

    def _evaluate(
        self, goal: str, disguised_prompt: str, observation: TargetObservation
    ) -> Evaluation:
        if self.config.target_kind == "llm":
            return self.response_judge.evaluate(goal, observation.output)
        intent = self.intent_judge.evaluate(goal, disguised_prompt)
        success = observation.decision == "safe" and intent.success
        reason = f"guard={observation.decision}; intent={intent.rationale}"
        return Evaluation(success=success, score=float(success), rationale=reason)

    def attack(self, goal: str, instance_id: str) -> AttackResult:
        strategies = {item.id: item for item in self.pool.active()}
        if not strategies:
            raise RuntimeError("Cannot attack with an empty strategy pool")
        self.selector.sync()
        goal_embedding = self.selector.encode_goal(goal)

        attempts: list[AttackAttempt] = []
        query_count = 0
        best_score = 0.0
        successful = False
        terminal_response = ""

        for chain_index in range(self.config.chains_per_instance):
            if successful or query_count >= self.config.max_target_queries:
                break
            current_prompt = goal
            sequence: list[int] = []
            previous_strategy_id: int | None = None
            strategy_id = self.selector.select_initial(goal_embedding)

            for step_index in range(self.config.max_chain_length):
                if query_count >= self.config.max_target_queries:
                    break
                if step_index > 0:
                    if previous_strategy_id is None:
                        raise RuntimeError("Missing previous strategy in composed chain")
                    strategy_id = self.selector.select_next(previous_strategy_id)
                strategy = strategies.get(strategy_id)
                if strategy is None:
                    strategies = {item.id: item for item in self.pool.active()}
                    strategy = strategies.get(strategy_id)
                if strategy is None:
                    raise RuntimeError(f"Selected inactive strategy: {strategy_id}")

                disguised = self.composer.apply(strategy, goal, current_prompt)
                observation = self.target.query(disguised)
                query_count += 1
                sequence.append(strategy.id)
                judged_response = observation.output
                extraction = {
                    "applied": False,
                    "extractor": "",
                    "marker_found": False,
                }
                if self.config.target_kind == "llm":
                    judged_response, extraction = extract_target_response(
                        observation.output,
                        strategy.metadata,
                    )
                judged_observation = TargetObservation(
                    output=judged_response,
                    decision=observation.decision,
                    raw=observation.raw,
                )
                evaluation = self._evaluate(goal, disguised, judged_observation)
                best_score = max(best_score, evaluation.score)
                terminal_response = judged_response

                if previous_strategy_id is not None:
                    self.selector.update(
                        previous_strategy_id,
                        strategy.id,
                        float(evaluation.success),
                    )

                attempt = AttackAttempt(
                    query_number=query_count,
                    chain_index=chain_index,
                    step_index=step_index,
                    strategy_id=strategy.id,
                    strategy_name=strategy.name,
                    strategy_sequence=tuple(sequence),
                    disguised_prompt=disguised,
                    raw_target_response=observation.output,
                    target_response=judged_response,
                    response_extraction=extraction,
                    target_decision=observation.decision,
                    score=evaluation.score,
                    success=evaluation.success,
                    evaluator_output=evaluation.rationale,
                )
                attempts.append(attempt)
                self.repository.record_attempt(self.scope, instance_id, goal, attempt)

                if evaluation.success:
                    successful = True
                    self.repository.store_success(
                        self.scope,
                        goal,
                        goal_embedding,
                        sequence,
                        evaluation.score,
                    )
                    break

                previous_strategy_id = strategy.id
                if (
                    step_index + 1 < self.config.max_chain_length
                    and query_count < self.config.max_target_queries
                ):
                    current_prompt = self.reflector.refine_prompt(
                        original=goal,
                        current=disguised,
                        feedback=observation.output or evaluation.rationale,
                    )

        harmfulness_score = None
        harmfulness_evaluation = None
        if self.config.target_kind == "llm" and self.harmfulness_judge and attempts:
            harmfulness_evaluation = {
                **self.harmfulness_judge.evaluate(goal, terminal_response),
                "query_number": attempts[-1].query_number,
            }
            harmfulness_score = harmfulness_evaluation["score"]
        return AttackResult(
            instance_id=instance_id,
            goal=goal,
            target_id=self.config.target_id,
            dataset_id=self.config.dataset_id,
            success=successful,
            query_count=query_count,
            best_score=best_score,
            harmfulness_score=harmfulness_score,
            attempts=tuple(attempts),
            harmfulness_evaluation=harmfulness_evaluation,
        )
