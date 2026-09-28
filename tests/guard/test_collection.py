import json

import unittest

from darwin_guard.collection import collect_pair
from darwin_guard.filtering import IntentPreservationFilter, build_filter_prompt, parse_filter_output


class Proposer:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.started = []
        self.inputs = []
        self.feedbacks = []

    def start(self, source):
        self.started.append(source)

    def propose(self, source, current_prompt, attempt):
        self.inputs.append((current_prompt, attempt))
        return {"text": next(self.texts), "strategy_ids": list(range(1, attempt + 1))}

    def feedback(self, source, candidate, decision, success):
        self.feedbacks.append((candidate["text"], decision, success))


class Guard:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.inputs = []

    def predict(self, text):
        self.inputs.append(text)
        return next(self.decisions)


class Filter:
    def __init__(self, keep=True):
        self.result = keep
        self.inputs = []

    def keep(self, original, rewritten, label):
        self.inputs.append((original, rewritten, label))
        return self.result


class CollectionTests(unittest.TestCase):
    def test_symmetric_early_stop_and_inherited_label(self):
        for label in (0, 1):
            with self.subTest(label=label):
                proposer = Proposer(["first variant", "second variant", "unused variant"])
                guard, intent_filter = Guard([label, 1 - label]), Filter()
                pair = collect_pair({"id": "example", "text": "original task", "label": label},
                                    proposer, guard, intent_filter, 3)
                self.assertEqual(pair, {
                    "source_id": "example", "raw_prompt": "original task", "disguised_prompt": "second variant",
                    "label": label, "attempts": 2, "misclassified": True, "strategy_ids": [1, 2],
                })
                self.assertEqual(proposer.inputs, [("original task", 1), ("first variant", 2)])
                self.assertEqual(proposer.feedbacks, [("first variant", label, False), ("second variant", 1 - label, True)])
                self.assertEqual(intent_filter.inputs, [("original task", "second variant", label)])
                self.assertEqual(len(proposer.started), 1)

    def test_budget_uses_last_candidate_and_unknown_is_not_a_hit(self):
        proposer, intent_filter = Proposer(["first variant", "last variant"]), Filter()
        pair = collect_pair({"id": "one", "text": "original task", "label": 0},
                            proposer, Guard([None, None]), intent_filter, 2)
        self.assertEqual(pair["disguised_prompt"], "last variant")
        self.assertFalse(pair["misclassified"])
        self.assertEqual(pair["attempts"], 2)
        self.assertEqual(proposer.feedbacks, [("first variant", None, False), ("last variant", None, False)])
        self.assertEqual(intent_filter.inputs, [("original task", "last variant", 0)])

    def test_rejected_terminal_is_not_replaced_by_previous_candidate(self):
        for decisions in ([0, 0], [0, 1]):
            with self.subTest(decisions=decisions):
                proposer, intent_filter = Proposer(["first variant", "last variant"]), Filter(False)
                self.assertIsNone(collect_pair({"id": "one", "text": "original task", "label": 0},
                                              proposer, Guard(decisions), intent_filter, 2))
                self.assertEqual(intent_filter.inputs, [("original task", "last variant", 0)])
                self.assertFalse(any(success for _, _, success in proposer.feedbacks))

    def test_rejected_first_misclassification_still_stops_search(self):
        proposer = Proposer(["first variant", "unused variant"])
        self.assertIsNone(collect_pair({"id": "one", "text": "original task", "label": 1},
                                      proposer, Guard([0]), Filter(False), 2))
        self.assertEqual(len(proposer.inputs), 1)
        self.assertEqual(proposer.feedbacks, [("first variant", 0, False)])

    def test_empty_terminal_does_not_fall_back(self):
        proposer, guard, intent_filter = Proposer(["first variant", " "]), Guard([0]), Filter()
        self.assertIsNone(collect_pair({"id": "one", "text": "original task", "label": 0},
                                      proposer, guard, intent_filter, 2))
        self.assertEqual(guard.inputs, ["first variant"])
        self.assertFalse(intent_filter.inputs)
        self.assertEqual(proposer.feedbacks[-1], (" ", None, False))

    def test_invalid_budget_is_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collect_pair({"id": "one", "text": "task", "label": 0}, Proposer([]), Guard([]), Filter(), value)

    def test_filter_fails_closed(self):
        for raw in (
            "", "not json", "[]", '"keep"',
            '{"decision":"keep","same_intent":"true","label_preserved":true}',
            '{"decision":"keep","same_intent":1,"label_preserved":true}',
            '{"decision":"keep","same_intent":true,"label_preserved":false}',
            '{"decision":"drop","same_intent":true,"label_preserved":true}',
            '{"decision":"drop","decision":"keep","same_intent":true,"label_preserved":true}',
        ):
            with self.subTest(raw=raw):
                self.assertFalse(parse_filter_output(raw))

    def test_filter_prompt_and_model_protocol(self):
        class Model:
            def generate(model_self, messages):
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0]["role"], "user")
                prompt = messages[0]["content"]
                self.assertIn("Route: benign_to_unsafe_misclass", prompt)
                self.assertIn("Expected label: safe", prompt)
                self.assertIn("<ORIGINAL_PROMPT>\noriginal task\n</ORIGINAL_PROMPT>", prompt)
                return json.dumps({"decision": "keep", "same_intent": True, "label_preserved": True})
        self.assertTrue(IntentPreservationFilter.from_model(Model()).keep("original task", "variant", 0))
        self.assertIn("Route: harmful_to_safe_misclass", build_filter_prompt("task", "variant", 1))


if __name__ == "__main__":
    unittest.main()
