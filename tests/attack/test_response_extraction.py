from __future__ import annotations

import unittest

from darwin_attack.response_extraction import extract_target_response, recorded_target_response


class ResponseExtractionTest(unittest.TestCase):
    def test_complete_response_is_unchanged_even_with_legacy_marker(self):
        raw = "  context\n[[Deliverable]]: first line\nsecond line  "
        response, metadata = extract_target_response(
            raw,
            {"response_extractor": "after_marker", "response_markers": ["[[Deliverable]]"]},
        )
        self.assertEqual(response, raw)
        self.assertFalse(metadata["applied"])
        self.assertEqual(metadata["evaluation_scope"], "full_response")
        self.assertTrue(metadata["legacy_extractor_ignored"])

    def test_missing_legacy_marker_retains_the_complete_response(self):
        response, metadata = extract_target_response(
            "unstructured output",
            {"response_extractor": "after_marker", "response_markers": ["[[Deliverable]]"]},
        )
        self.assertEqual(response, "unstructured output")
        self.assertFalse(metadata["applied"])

    def test_old_record_uses_raw_even_when_empty(self):
        self.assertEqual(recorded_target_response({
            "raw_target_response": "context\n[[Answer]]: answer", "target_response": "answer",
        }), "context\n[[Answer]]: answer")
        self.assertEqual(recorded_target_response({
            "raw_target_response": "", "target_response": "old extracted text",
        }), "")

    def test_older_records_without_raw_field_have_explicit_fallback(self):
        self.assertEqual(recorded_target_response({"target_response": "available response"}), "available response")
        with self.assertRaises(ValueError):
            recorded_target_response({"raw_target_response": None, "target_response": "fallback"})
        with self.assertRaisesRegex(ValueError, "without raw_target_response"):
            recorded_target_response({
                "target_response": "cropped", "response_extraction": {"applied": True},
            })


if __name__ == "__main__":
    unittest.main()
