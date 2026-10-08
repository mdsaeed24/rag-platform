"""Report gates reject incomplete/live evidence and invalid timing statistics."""

import copy
import math
import unittest

from eval.benchmark_pipeline import benchmark, latency_summary
from eval.final_validation import EXPECTED_COUNTS, render_markdown, summarize_suite


class PipelineStatisticsTests(unittest.TestCase):
    def test_median_and_nearest_rank_p95(self):
        self.assertEqual(latency_summary(list(range(1, 21))),
                         {"samples": 20, "median_ms": 10.5, "p95_ms": 19})
        self.assertEqual(latency_summary([0.25]), {"samples": 1, "median_ms": .25, "p95_ms": .25})

    def test_empty_negative_and_nonfinite_measurements_are_rejected(self):
        for samples in ([], [-1], [math.nan], [math.inf], [1, -math.inf]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                latency_summary(samples)

    def test_invalid_iterations_fail_before_opening_storage(self):
        for iterations, warmup in ((0, 1), (-1, 1), (1, -1)):
            with self.subTest(iterations=iterations, warmup=warmup), self.assertRaises(ValueError):
                benchmark(iterations, warmup)


class ValidationEvidenceTests(unittest.TestCase):
    def sample_report(self):
        count = EXPECTED_COUNTS["graph"]
        return {"passed": count, "total": count, "completion_calls": 9,
                "live_answer_correctness_evaluated": False,
                "checks": [{"id": str(index), "passed": True, "checks": {"authorized": True}}
                           for index in range(count)]}

    def test_complete_offline_report_is_accepted(self):
        summary = summarize_suite("graph", self.sample_report())
        self.assertTrue(summary["checks_passed"])
        self.assertEqual(summary["mock_completion_calls"], 9)

    def test_incomplete_or_inconsistent_evidence_is_rejected(self):
        original = self.sample_report()
        variants = []
        for change in ({"passed": 15}, {"total": 15}, {"checks": original["checks"][:-1]}):
            variants.append({**original, **change})
        for mutation in ("failed", "false_check", "empty_checks", "duplicate", "missing_id"):
            report = copy.deepcopy(original)
            if mutation == "failed":
                report["checks"][0]["passed"] = False
            elif mutation == "false_check":
                report["checks"][0]["checks"]["authorized"] = False
            elif mutation == "empty_checks":
                report["checks"][0]["checks"] = {}
            elif mutation == "duplicate":
                report["checks"][0]["id"] = report["checks"][1]["id"]
            else:
                del report["checks"][0]["id"]
            variants.append(report)
        for index, report in enumerate(variants):
            with self.subTest(index=index):
                self.assertFalse(summarize_suite("graph", report)["checks_passed"])

    def test_live_evaluation_or_unsafe_context_cannot_pass_offline_gate(self):
        for field in ("live_answer_correctness_evaluated", "answer_behavior_evaluated",
                      "real_provider_calls", "unsafe_outbound_context_detected", "unauthorized_outbound_context_detected"):
            with self.subTest(field=field):
                self.assertFalse(summarize_suite("graph", {**self.sample_report(), field: True})["checks_passed"])

    def test_renderer_keeps_failure_visible_and_does_not_copy_raw_answers(self):
        report = {"timestamp": "test", "status": "failed", "real_provider_calls": 0,
                  "validation_inputs_sha256": "digest", "failed_phase": "graph",
                  "raw_answer": "PRIVATE_PROVIDER_BODY", "suites": []}
        markdown = render_markdown(report)
        self.assertIn("**failed**", markdown)
        self.assertIn("Failed phase: `graph`", markdown)
        self.assertNotIn("PRIVATE_PROVIDER_BODY", markdown)
        self.assertIn("does not certify production readiness", markdown)


if __name__ == "__main__":
    unittest.main()
