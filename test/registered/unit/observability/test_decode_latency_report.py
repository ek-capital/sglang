# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.srt.observability.decode_latency_report import analyze_decode_latency


def _record(*, bs, commits, step, draft, target):
    return {
        "bs": bs,
        "step_gpu_ms": step,
        "prepare_window_gpu_ms": 0.0,
        "draft_gpu_ms": draft,
        "confidence_budget_gpu_ms": 0.0,
        "schedule_layout_gpu_ms": 0.0,
        "target_verify_gpu_ms": target,
        "accept_finalize_gpu_ms": 0.0,
        "state_commit_gpu_ms": 0.0,
        "reqs": [{"acc_len": commit} for commit in commits],
    }


class TestDecodeLatencyReport(unittest.TestCase):
    def test_request_weighted_committed_token_normalization(self):
        report = analyze_decode_latency(
            {
                "mode": "compact",
                "records": [
                    _record(bs=2, commits=[3, 1], step=10.0, draft=2.0, target=5.0),
                    _record(bs=1, commits=[2], step=6.0, draft=1.0, target=3.0),
                ],
            }
        )
        self.assertEqual(report["committed_tokens"], 6)
        self.assertEqual(report["request_rounds"], 3)
        self.assertAlmostEqual(report["avg_committed_tokens_per_request_round"], 2)
        self.assertAlmostEqual(report["avg_gpu_ms_per_next_token"], 26 / 6)
        self.assertAlmostEqual(report["gpu_service_ms_per_token"], 16 / 6)

        phases = {row["phase"]: row for row in report["phases"]}
        self.assertAlmostEqual(phases["draft"]["avg_ms_per_next_token"], 5 / 6)
        self.assertAlmostEqual(phases["target_verify"]["avg_ms_per_next_token"], 13 / 6)
        self.assertAlmostEqual(
            phases["runtime_unattributed"]["avg_ms_per_next_token"], 8 / 6
        )
        self.assertAlmostEqual(report["phase_timing_coverage"], 18 / 26)

    def test_missing_commit_lengths_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "commit lengths"):
            analyze_decode_latency(
                {"records": [{"bs": 1, "step_gpu_ms": 1.0, "reqs": None}]}
            )

    def test_incomplete_records_are_skipped(self):
        report = analyze_decode_latency(
            {
                "records": [
                    {"bs": 2, "reqs": [{"acc_len": 1}, {"acc_len": 1}]},
                    _record(bs=1, commits=[1], step=2.0, draft=0.5, target=1.0),
                ]
            }
        )
        self.assertEqual(report["rounds"], 1)
        self.assertEqual(report["skipped_records"], 1)


if __name__ == "__main__":
    unittest.main()
