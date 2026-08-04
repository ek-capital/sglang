# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.debug_utils.profile_guided_plan import compile_capture_plan
from sglang.srt.observability.hotloop_report import analyze_traces, select_sections
from sglang.srt.observability.hotloop_sections import REGISTRY
from sglang.srt.observability.k3_profile_topology import build_launch


class TestProfileGuidedCapture(unittest.TestCase):
    def test_kernel_ownership_is_exclusive_and_priority_ordered(self):
        self.assertEqual(
            REGISTRY.classify_kernel("shared_expert_gemm", "kimi_k3"),
            "moe.shared_experts",
        )
        self.assertEqual(
            REGISTRY.classify_kernel("kda_state_commit_kernel", "kimi_k3"),
            "speculative.state_commit",
        )

    def test_report_uses_critical_rank_and_selects_top_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank0 = root / "rank0.json"
            rank1 = root / "rank1.json"
            rank0.write_text(
                json.dumps(
                    {
                        "rank": 0,
                        "decode_steps": 2,
                        "kernel_events": [
                            {"name": "dspark_draft_gemm", "duration_us": 6000},
                            {"name": "flashinfer_moe_gemm", "duration_us": 3000},
                            {"name": "causal_conv1d", "duration_us": 1000},
                        ],
                    }
                )
            )
            rank1.write_text(
                json.dumps(
                    {
                        "rank": 1,
                        "decode_steps": 2,
                        "kernel_events": [
                            {"name": "dspark_draft_gemm", "duration_us": 8000},
                            {"name": "flashinfer_moe_gemm", "duration_us": 4000},
                            {"name": "causal_conv1d", "duration_us": 2000},
                        ],
                    }
                )
            )
            report = analyze_traces((rank0, rank1), model_family="kimi_k3")
            self.assertEqual(report["critical_rank"], 1)
            self.assertAlmostEqual(report["total_attributed_ms_per_step"], 7.0)
            selection = select_sections(report, 3)
            self.assertEqual(
                [row["section"] for row in selection["selected"]],
                ["speculative.draft", "moe.routed_experts", "attention.kda"],
            )

    def test_report_preserves_explicit_architecture_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "rank0.json"
            trace.write_text(
                json.dumps(
                    {
                        "rank": 0,
                        "decode_steps": 2,
                        "kernel_events": [
                            {
                                "name": "nccl_allreduce",
                                "duration_us": 2000,
                                "architecture_path": (
                                    "target_verify.target.attention.kda."
                                    "output_projection.tp_allreduce"
                                ),
                            },
                            {
                                "name": "flashinfer_moe_gemm",
                                "duration_us": 1000,
                                "architecture_path": (
                                    "target_verify.target.moe.experts.w13"
                                ),
                            },
                        ],
                    }
                )
            )
            report = analyze_traces((trace,), model_family="kimi_k3")
            architecture = {row["path"]: row for row in report["architecture_sections"]}
            self.assertAlmostEqual(report["architecture_attribution_coverage"], 1)
            self.assertAlmostEqual(
                architecture[
                    "target_verify.target.attention.kda."
                    "output_projection.tp_allreduce"
                ]["time_ms"],
                1.0,
            )
            self.assertEqual(
                architecture["target_verify.target.moe.experts.w13"]["attribution"],
                "explicit",
            )

    def test_report_joins_kernel_external_id_to_semantic_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "rank0.json"
            trace.write_text(
                json.dumps(
                    {
                        "rank": 0,
                        "decode_steps": 1,
                        "traceEvents": [
                            {
                                "name": (
                                    "sglang.hotloop/attention.kda.recurrence/"
                                    "model_role=target,phase=target_verify"
                                ),
                                "cat": "cpu_op",
                                "pid": 10,
                                "tid": 20,
                                "ts": 0,
                                "dur": 100,
                                "args": {},
                            },
                            {
                                "name": "launch recurrent kernel",
                                "cat": "cpu_op",
                                "pid": 10,
                                "tid": 20,
                                "ts": 10,
                                "dur": 5,
                                "args": {"External id": 7},
                            },
                            {
                                "name": "fused_recurrent_kernel",
                                "cat": "kernel",
                                "pid": 0,
                                "tid": 1,
                                "ts": 200,
                                "dur": 250,
                                "args": {"External id": 7},
                            },
                        ],
                    }
                )
            )
            report = analyze_traces((trace,), model_family="kimi_k3")
            self.assertEqual(
                report["architecture_sections"][0]["path"],
                "target_verify.target.attention.kda.recurrence",
            )
            self.assertEqual(
                report["architecture_sections"][0]["attribution"],
                "semantic_correlation",
            )

    def test_plan_is_bounded_and_rank_complete(self):
        selection = {
            "model_family": "kimi_k3",
            "timing_basis": "exclusive_kernel",
            "selected_share": 0.9,
            "selected": [
                {"section": "speculative.draft"},
                {"section": "attention.kda"},
            ],
        }
        plan = compile_capture_plan(
            selection,
            case_bytes={
                "speculative.draft": 1000,
                "attention.kda": 2000,
            },
            cases_per_section=4,
            world_size=16,
        )
        self.assertTrue(plan["require_all_ranks"])
        self.assertEqual(plan["ranks"], "all")
        self.assertEqual(plan["budget"]["estimated_bytes"], 132_000)
        self.assertNotIn("points", plan)
        self.assertIn("speculative.draft_generation", plan["operations"])
        self.assertNotIn("speculative.draft_round", plan["operations"])

    def test_b300_and_h200_launches(self):
        b300 = build_launch(
            "b300-8",
            model_path="/mnt/kimi3",
            draft_model_path="/mnt/kimi3-draft",
            mamba_full_memory_ratio=0.2,
            profile=True,
        )
        self.assertEqual(b300["topology"]["world_size"], 8)
        self.assertIn("--dcp-size", b300["argv"])
        self.assertNotIn("--enable-symm-mem", b300["argv"])
        self.assertEqual(b300["environment"]["SGLANG_HOTLOOP_PROFILE"], "1")

        h200 = build_launch(
            "h200-16",
            model_path="/mnt/kimi3",
            draft_model_path="/mnt/kimi3-draft",
            mamba_full_memory_ratio=0.2,
            node_rank=1,
            dist_init_addr="10.0.0.1:5000",
            capture_plan="/mnt/capture/plan.json",
        )
        self.assertIn("--enable-symm-mem", h200["argv"])
        self.assertIn("flashmla", h200["argv"])
        self.assertEqual(h200["environment"]["NCCL_MNNVL_ENABLE"], "1")
        self.assertEqual(
            h200["environment"]["SGLANG_REPLAY_CAPTURE_PLAN"],
            "/mnt/capture/plan.json",
        )

    def test_h200_requires_rendezvous(self):
        with self.assertRaisesRegex(ValueError, "dist_init_addr"):
            build_launch(
                "h200-16",
                model_path="m",
                draft_model_path="d",
                mamba_full_memory_ratio=0.2,
            )


if __name__ == "__main__":
    unittest.main()
