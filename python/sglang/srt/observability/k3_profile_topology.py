# SPDX-License-Identifier: Apache-2.0
"""Generate reproducible Kimi K3 profile/capture launch commands.

The presets intentionally encode topology-sensitive choices only.  Capacity
and workload-sensitive values (notably the Mamba full-memory ratio) remain
required inputs so a convenient preset cannot silently become a bad recipe.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class K3Topology:
    name: str
    nodes: int
    gpus_per_node: int
    tp_size: int
    ep_size: int
    dcp_size: int
    decode_attention_backend: str | None
    moe_runner_backend: str | None
    enable_symm_mem: bool

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node


TOPOLOGIES = {
    "b300-8": K3Topology(
        name="b300-8",
        nodes=1,
        gpus_per_node=8,
        tp_size=8,
        ep_size=1,
        dcp_size=8,
        decode_attention_backend=None,
        moe_runner_backend=None,
        enable_symm_mem=False,
    ),
    "h200-16": K3Topology(
        name="h200-16",
        nodes=2,
        gpus_per_node=8,
        tp_size=16,
        ep_size=16,
        dcp_size=1,
        decode_attention_backend="flashmla",
        moe_runner_backend="marlin",
        enable_symm_mem=True,
    ),
}


def build_launch(
    topology_name: str,
    *,
    model_path: str,
    draft_model_path: str,
    mamba_full_memory_ratio: float,
    node_rank: int = 0,
    dist_init_addr: str | None = None,
    host: str = "0.0.0.0",
    port: int = 30000,
    context_length: int = 262_144,
    mem_fraction_static: float = 0.88,
    profile: bool = False,
    capture_plan: str | None = None,
    extra_args: Sequence[str] = (),
) -> dict[str, object]:
    topology = TOPOLOGIES[topology_name]
    if not 0 <= node_rank < topology.nodes:
        raise ValueError(f"node_rank must be in [0, {topology.nodes})")
    if topology.nodes > 1 and not dist_init_addr:
        raise ValueError("dist_init_addr is required for a multi-node topology")
    if not 0 < mamba_full_memory_ratio <= 1:
        raise ValueError("mamba_full_memory_ratio must be in (0, 1]")
    if profile and capture_plan:
        raise ValueError("profile and capture are separate passes")

    argv = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        str(topology.tp_size),
        "--ep-size",
        str(topology.ep_size),
        "--dcp-size",
        str(topology.dcp_size),
        "--nnodes",
        str(topology.nodes),
        "--node-rank",
        str(node_rank),
        "--context-length",
        str(context_length),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--mamba-full-memory-ratio",
        str(mamba_full_memory_ratio),
        "--trust-remote-code",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        draft_model_path,
        "--speculative-dspark-block-size",
        "7",
    ]
    if dist_init_addr:
        argv.extend(("--dist-init-addr", dist_init_addr))
    if topology.decode_attention_backend:
        argv.extend(("--decode-attention-backend", topology.decode_attention_backend))
    if topology.moe_runner_backend:
        argv.extend(("--moe-runner-backend", topology.moe_runner_backend))
    if topology.enable_symm_mem:
        argv.append("--enable-symm-mem")
    argv.extend(extra_args)

    env: dict[str, str] = {}
    if profile:
        env["SGLANG_HOTLOOP_PROFILE"] = "1"
    if capture_plan:
        env["SGLANG_REPLAY_CAPTURE_PLAN"] = capture_plan
    if topology.name == "h200-16":
        # These values are part of the official two-node recipe.  Interface
        # selection remains an operator input because cloud NIC names vary.
        env.update({"NCCL_MNNVL_ENABLE": "1", "NCCL_CUMEM_ENABLE": "1"})
    return {
        "schema_version": 1,
        "topology": {**asdict(topology), "world_size": topology.world_size},
        "node_rank": node_rank,
        "environment": env,
        "argv": argv,
        "shell_command": " ".join(
            [*(f"{key}={shlex.quote(value)}" for key, value in env.items())]
            + [shlex.join(argv)]
        ),
        "operator_requirements": (
            [
                "Set GLOO_SOCKET_IFNAME, NCCL_SOCKET_IFNAME, and SGLANG_HOST_IP "
                "to the routable cross-node interface/address on both nodes."
            ]
            if topology.nodes > 1
            else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", choices=sorted(TOPOLOGIES))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--mamba-full-memory-ratio", type=float, required=True)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--dist-init-addr")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--context-length", type=int, default=262_144)
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--capture-plan")
    args, extra = parser.parse_known_args()
    result = build_launch(
        args.topology,
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        mamba_full_memory_ratio=args.mamba_full_memory_ratio,
        node_rank=args.node_rank,
        dist_init_addr=args.dist_init_addr,
        host=args.host,
        port=args.port,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        profile=args.profile,
        capture_plan=args.capture_plan,
        extra_args=extra,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
