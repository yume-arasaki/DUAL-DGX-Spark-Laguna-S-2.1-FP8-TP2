#!/bin/bash
# NCCL/RoCE + vLLM env for the Laguna TP2 recipe.
# MUST be sourced before `vllm serve` AND before `ray start` on both nodes.
# NCCL_* propagate to Ray workers via vLLM.
set -uo pipefail

export NCCL_IB_HCA=rocep1s0f1
export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_DISABLE=0

# dmabuf GDR knobs — harmless; GDR does NOT actually engage on GB10 (unified memory),
# kept for clarity and in case future driver versions enable it:
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1
export NCCL_DMABUF_ENABLE=1

# Per-step worker RPC timeout: low on purpose so any cross-node stall -> fast
# EngineDeadError -> watchdog auto-relaunches within minutes instead of hanging ~20min.
# Do NOT raise this unless spec decode is re-enabled (which makes steps legitimately long).
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300

export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1

# For dual-rail experiments only (NOT used by default; NCCL won't bind the 2nd PCIe fn to 1 GPU):
# export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_CROSS_NIC=1 NCCL_IB_MERGE_NICS=1
