# Laguna S 2.1 FP8 — Dual DGX Spark TP=2 Recipe

> A stable, reproducible config for running poolside Laguna-S-2.1-FP8 across two NVIDIA DGX Spark (GB10) nodes with vLLM tensor-parallel.

## The problem

Every published config for Laguna S 2.1 breaks under real agent load on GB10 cross-node TP:

- **Decode deadlocks** — one GPU pins at 96%, the other at 0%, engine dies
- **Doom loops past 46K context** — model repeats tool calls, reasoning collapses
- **Broken reasoning output** — `</think>` tags leak into content
- **Sudden tok/s drops** and freezes during sustained generation

This recipe fixes all four. The trade-off: you lose speculative decoding. Everything else works — 23 tok/s decode flat to 200K context, tool calling clean, reasoning properly separated.

## Hardware requirements

- 2× NVIDIA DGX Spark (GB10 Grace-Blackwell, 128GB unified memory each)
- ConnectX-7 RoCE link (QSFP cable between nodes, 200 GbE nominal)
- Ubuntu 24.04 aarch64, Python 3.12, CUDA 13.0

## Quick start

```bash
# 1. Clone this repo to both Sparks
git clone <your-repo-url> ~/laguna-recipe
cd ~/laguna-recipe

# 2. Run the setup script on BOTH nodes (installs venv, vLLM, dependencies)
bash scripts/setup.sh

# 3. Download the model to spark-1 (113 GB, needs HuggingFace token)
bash scripts/download-model.sh

# 4. Mirror to spark-2
bash scripts/mirror-to-spark2.sh

# 5. Start Ray cluster
bash scripts/start-ray.sh

# 6. Launch vLLM
bash scripts/load.sh

# Wait 8-10 minutes for model load (112 GiB across 24 shards)
# Then verify:
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

## What this recipe does

| Finding | Problem | Fix |
|---------|---------|-----|
| 01 | CUDA graphs deadlock cross-node NCCL all-reduce on GB10 (no GPUDirect) | `--enforce-eager` |
| 02 | NVFP4 single-node doom-loops past 46K (bandwidth starvation) | TP=2 halves KV per node, pushes crossover to 90K |
| 03 | Reasoning parser defaults to non-thinking mode | One-line parser patch (default thinking=True) |
| 04 | Speculative decoding loses on cross-node TP2 (draft pays full collective toll) | Disable spec decode entirely |

## Verified performance (2026-07-28)

All numbers measured on real hardware. Not vendor claims.

- **Decode:** ~23 tok/s single stream, ~58 tok/s at 8 concurrent
- **Context:** flat throughput from 2K to 200K (sliding-window-512 attention)
- **KV cache:** 2,037,098 tokens (24 GB), ~13 concurrent 150K sessions
- **Fabric:** 1.43µs latency, ~110 Gb/s per rail
- **Tool calling:** clean, zero drops past 200K context
- **Cold start:** 8-10 minutes (112 GiB weight load across 24 shards)

## Model scale

- **113 GiB checkpoint** (FP8 block-scaled, 24 shards × ~4.8 GB)
- **120B total parameters** (BF16 uncompressed: 241 GB)
- **12,288 expert FFNs** (256 experts × 48 layers)
- **8.2B active params per token** (10 routed + 1 shared expert per layer)
- **6.8% active ratio** — this is why decode is fast despite massive size
- **256K max context** (262,144 tokens)

## Software stack (pinned)

- vLLM 0.25.1
- Driver 580.159.03 (open kernel module)
- NCCL 2.28.9
- CUDA 13.0
- Ray 2.56.1
- PyTorch 2.11.0

## Files

```
├── README.md                  # This file
├── config/
│   ├── config.yaml            # vLLM native serve config
│   ├── nccl-env.sh            # NCCL/RoCE environment variables
│   └── poolside_v1_reasoning_parser.py  # Patched reasoning parser
├── scripts/
│   ├── setup.sh               # One-time setup (both nodes)
│   ├── download-model.sh      # Download FP8 checkpoint to spark-1
│   ├── mirror-to-spark2.sh    # rsync weights + draft to spark-2
│   ├── create-symlink.sh      # Path symlink on spark-2 (critical)
│   ├── patch-parser.sh        # Apply reasoning parser patch to venv
│   ├── start-ray.sh           # Start Ray head + worker
│   ├── load.sh                # Launch vLLM (the main entry point)
│   ├── test.sh                # Health check + decode speed test
│   └── watchdog.sh            # Auto-recovery on engine death
└── docs/
    └── FINDINGS.md            # Full debugging journal (EXP 1-13)
```

## The key insight

On GB10 cross-node TP, the ~23 tok/s decode ceiling is architectural — cross-node collective latency without GPUDirect. Not a config miss. Every lever was tested: MTU, dual-rail, dmabuf GDR, pipeline-parallel, NCCL protocol. None moved it.

Speculative decoding can lose on slow interconnects. Acceptance has to beat the per-draft collective toll. On no-GDR cross-node TP2, it doesn't.

**Test sustained decode, not short bursts.** The first few decode tokens after prefill have startup overhead. A 3-token generation reports 5 tok/s when steady-state is actually 23 tok/s.

## Disclaimer

Speculative decoding was tested with 3 configs (W4A16 draft, FP8 DFlash + DeepGEMM, FP8 DFlash + Triton), all net-negative on this fabric. However, this was tested primarily on prose. Structured output, tool-call traffic, and code may accept higher. If someone cracks spec decode on cross-node TP2, open an issue — I'd love to compare notes.

## License

MIT. Use it, fork it, improve it.

## Acknowledgments

- Recipe approach inspired by [MiaAI-Lab's Dual DGX Spark recipe](https://github.com/MiaAI-Lab/Dual-DGX-Spark-Step-3.7-Flash-NVFP4)
- Model: poolside Laguna-S-2.1
- Community benchmarks: BlackwellBoy, gitcommit90
- [howtospark.com](https://howtospark.com) recipe reference
