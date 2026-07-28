# DGX Spark TP=2 + ConnectX-7 RoCE — Behavior Study

Goal: map how the 2-node tensor-parallel vLLM cluster (Laguna-S-2.1-FP8, GB10 / sm_121,
NCCL 2.28.9 over RoCE on rocep1s0f1 / 10.0.0.x) actually behaves, so we understand the
Sparks + cross-node TP rather than chase single toggles. Log every experiment.

Hardware: 2× GB10 (Grace-Blackwell, sm_121, ~128GB unified LPDDR5X each). Direct
ConnectX-7 QSFP, 200 Gb/s, RoCEv2. Model is a large MoE (FP8 checkpoint ~112 GiB) →
does NOT fit one node → cross-node TP=2 is required. NCCL 2.28.9+cuda13. No GPUDirect
(GDR 0 in all NCCL logs; host bounce buffers). sudo needs a password (can't modprobe
nvidia_peermem or change MTU).

## THE CORE PHENOMENON (as of 07-27 ~15:00)
Sustained cross-node DECODE deadlocks. Signature: prefill runs fine (BOTH GPUs ~90%),
then during decode ONE rank spins at 96% (stuck in the all-reduce) while the OTHER sits
at 0% (never enters the collective) → request hangs → 300s per-step RPC timeout →
EngineDeadError → process exits → watchdog relaunches. Which rank spins varies run to run
(seen both spark-1-spins and spark-2-spins) = a RACE, not a fixed-rank fault.

Timing varies: sometimes hangs at the very first decode step, sometimes after ~200-300
tokens (~20s). Variability = race, consistent with user's "it worked for a while."

## RULED OUT (each independently reproduced the hang)
- Attention backend: FlashInfer AND Triton both hang in sustained decode. (Triton ALSO
  has a separate JIT-during-inference stall, but that was a red herring for THIS deadlock.)
- Stale Ray state: full clean `ray stop` both nodes + fresh head+worker → still hangs.
- Async scheduling: `--no-async-scheduling` (confirmed "Asynchronous scheduling is
  disabled") → still hangs.
- Physical link / RoCE hardware: rocep1s0f1 ACTIVE, 200 Gb/s, link_downed=0,
  port_rcv_errors=0, port_xmit_discards=0. NCCL selects NET/IB RoCE (not TCP fallback).
  PREFILL does large cross-node all-reduces successfully → the RDMA path itself works.
  Therefore the fault is DECODE-SPECIFIC software, not the fabric.
- KV/OOM: kv_cache_usage ~0.1-0.2% when it hangs on short requests. Not memory.

## KEY DISCRIMINATOR
Prefill (works) vs decode (hangs). Biggest difference = decode replays CUDA graphs
(cudagraph_mode FULL_AND_PIECEWISE); prefill does not. → Current hypothesis: a
CUDA-graph-captured cross-node NCCL all-reduce deadlocks on GB10. Testing --enforce-eager.

## EARLIER (SEPARATE, real) BUGS already fixed this arc
1. Spec decode (DFlash k=7) sample_tokens cross-node hang at long ctx → fixed by k=0
   (vLLM #40926/#41530). Also k=7 only got 17% draft acceptance here (net-negative).
2. FlashInfer prefill kernel hang >~128k positions on sm_121 → Triton avoided it, but
   Triton introduced the JIT stall, so we're back on FlashInfer.
NOTE: my earlier "150k usable" validation used gen=32 (short) requests that finished
before the decode-deadlock window — so it MISSED this bug. Lesson: always test SUSTAINED
decode (hundreds of tokens), not just TTFT + a few tokens.

## CONFIG KNOBS (script: ~/spark-tp2-ray-laguna.sh, backups .bak-*)
k (0=spec off), ATTENTION_BACKEND (FLASHINFER default), --disable-custom-all-reduce,
--no-async-scheduling, --enforce-eager (testing), VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300.
Watchdog: ~/laguna-watchdog.sh + cron (CURRENTLY PAUSED during study).

## EXPERIMENT LOG
(appended below as we go; structured data in experiments.jsonl)

### EXP 1 (07-27 ~15:50): --enforce-eager  -> FIXES DECODE DEADLOCK
Config: FLASHINFER + k=0 + --no-async-scheduling + --enforce-eager + timeout 300, fresh cluster.
2x sustained 400-tok requests BOTH completed, decode ~23 tok/s, BOTH GPUs balanced ~85-90%
during decode (no 96%/0% split). => ROOT CAUSE CONFIRMED: CUDA-graph-captured cross-node
NCCL all-reduce deadlocks on GB10 (sm_121). Eager issues collectives live each step -> both
ranks participate -> no deadlock. Cost: lose cudagraph decode speedup (acceptable; cross-node
is the bottleneck anyway). Next: stress with many/longer/concurrent requests to confirm.

### EXP 2 (07-27 ~15:55): eager sequential stress = STABLE
6x sustained (500-tok, 800-tok prompt) requests, ALL OK, decode 23.3-23.5 tok/s steady,
both GPUs balanced ~90/88% during decode, server healthy after. 8/8 sustained requests
across EXP1+2. Eager confirmed stable for sequential load. (imbalance=True flags were the
first GPU sample catching the prefill instant, not real deadlocks.)

### EXP 3 (07-27 ~15:58): eager concurrency = STABLE
4 simultaneous 300-tok requests ALL OK (http 200), both GPUs balanced 92/90% throughout,
~21-24s each (~50 tok/s aggregate batched). No deadlock under concurrent load.

### EXP 4 (07-27 ~15:59): 150k long context under eager = WORKS
Cold 150k prefill + decode: OK, TTFT 51.5s (~2900 tok/s prefill burst over bounce path),
decode 20.8 tok/s, both GPUs balanced 96/96%. 3 transient shm-stall warnings but PUSHED
THROUGH (no hard deadlock). => the earlier ">128k FlashInfer prefill hangs" were largely
the SAME cudagraph/race deadlock; eager resolves them too. Eager config now handles
sequential + concurrent + 150k. This is the stable baseline.

### EXP 5 (07-27 ~16:02): raw RoCE fabric bandwidth (ib_write_bw)
65536B writes: BW average 109.03 Gb/sec (of 200 nominal = ~55% line rate). MTU=1024 (small;
RoCE inefficient vs 4096 -> raising MTU to 4096 likely ~1.7-1.9x, but needs sudo). MsgRate
0.21 Mpps. => interconnect functional but MTU-limited. Prefill (bandwidth-bound) sees ~this.

### EXP 6 (07-27 ~16:03): raw RoCE fabric latency (ib_write_lat)
2-byte RDMA write: t_typical 1.43 us, t_avg 1.43, 99% 1.52 us. => fabric latency is GOOD
(NVLink-class). NOT the bottleneck. Implication: eager decode (~23 tok/s) is slow because
eager launches every kernel + all cross-node collectives per token with no CUDA-graph
batching (2 all-reduce x 48 layers = 96 cross-node round-trips/token) + host bounce (no GDR).
CUDA graphs would batch launches -> faster decode, BUT full graphs deadlock the cross-node
NCCL all-reduce. KEY OPTIMIZATION: find a cudagraph mode that batches compute yet keeps
collectives safe (test PIECEWISE).

Fabric summary: 200Gb nominal | 109 Gb/s write BW (MTU 1024-limited) | 1.43us latency | GDR off.

### OPS LESSON (07-27 ~16:58): Ray placement-group leak on kill -9
kill -9 of the vllm driver LEAKS its Ray placement group: `ray status` then shows
"2.0/2.0 GPU (2.0 used ... in placement groups)" with ZERO vllm procs, and the next
launch fails: "Cannot provide a placement group requiring 2.0 GPUs within 1800s".
Fix: graceful SIGTERM (pkill without -9) lets vllm release the PG; or ray stop/start to
clear. WATCHDOG IMPLICATION: relaunch must ensure 2 GPUs are actually free (graceful kill,
and/or verify `ray status` shows them free) before starting, else it fails to recover.

### SUDO-GATED OPTIMIZATIONS (identified 07-27, not yet applied - sudo needs password)
1. RoCE MTU 1024->4096 (`ip link set enp1s0f1np1 mtu 4096` both nodes): fabric BW
   109->~180 Gb/s. Helps prefill/long-ctx TTFT (bandwidth-bound).
2. GPUDirect RDMA (`modprobe nvidia_peermem` both nodes): remove host-bounce in NCCL;
   could speed cross-node collectives AND possibly make cudagraphs viable. (GB10 GDR
   support uncertain per MiniMax report, but untested here.)
3. MPI for nccl-tests all_reduce_perf (definitive collective bench). Low priority.

### EXP 7 (07-27 ~17:07): PIECEWISE cudagraph = STABLE but NOT faster
4/4 sustained 400-tok requests OK, balanced ~92/85%, decode 21.3-22.1 tok/s (vs eager ~23).
=> PIECEWISE avoids the deadlock (only FULL graph captures the cross-node all-reduce) BUT
gives NO speedup. KEY INSIGHT: decode is COLLECTIVE-BOUND (96 cross-node all-reduces/token
over host-bounce RoCE), not kernel-launch-bound. So cudagraph batching of launches does not
help; eager == piecewise speed. Decode ceiling ~22 tok/s is set by cross-node collective cost.
=> To raise it: GPUDirect (remove host bounce) and/or MTU 4096. FULL cudagraph MIGHT also
become viable+faster if GDR makes the all-reduce device-only/graph-capturable. All need sudo.
DECISION: eager is the recommended stable config (simplest, equal speed to piecewise).

### EXP 8 (07-27 ~17:16): throughput curve, eager (recommended config)
ctx 2k: TTFT 1.1s, decode 23.2 tok/s | 16k: 3.9s, 23.4 | 64k: 14.6s, 22.1 | 128k: 24.0s, 21.0
(warm/prefix-cached TTFT). Decode ~flat 21-23 tok/s across context (sliding-window attn keeps
decode cheap); TTFT ~linear in prefill length. Cold long prefill much slower (120k cold ~191s
vs warm 128k ~24s) - prefix caching is decisive for TTFT. This is the stable usable envelope.

## CURRENT STATE / RECOMMENDED CONFIG (07-27 ~17:16)
Running: FLASHINFER + k=0 (spec off) + --disable-custom-all-reduce + --no-async-scheduling +
--enforce-eager + timeout 300. Handles seq + concurrent + 150k. ~22 tok/s decode.
Watchdog re-enabled. STABLE. Ceiling (~22 tok/s decode) is cross-node-collective-bound;
raising it needs GDR/MTU (sudo).

### EXP 9 (07-27 ~17:27): NCCL_PROTO=LL = negligible (23.8 tok/s vs ~22-23 baseline)
Within noise. Confirms decode collective-bound; no-sudo NCCL tuning cannot move the ceiling.

### === SUDO UNLOCKED (07-27 ~17:30) - applying MTU 4096 + GPUDirect ===

### EXP 10 (07-27 ~17:35): SUDO experiments - MTU + GDR
- MTU: set netdev 9000 both nodes -> RoCE active_mtu 1024 -> 4096 (verified). BUT ib_write_bw
  UNCHANGED: 109.03 (MTU1024) -> 109.22 (MTU4096) -> 111.63 (MTU4096, 8 QPs). So bandwidth is
  NOT MTU-limited and NOT single-QP-limited.
- => ~110 Gb/s is a HARD platform ceiling (of 200 GbE nominal, ~55%). Almost certainly the
  PCIe/C2C link between the ConnectX-7 and the GB10 (NIC likely on Gen5 x8, not x16). Cannot
  be raised by config.
- GPUDirect: nvidia_peermem modprobe FAILS "Invalid argument" on both. Driver is the OPEN
  kernel module (580.159.03) which uses dmabuf, not the legacy peer-mem API. NCCL shows GDR 0
  and dmabuf isnt engaging either. = genuinely UNAVAILABLE on GB10 this way. Host-bounce

### EXP 10 (07-27 ~17:35): SUDO experiments - MTU + GDR (root unlocked)
- MTU: set netdev 9000 both nodes -> RoCE active_mtu 1024 -> 4096 (verified). BUT ib_write_bw
  UNCHANGED: 109.03 (MTU1024) -> 109.22 (MTU4096) -> 111.63 (MTU4096, 8 QPs). Bandwidth is
  NOT MTU-limited and NOT single-QP-limited.
- ~110 Gb/s is a HARD platform ceiling (of 200 GbE nominal, ~55%). Almost certainly the
  PCIe/C2C link between the ConnectX-7 and the GB10 (NIC likely Gen5 x8, not x16). Not
  raisable by config.
- GPUDirect: nvidia_peermem modprobe FAILS "Invalid argument" both nodes. Driver is the OPEN
  kernel module 580.159.03 which uses dmabuf, not the legacy peer-mem API nvidia_peermem
  expects. NCCL shows GDR 0; dmabuf not engaging either. GDR genuinely UNAVAILABLE on GB10.
  Host-bounce collectives are unavoidable.

## STUDY CONCLUSION: the ~22 tok/s decode ceiling is ARCHITECTURAL on GB10 cross-node TP.
No working GPUDirect -> collectives bounce through host memory -> decode is collective-bound
(~22 tok/s) AND CUDA graphs cannot capture the CPU-proxy path -> --enforce-eager required.
Fabric caps ~110 Gb/s. None of this moves with config / NCCL protocol / MTU / QP-count.
The eager config already reaches this ceiling. Going faster needs different interconnect
(real NVLink / working GDR) or fewer cross-node hops: pipeline-parallel instead of
tensor-parallel (fewer, larger transfers per token vs 96 small all-reduces), or a model
that fits a single GB10 (no cross-node at all).

MTU left at 9000/active_mtu 4096 (harmless, non-persistent - resets on reboot; not set in
serve/ray scripts). To persist would need netplan/networkd config.

### EXP 11 (07-27 ~17:52): PP=2 vs TP=2 (pipeline vs tensor parallel)
PP=2 single-stream: 13-15 tok/s (SLOWER than TP 22) - pipeline bubble, each node ~30-70%% util
for one sequence. PP=2 8-concurrent: 36.1 tok/s aggregate (8/8 ok). TP=2 was ~50 tok/s at only
4-concurrent. => TP=2 BEATS PP=2 at both low and moderate concurrency for this MoE on host-bounce
fabric. PP not worth it. TP=2 eager remains the production config. (PP would only pull ahead at
very high batch that fills the pipeline AND if compute-bound, not the case here.)

### EXP 12 (07-27 ~18:03): TP=2 concurrency confirms it beats PP=2
TP=2 eager: single-stream 23 tok/s; 8-concurrent 57.7 tok/s aggregate (8/8 ok). vs PP=2 36.1.
TP=2 eager is the definitive production config: fastest at both low and high concurrency, stable.

## FINAL PRODUCTION CONFIG (running 07-27 ~18:03)
FLASHINFER + k=0 + --disable-custom-all-reduce + --no-async-scheduling + --enforce-eager +
TP_SIZE=2 PP_SIZE=1 + VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300, MTU 4096. Watchdog re-enabled.
Single ~23 tok/s, 8-concurrent ~58 tok/s agg, handles 150k. This is the GB10 cross-node ceiling.

### EXP 13 (07-27 ~18:20): ONLINE RESEARCH corrections + dmabuf GDR + dual-rail
Research (NVIDIA forums, ArgentAIOS dgx-spark-cluster playbook, ServeTheHome GB10 teardown):
- nvidia_peermem fails on GB10 aarch64 (missing ib_register_peer_memory_client) - EXPECTED.
  DMA-BUF is the intended GPUDirect path; open driver 580 + kernel 6.x support it. Env:
  NCCL_DMABUF_ENABLE=1, NCCL_NET_GDR_LEVEL=5 (>4 => SYS), NCCL_NET_GDR_READ=1.
- ConnectX-7 on GB10 is MULTI-HOST mode: two PCIe Gen5 x4 links (~100 Gb/s each) = 200 nominal.
  One link ~92-109 Gb/s (matches our earlier 109). Both links ~185-190 in NVIDIA field reports.

dmabuf GDR TEST: added the 3 env vars, relaunched. NCCL log STILL "GDR 0" - dmabuf did NOT
engage (no "GPU Direct RDMA Enabled" line). GDR_LEVEL=5 is valid (=SYS) so not that. Likely
either dmabuf verbs registration not succeeding on this NIC/driver combo, OR moot because GB10
unified Grace-Blackwell memory means the NIC already DMAs from the same coherent pool the
tensors live in (no GPU-HBM bounce to avoid). => decode ceiling ~22 tok/s (latency-bound) stands.
Needs NCCL_DEBUG_SUBSYS=NET to get the exact reason (future).

DUAL-RAIL TEST (real win): the 2nd ConnectX-7 function roceP2p1s0f1 / enP2p1s0f1np1 was UP but
had no IP. Configured 10.0.1.1/10.0.1.2 (sudo) - and it PINGS spark-2 (0.77ms) => 2nd rail is
physically connected. ib_write_bw both rails concurrently: rail1(10.0.0)=109 + rail2(10.0.1)=50
= ~159 Gb/s aggregate (vs 109 single, +46%). rail2 asymmetric (~50) - 2nd PCIe function path.
=> DUAL-RAIL is a real ~40-46% bandwidth lever for PREFILL / long-ctx TTFT (bandwidth-bound),
NOT for decode (latency-bound). To use in vLLM: NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 + persist
the 2nd-link IPs (currently runtime-only, reset on reboot).

### EXP 14 (07-27 ~20:05): dual-rail in vLLM = needs more than NCCL_IB_HCA list
Launched with NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 but worker env ended up NCCL_IB_HCA=rocep1s0f1
only, and NCCL log shows "Using [0]rocep1s0f1" (single rail). So multi-rail NCCL over two RoCE
subnets did NOT take effect from the env alone. Fabric-level dual-rail is PROVEN (+46
### EXP 14 (07-27 ~20:05): dual-rail in vLLM needs more than NCCL_IB_HCA list
Launched with NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 but worker env ended up rocep1s0f1 only, and
NCCL log shows "Using [0]rocep1s0f1" (single rail). Multi-rail NCCL over two RoCE subnets did
NOT take effect from the env alone. Fabric-level dual-rail is PROVEN (+46%, EXP13) but the
vLLM/NCCL integration needs: reliable propagation of BOTH HCAs to workers, likely NCCL_CROSS_NIC=1,
and the 2nd-link IPs (10.0.1.x) made persistent. Deferred to do deliberately with user.

## NET RESULT OF RESEARCH SESSION
- dmabuf GDR: does not engage here (env reaches NCCL, GDR stays 0; likely moot on GB10 unified memory)
- MTU 4096: no bandwidth gain (not MTU-bound)
- Dual-rail: +46 percent fabric BW proven; vLLM integration pending (finicky); would help long-ctx TTFT
- FINAL production config: TP=2 eager FlashInfer k=0 single-rail. Stable + protected (watchdog on).

### EXP 15 (07-28 ~03:20): OUTPUT QUALITY ROOT CAUSE + FIX (the live-traffic complaint)
Symptom: rambling answer-less output, literal </think> leaking into content.
Root cause chain:
1. Laguna chat template defaults enable_thinking=true (generation_config default_chat_template_kwargs)
   and force-injects <think> at the end of the prompt -> model reasons on EVERY request.
2. vLLM poolside_v1 parser inherits DeepSeekV3 parser whose __init__ defaults thinking=False when the
   request does not pass chat_template_kwargs -> silently degrades to IdentityReasoningParser -> NO
   separation: reasoning dumped into content, </think> leaks, reasoning field empty. Same mismatch made
   startup log "Auto-initialization of reasoning token IDs failed".
FIX (local patch, backup .bak-20260728): poolside_v1_reasoning_parser.py __init__ now defaults
thinking/enable_thinking=True when unspecified (mirrors DeepSeekV3ReasoningWithThinkingParser),
aligning parser default with template default. Explicit enable_thinking=false still respected.
ALSO: baked min_p 0.05 into --override-generation-config (user-sourced tip; kills low-prob tail at temp 0.7).
VERIFIED: startup warning gone; hard prompt -> reasoning field 4739 chars separated, content clean
correct answer, no leak; enable_thinking=false -> direct answer ("READY").
NOTE: this vLLM returns reasoning under message.reasoning (reasoning_content is DEPRECATED/renamed).
Clients must read .content for answers; budget max_tokens 3000+ for hard tasks (reasoning eats budget).
ALSO FOUND: earlier "crash" was TWO vllm serve processes fighting port 8000 (watchdog relaunched while
a manual test instance was still up). Lesson: always pause watchdog cron BEFORE manual restarts.

### EXP 16 (07-28 ~04:20): DFlash PROPERLY CONFIGURED per official recipe = stable-ish but 2x SLOWER
Official recipe (recipes.vllm.ai/poolside/Laguna-S-2.1) reveals we ran DFlash WRONG before:
DeepGEMM MoE backend is documented INCOMPATIBLE with the DFlash draft path (we had DeepGEMM!),
max-num-seqs must be small (crashes at 256), min_p rejected with speculation.
Retest with --moe-backend triton, k=7, max-num-seqs 8, no min_p:
- Config verified (TRITON Fp8 MoE, spec active). 3/3 sustained decodes OK, no deadlock.
- 150k cold attempt: no EngineDead during prefill window (client timed out; inconclusive on
  long-ctx decode, but no crash - DeepGEMM may well have been the original 112k crash trigger).
- BUT decode = 12.2 tok/s vs 22-23 at k=0 (~2x SLOWER). Acceptance still poor: mean accepted
  1.83-2.42 (doc claims ~3.1), rate 12-20%. On cross-node TP2 every draft pass pays full
  collective latency; at ~2 accepted/step the math loses badly.
VERDICT: DFlash on cross-node TP2 = net loss regardless of config. k=0 rolled back.
KEPT from window: KV cache 24GB (1,371,148 tok capacity, 2x), top_k 20 server default
(min_p removed - clients may send min_p per-request while spec is off), MAX_NUM_SEQS env toggle.

### CORRECTION (07-28, from Joey): original spec-decode config was W4A16 draft_model, NOT dflash
The FIRST config that broke at ~100k context was:
  --speculative-config '{"model":"Laguna-S-2.1-DFlash-W4A16","method":"draft_model","num_speculative_tokens":6}'
  (with --max-num-batched-tokens 8192, KV 5GB, max-num-seqs 8, enforce-eager already on)
method=draft_model runs the draft as a FULL separate TP=2 model -> every draft token pays its own
cross-node collectives; separate spec+TP deadlock surface. The DeepGEMM/DFlash incompatibility note
applies only to method=dflash, so DeepGEMM attribution for the ORIGINAL breakage is WITHDRAWN.
SPEC DECODE SCORECARD on this cluster (3 independent failures/losses):
  1) W4A16 draft_model k=6 -> broke at ~100k (original, Joey-reported)
  2) FP8 dflash k=7 + DeepGEMM -> sample_tokens deadlock ~112k (reproduced 07-25)
  3) FP8 dflash k=7 + Triton MoE (recipe-correct) -> stable but 12 tok/s vs 23 (2x slower, EXP16)
=> The wall is topology economics (no-GDR cross-node collective latency per draft pass at ~2
accepted/step), not any single config. k=0 verdict is robust. Do not retry spec on TP2.
