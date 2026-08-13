# SOURCE: NVIDIA Developer Technical Blog — Inside NVIDIA Groq 3 LPX
URL: https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/
Retrieved: 2026-07-07
Type: PRIMARY (vendor technical blog), GTC 2026, March 2026

## RACK-SCALE SPECS (NVIDIA Groq 3 LPX)
| Specification | NVIDIA Groq 3 LPX |
|---|---|
| AI inference compute | 315 PFLOPS |
| Total SRAM capacity | 128 GB |
| On-chip SRAM bandwidth | 40 PB/s |
| Scale-up density | 256 chips |
| Scale-up bandwidth | 640 TB/s |

## PER TRAY (32 liquid-cooled 1U compute trays per rack)
| Resource | Per LPX Tray |
|---|---|
| LP30 chips | 8 |
| On-chip SRAM | 4 GB |
| SRAM bandwidth | 1.2 PB/s |
| DRAM via fabric expansion logic | Up to 256 GB |
| DRAM via host CPU | Up to 128 GB |
| AI inference compute (FP8) | 9.6 PFLOPS |
| Scale-up bandwidth | 20 TB/s |

Note: 32 trays × 8 LP30 = 256 chips. Cableless design. MGX ETL rack architecture.

## PER LPU (LP30 chip) — derived/confirmed
- 500 MB on-chip SRAM (×256 = 128 GB total ✓)
- 150 TB/s on-chip SRAM bandwidth (×256 = ~38.4 PB/s ≈ 40 PB/s ✓)
- 2.5 TB/s scale-up bi-directional bandwidth per LPU (×256 = 640 TB/s ✓)
- 96 C2C links @ 112 Gbps each
- FP8 compute per chip: 315 PFLOPS / 256 ≈ 1.23 PFLOPS per LPU (or 9.6 PFLOPS / 8 = 1.2 PFLOPS per tray chip ✓)

## MICROARCHITECTURE — NVIDIA Groq 3 LPU ("the seventh chip of the Vera Rubin Platform")
- Based on Groq's TSP (Tensor Streaming Processor) architecture. References research papers:
  - "Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads"
  - "A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning"
- **Unit of work = 320-byte vectors.** Arithmetic, memory access, and inter-device transfers all operate on fixed-size 320-byte vectors. Simplifies scheduling/synchronization.
- **Tensor-first compute + explicit data movement.** Specialized execution modules:
  - MXM (Matrix eXecution Module): dense MAC for tensor ops, fixed data types, predictable throughput.
  - VXM (Vector eXecution Module): pointwise arithmetic, type conversions, activations — mesh of ALUs per lane.
  - SXM (Switch eXecution Module): structured data movement — permutation, rotation, distribution, transposition of vectors.
  - MEM block: flat, SRAM-first memory architecture. 500 MB SRAM is PRIMARY working storage (weights, activations, KV state). NO hardware-managed cache — compiler/runtime places data explicitly. Reduces unpredictable stalls → low stable latency.
- **C2C scaling (deterministic):** 96 C2C links @ 112 Gbps each → 2.5 TB/s aggregate I/O bi-directional. High-radix, high-speed, deterministic data exchange. Plesiosynchronous chip-to-chip protocol cancels clock drift, aligns hundreds of LPUs as single coordinated system.
- **Deterministic, compiler-orchestrated execution:** Groq's spatial execution model. Compiler explicitly schedules compute, data movement, synchronization. No dynamic hardware scheduler at runtime. Enables: precise memory-compute coordination, explicit instruction timing control, reduced jitter under variable loads. Stable TTFT and per-token latency even at small batch sizes.
- Larger models scaled across many LPUs via layer-wise partitioning. Performance governed less by peak arithmetic, more by how consistently compute is fed.

## HETEROGENEOUS INFERENCE ARCHITECTURE (Vera Rubin NVL72 GPU + LPX LPU)
- Two-engine architecture. "Up to 35x higher inference throughput per megawatt and up to 10x more revenue opportunity for trillion-parameter models."
- **Decode phase = two-engine loop (Attention-FFN Disaggregation, AFD):**
  - GPUs (Vera Rubin NVL72): prefill (build KV cache), decode attention over accumulated KV cache.
  - LPX (LPU): latency-sensitive decode FFN / MoE expert execution + pointwise ops.
  - Intermediate activations exchanged per token between engines.
- NVIDIA Dynamo = orchestration layer: classifies requests, routes by latency target, moves activations low-overhead, KV-aware routing, keeps tail latency stable under bursty traffic.
- **Speculative decoding:** LPX = draft model (fast deterministic draft token generation via high SRAM BW); Rubin GPU = verifier (prefill, attention, token verification).
- 35x higher TPS/MW at 400 TPS/user vs NVIDIA GB200 NVL72.
- Up to 5x more revenue/MW vs GB200 NVL72; up to 10x with NVL72+LPX for latency-sensitive workloads (agentic coding, multi-agent).

## PARADIGM / POSITIONING
- "Speed of thought computing" — approaching 1,000 tokens/sec/user.
- Agentic future: latency compounds across inference→retrieval→tool use→reasoning loops.
- Pareto frontier framing: X-axis = TPS/user (interactivity), Y-axis = TPS/MW (factory throughput). Heterogeneous arch expands achievable region.
- Premium tier example: 2-trillion-parameter MoE model, 400K input context, ~400 TPS/user.
- MGX ETL rack architecture; common infrastructure with Vera Rubin NVL72.
- LPX currently depends on x86 CPU's PCIe root complex; Vera (Rubin GPU) optimized for NVLink C2C. (from SERP snippet)

## KEY RESOURCES referenced
- NVIDIA LPX page; Press release "NVIDIA Vera Rubin Opens Agentic AI Frontier"
- "Inside the NVIDIA Rubin Platform: Six New Chips, One AI Supercomputer"
- "NVIDIA Vera Rubin POD: Seven Chips, Five Rack-Scale Systems, One AI Supercomputer"
- "Announcing NVIDIA Dynamo 1.0: Scaling MultiNode Inference in Production"
- Research: TSP papers (Think Fast; Software-defined TSM)
