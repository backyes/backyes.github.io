# SOURCE: SemiAnalysis (via moomoo mirror) — "GTC In-Depth Analysis: Behind Three New Systems, NVIDIA Is Redefining the Boundaries of AI Infrastructure"
URL: https://www.moomoo.com/news/post/67335112/semianalysis-gtc-in-depth-analysis-behind-three-new-systems-nvidia
Original: newsletter.semianalysis.com/p/nvidia-the-inference-kingdom-expands (SemiAnalysis: "NVIDIA: The Inference Kingdom Expands")
Retrieved: 2026-07-07
Type: SECONDARY (expert analyst, SemiAnalysis), GTC 2026, Mar 24 2026

## STRATEGIC SIGNAL (SemiAnalysis thesis)
- GTC 2026 three new systems: **LPX inference rack** (Groq LP30), **Vera ETL256** (256-CPU liquid-cooled rack), **STX** (storage reference architecture).
- "NVIDIA is no longer just a GPU vendor; it is evolving into a full-stack AI infrastructure platform provider" — reach into inference optimization, CPU density, storage orchestration. "widened NVIDIA's product moat... larger share of AI infra supply chain concentrates on NVIDIA."

## GROQ DEAL STRUCTURE (critical / investment lens)
- Structured as **IP licensing + talent acquisition (acquihire), NOT traditional M&A.** ~**$20B**.
- Non-exclusive licensing agreement (announced ~Dec 2024/2025).
- NVIDIA got ALL Groq IP + core team; launched LP30 + LPX in **<4 months** after deal close.
- Cross-ref (SERP): "Nvidia paid Groq $20B to license their IP and hire most the team. This functions almost as an acquisition." / "defensive consolidation disguised as platform expansion... paying nearly 3x valuation." / "By acquiring Groq's assets, Nvidia can now sell more AI chips than TSMC has CoWoS production capacity to make for them."

## LP30 CHIP (Groq 3rd-gen LPU) — SemiAnalysis
- **Samsung SF4 process.** Single monolithic die — NO advanced packaging.
- 500 MB on-chip SRAM, **1.2 PFLOPS FP8** compute per chip.
- vs Groq 1st-gen LPU: 230 MB SRAM, 750 TFLOPS INT8. Boost from GF16 → SF4 node.
- **SF4 does NOT consume TSMC N3 capacity NOR HBM** → "genuine incremental capacity and revenue... differentiating advantage competitors cannot replicate."
- (Confirms earlier derivation: 9.6 PFLOPS/tray ÷ 8 = 1.2 PFLOPS/chip; 256 × 1.2 ≈ 307 PFLOPS, NVIDIA rounds rack to 315 PFLOPS.)

## LPU CORE VALUE + INHERENT LIMITATION (critical depth)
- Advantage: high-BW SRAM + deterministic pipelined execution → TTFT/first-token speeds GPUs can't match in single-user, low-latency.
- Trade-off: high-density SRAM = limited capacity. After weights loaded, little space; as batch grows, **KV cache saturates fast → overall throughput << GPUs.**
- "A standalone LPU system is NOT cost-effective for large-scale token services, but commands premium in latency-sensitive apps — basis for LPU's positioning within decoupled decoding (AFD)."

## AFD (Attention-FFN Disaggregation) — mechanics
- Attention (dynamic KV cache loading) → **GPU**. FFN (stateless, statically schedulable) → **LPU**.
- GPU dedicated to attention → frees HBM for KV cache → more concurrent tokens.
- LPU handles FFN → leverages low-latency.
- GPU↔LPU via **all-to-all collectives**; **ping-pong pipelining** hides comm latency.
- Speculative decoding: draft model / MTP (multi-token prediction) layers on LPU → **1.5–2x output tokens/step.**

## LPX RACK ARCHITECTURE — SemiAnalysis (richer than vendor blog)
- 32 × 1U LPU compute trays + **two Spectrum-X switches.**
- Per compute tray (SemiAnalysis): **16 LP30 chips**, 2 Altera FPGAs ("Fabric Expansion Logic"), 1 Intel Granite Rapids host CPU, 1 BlueField-4 front-end module.
  - **DISCREPANCY:** NVIDIA official blog table says **8 LP30/tray** (per-tray SRAM 4GB = 8×500MB, compute 9.6 PFLOPS = 8×1.2). SemiAnalysis says 16/tray with "back-to-back 8 top + 8 bottom" mounting. Official per-tray math is only consistent with 8/tray (256 total). Likely SemiAnalysis's "16" = a 2-tray NODE (8+8 back-to-back), OR a higher-density variant. Report flags this.
- **Fabric Expansion Logic (FPGA):** (1) converts LPU C2C → Ethernet for Spectrum-X scale-out; (2) PCIe bridge LPU↔host CPU; (3) up to **256 GB DDR5 expansion memory per board for KV cache.**
- Rack scale-out BW ~**640 TB/s.**
- LPU modules back-to-back (8 top + 8 bottom) to minimize X/Y trace length for full mesh. Intra-node 16-LPU full mesh; inter-node via **copper backplane**; cross-rack via **front-panel OSFP.**

## VERA ETL256 (CPU rack) — context, the 2nd new system
- 256 Vera CPUs/rack, liquid-cooled. Addresses CPU bottleneck as AI scales (esp. **RL**: simulation + code exec + validation must run concurrently; GPU scaling outstrips CPU).
- 32 compute trays (16 top / 16 bottom), 4 × 1U MGX ETL switch trays (Spectrum-6). Symmetric layout → equalize cable lengths → ALL intra-rack **copper** (no optical transceivers; copper savings offset liquid-cooling cost).
- Switch rear ports = in-rack copper trunk; 32 front OSFP = fiber to POD. Spectrum-X multi-plane, 200 Gb/s lanes × 4 switches, full-mesh Ethernet among 256 CPUs in one network layer, 8 Vera CPUs/compute tray.
- Design logic mirrors NVL compute rack: push density to copper-cabling threshold.

## STX (storage rack) — the 3rd new system
- Storage reference rack architecture (with CMX context storage platform).
- Each STX chassis: 2 BF-4, 2 Vera CPUs, 4 CX-9 NICs, 4 SOCAMM. Rack: 16 chassis = 32 Vera CPUs, 64 CX-9, 64 SOCAMM.
- Storage vendor partners: DDN, Dell, HPE, IBM, NetApp, Supermicro, VAST Data.
- SemiAnalysis: BF-4 + CMX + STX = NVIDIA systematically advancing into storage/software/infra-ops layers after dominating compute (GPU) + networking (Spectrum-X, NVLink).

## NOTE — "Vera Rubin POD: Seven Chips, Five Rack-Scale Systems"
- The Vera Rubin platform = 7 chips (Rubin GPU, Vera CPU, Groq LP30 LPU, BlueField-4 DPU, Spectrum-X/6 switch, CX-9 NIC, SOCAMM memory) across 5 rack-scale systems (NVL72 GPU, LPX LPU, ETL256 CPU, STX storage, + CMX/another). The LPU is "the seventh chip."
