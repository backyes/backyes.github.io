# SOURCE: The Next Platform — "Lots Of Questions On Google's Trillium TPU v6, A Few Answers"
URL: https://www.nextplatform.com/ai/2024/06/10/lots-of-questions-on-googles-trillium-tpu-v6-a-few-answers/1633984
Retrieved: 2026-07-07
Type: SECONDARY (expert analyst), 2024-06-10

## KEY ANALYST INSIGHTS (critical perspective)
- Trillium = TPU v6 (codename after the wood lily plant; Lenoir NC datacenter).
- "probably the first in what will very likely be a pair of devices, including one that has more oomph for inference workloads" — predicts v6e (inference-tuned) + v6p (training) split, mirroring v4i/v5e and v4/v5p pattern. (Confirmed: v6e is the cloud SKU.)
- MXU change: "larger matrix multiply units (MXUs) than the ones used in the prior four generations, which use 128×128 matrices... The TPU v1 chips used a 256×256 matrix in its cores, and we think Google may have reverted to this format and figured out a way to double pump it efficiently. There is an outside chance that Google created a 192×192 matrix, but we doubt it. It violates our sense of symmetry."
  → Reconciles the docs "2 MXU per TensorCore": likely 2 × (larger 256×256) MXU, vs v5e's 1 × (128×128).
- Precision: v6 supports INT8 + BF16; "there is a chance that lower eight-bit and four-bit floating point formats will also be supported" (FP8/FP4 — confirmed Trillium supports FP8).
- v1 did only INT8; v2/v3 BF16; v4/v5 INT8+BF16.
- "As far as we know, the TPU v6 devices do not support 64-bit or 32-bit floating point math."

## POSITIONING (critical)
- "The TPU engines... are not just a way to negotiate better pricing with Nvidia for GPU-style matrix math engines. The TPUs are also a way to drive fundamental research in mixed precision, matrix and serial processing design, memory subsystems, and interconnects."
- Baseline to compare vs Nvidia/AMD GPUs, Intel Gaudi, SambaNova, Cerebras, Tenstorrent.

## TREND (price/perf) — investment-manager lens
- TNP predicts Google raises Trillium rental price ~2X over v5e, still delivering ~2.3X better bang-for-buck than v5e and 3.4X better than v5p.
- Spot-instance cost-per-unit-performance fell 14.5X from TPU v2 (2017) to TPU v4 (2024); on-demand 24.2X; 1yr reserved 21.7X; 3yr reserved 24.2X.
- Quote: "TPU v6 is so much better that Google can charge more per device and still give customers a much better deal."

## NUMBERS (TNP-estimated/calc, June 2024 — pre-docs)
- v5e baseline: ~197 TFLOPs BF16 / 393 TOPs INT8 per chip.
- Trillium: 4.7X → ~918 TFLOPs BF16 / ~1836 TOPs INT8 (matches later official docs exactly). TNP's calc was correct.
