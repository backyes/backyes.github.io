# Disambiguation: NVIDIA LPX

"NVIDIA LPX" = **NVIDIA Groq 3 LPX** — a rack-scale low-latency inference accelerator
that is part of the NVIDIA Vera Rubin platform. Announced at GTC 2026 (March 2026),
following NVIDIA's acquisition of Groq.

Key early facts (from Google SERP, 2026-07-07):
- Each LPU accelerator delivers **500 MB of SRAM**, **150 TB/s SRAM bandwidth**, and **2.5 TB/s scale-up bandwidth**.
- Source: NVIDIA developer blog "Inside NVIDIA Groq 3 LPX: The Low-Latency Inference Accelerator for the NVIDIA Vera Rubin Platform" (March 2026)
  https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/
- Product page: https://www.nvidia.com/en-us/data-center/lpx/
- The current LPX architecture depends on an x86 CPU's mature PCIe root complex, while Vera (Rubin GPU) is optimized for NVLink C2C. (SERP snippet, 2026-04-14)
- "The final form of the LPX remains uncertain, but it is certain that this chip will be strongly aligned with the new inference paradigm."

Implication for analysis:
This is NOT a GPU-vs-TPU comparison. It is a **deterministic LPU/SRAM inference accelerator
(Groq LPX)** vs a **systolic-MXU TPU (Trillium)** comparison — both non-traditional-GPU
accelerator architectures. This makes the microarchitecture comparison especially rich:
- Trillium = MXU systolic array + HBM + ICI, optimized for training & serving
- LPX = deterministic dataflow LPU + large SRAM + scale-up fabric, optimized for ultra-low-latency inference

Primary sources to fetch:
1. NVIDIA developer blog on Groq 3 LPX
2. NVIDIA LPX product page
3. GTC 2026 Rubin/LPX technical talks / secondary analysis (SemiAnalysis, The Next Platform, etc.)
4. Google Cloud Trillium docs + Google blog posts on Trillium architecture
