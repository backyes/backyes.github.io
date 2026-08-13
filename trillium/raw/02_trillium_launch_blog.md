# SOURCE: Google Cloud Blog — Introducing Trillium, sixth-generation TPUs
URL: https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus
Retrieved: 2026-07-07
Type: PRIMARY (vendor launch blog), Google I/O May 2024

## HEADLINE SPECS (vs TPU v5e)
- **4.7X increase in peak compute performance per chip** vs TPU v5e.
- **2X HBM capacity AND 2X HBM bandwidth** vs v5e (next-gen HBM, flexible channel architecture).
- **2X ICI (Interchip Interconnect) bandwidth** vs v5e.
- **>67% more energy-efficient** than TPU v5e.
- **Third-generation SparseCore** — specialized accelerator for ultra-large embeddings in advanced ranking & recommendation workloads. Offloads random/fine-grained access from TensorCores.
- Expanded **MXU size** + increased **clock speed** to achieve 4.7X.

## SCALE / CLUSTER
- Up to **256 TPUs per pod** (high-bandwidth, low-latency).
- **Multislice technology + Titanium IPUs** → scale to hundreds of pods, tens of thousands of chips, building-scale supercomputer.
- **Multi-petabit-per-second datacenter network.**
- Custom **optical ICI interconnects** (256 chips/pod) + **Google Jupiter Networking** (extends to hundreds of pods).
- Part of Google Cloud **AI Hypercomputer**: performance-optimized infra + open-source frameworks (JAX, PyTorch/XLA, Keras 3) + flexible consumption models.
- **Dynamic Workload Scheduler (DWS)** with flex start mode for bursty training/fine-tuning/batch jobs.

## SOFTWARE STACK
- JAX + XLA (declarative model description maps directly from any prior TPU gen).
- PyTorch/XLA, Keras 3.
- Hugging Face Optimum-TPU.

## CONTEXT (quoted, original language English)
- "Generative AI is transforming how we interact with technology while simultaneously opening tremendous efficiency opportunities... these advances require ever greater compute, memory, and communication to train and fine tune the most capable models and to serve them interactively to a global user population."
- Trained/serves: Gemini 1.5 Flash, Imagen 3, Gemma 2.
- "the scale and efficiency of TPUs enabled foundational work on Transformers in Google Research, the algorithmic underpinnings of modern generative AI."

## NOTE — precision specs to verify from docs (SERP snippets suggest):
- HBM capacity: 32 GB per chip
- ICI bandwidth: 3,200 Gbps per chip
- 1536 GiB DRAM per host (3x v5e)
- 2D Torus topology, 256 chips/pod
- MXU count per TensorCore: DISCREPANCY (2 vs 4) — verify in docs
