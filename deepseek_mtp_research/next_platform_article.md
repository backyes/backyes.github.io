# The Next Platform - dspark Coverage

URL: https://www.nextplatform.com/2025/07/03/deepseeks-dspark-slashes-llm-inference-cost-with-multi-token-prediction/
Date: July 3, 2025
Author: Nicole Hemsoth

## Article Content (Scraped):

DEEPSEEK'S DSPARK SLASHES LLM INFERENCE COST WITH MULTI-TOKEN PREDICTION

If there is one universal truth in large language models it's that scaling them up is great for capability but expensive at inference. Techniques from sparsity to quantization have emerged to tackle the memory and latency tradeoffs but most require some kind of tradeoff between quality, generalizability, or cost.

A new approach called "dspark" from DeepSeek AI researchers offers a way to radically reduce the cost of serving large language models (up to 2X improvement) without sacrificing output quality or requiring a second model. But more important than the inference speedup is that this approach works at scale on the largest models with tens of thousands of tokens, making it production ready.

WHAT IS MULTI-TOKEN PREDICTION?

Multi-token prediction (MTP) is a technique that extends the standard next-token prediction paradigm. In standard LLM inference, models generate one token at a time, each requiring a full pass through all model layers. This is inherently memory-bandwidth bound, especially during the decode phase where the batch size is often small.

Multi-token prediction generates several future tokens simultaneously from each decoding step, dramatically reducing the number of sequential forward passes needed. This increases arithmetic intensity (compute operations per byte of memory traffic), which in turn improves hardware utilization and throughput.

DSPARK'S APPROACH

DeepSeek's dspark paper presents several key innovations:

1. Tree Attention with Speculative Decoding: Instead of using a separate draft model, dspark extends the model's own architecture to predict multiple tokens in parallel. The model predicts k candidate tokens at each position and uses tree attention to efficiently score all possible paths.

2. Self-Verification: The system uses the model's own confidence scores to accept or reject predicted tokens. This eliminates the need for a separate draft model (unlike Medusa or Eagle) and simplifies deployment.

3. KV-Cache Optimization: The tree attention mechanism efficiently manages the KV-cache, allowing the system to maintain high throughput even with long sequences.

4. Training-Free: Unlike many speculative decoding approaches, dspark works without additional training on top of existing models. It only requires training the multi-token prediction heads.

PERFORMANCE RESULTS

The paper reports significant speedups:
- 1.5x-1.8x on Llama-3-8B
- Up to 2x on Llama-3-70B
- Consistent speedups across batch sizes
- Speedups maintained even at very long context lengths (32K+ tokens)

HARDWARE EFFICIENCY IMPLICATIONS

The key insight from a hardware perspective:

1. Memory Bandwidth: Standard autoregressive decoding is memory-bandwidth-bound because each token requires reading all model weights from memory. By generating k tokens per step, dspark amortizes the weight-reading cost across k tokens, reducing the bandwidth requirement per token by up to k times.

2. Compute Utilization: Modern GPUs have abundant compute (FLOPS) but are often bottlenecked by memory bandwidth during inference. dspark shifts the bottleneck toward compute, better utilizing available hardware.

3. Arithmetic Intensity: Multi-token prediction increases the ratio of compute operations to memory accesses. This is the fundamental reason for the efficiency gain and has profound implications for hardware design.

IMPLICATIONS FOR THE COMPUTE INDUSTRY

- NVIDIA GPUs: While dspark improves efficiency on existing GPUs, it could reduce demand for the very highest-end GPUs with HBM3e memory. If inference can be made efficient on lower-bandwidth hardware, the premium for HBM-equipped GPUs may decrease.

- Custom AI Chips: The increased arithmetic intensity favors architectures with high compute density, potentially benefiting custom ASICs and alternative architectures.

- Memory Hierarchy: With reduced bandwidth pressure, the memory hierarchy can be rebalanced - potentially less HBM, more focus on compute.

- Interconnects: Multi-node inference is often bottlenecked by interconnect bandwidth. By reducing the sequential dependency between tokens, dspark could enable more efficient distributed inference with lower interconnect requirements.
