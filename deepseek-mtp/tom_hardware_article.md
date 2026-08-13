# Tom's Hardware - dspark Coverage

URL: https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-dspark-mtp-multi-token-prediction
Date: July 2, 2025
Author: Anton Shilov

## Article Content:

DeepSeek's 'dspark' paper details multi-token prediction for inference — up to 2X higher throughput

The dspark paper describes a method that uses the model's existing capabilities to generate multiple tokens per step instead of one, without a draft model.

DeepSeek has published a paper detailing a novel method to improve the inference performance of large language models (LLMs). Called dspark, the technique builds upon the concept of multi-token prediction (MTP) which was originally proposed by Meta. Unlike other speculative decoding methods that require a separate draft model, DeepSeek's approach uses the main model's own intermediate representations to predict multiple future tokens simultaneously.

The key insight behind dspark is that during autoregressive decoding, LLMs spend most of their time waiting for weights to be loaded from memory rather than actually computing. This is the well-known 'memory wall' problem in AI inference. By predicting multiple tokens per forward pass, dspark amortizes the cost of loading model weights across multiple generated tokens, effectively increasing the arithmetic intensity of the decoding process.

Key technical details:
- Uses tree-structured attention to handle branching token predictions
- Self-verification mechanism ensures output quality
- Works on top of existing pre-trained models (only MTP heads need training)
- Tested on Llama-3-8B and Llama-3-70B
- Reports 1.5x-2x speedup across various configurations
- Maintains output quality equivalent to standard autoregressive decoding

The paper represents another significant contribution from DeepSeek, following their earlier breakthroughs with DeepSeek-V2 (MoE architecture) and DeepSeek-V3 (which challenged assumptions about training compute requirements).

From a hardware perspective, the most significant implication is that dspark reduces the memory bandwidth bottleneck that has made high-bandwidth memory (HBM) a critical component for LLM inference. If inference can achieve high throughput with lower memory bandwidth, it could change the competitive dynamics between different GPU tiers and between GPUs and custom ASICs.

The paper was published on arxiv on July 2, 2025, and has already generated significant discussion in the AI research community about its implications for inference hardware requirements and the future of LLM serving infrastructure.
