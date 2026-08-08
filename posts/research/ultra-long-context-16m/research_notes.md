# Research Notes: Every Token Counts - Generalizing 16M Ultra-Long Context
## Date: 2026-08-08

## Paper Metadata
- **Title**: Every Token Counts: Generalizing 16M Ultra-Long Context in Large Language Models
- **arXiv ID**: 2511.23319
- **Submitted**: November 28, 2025
- **Authors**: Xiang Hu*, Zhanchao Zhou*, Ruiqi Liang, Zehuan Li, Wei Wu, Jianguo Li†
- **Affiliations**: Ant Group (1), Westlake University (2)
- **Categories**: cs.CL, cs.AI
- **Pages**: 14

## URLs Visited
- https://arxiv.org/abs/2511.23319 (abstract page)
- https://arxiv.org/pdf/2511.23319 (PDF, downloaded)
- https://arxiv.org/html/2511.23319v1 (HTML version for text extraction)
- https://arxiv.org/abs/2510.17196 (related HSA paper: "Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models")

## Prior Work by Same Authors (cited in paper)
1. "Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models" (arXiv:2510.17196, 2025) - establishes core HSA principles: chunk encoder with CLS token, bypassing residual path, enforced sparsity during pretrain
2. "Hardware-Aligned Hierarchical Sparse Attention for Efficient Long-Term Memory Access" (NeurIPS 2025) - hardware-efficient implementation
3. "Efficient Length-Generalizable Attention via Causal Retrieval" (ICML 2025)
4. Ling-2.0 (MoE architecture reference)
5. Grove MoE (SFT data reference)

## Key Technical Contributions

### 1. Hierarchical Sparse Attention (HSA) - Core Innovation
- **Analogy**: HSA is to attention what MoE is to FFN
- **Mechanism**: 
  - Split sequence into fixed-length chunks (size=64 by default)
  - Each chunk has a "landmark" representation (summary)
  - Current token computes dot-product with landmarks → retrieval scores
  - Select top-k chunks based on scores
  - Perform attention with EACH chunk separately
  - Fuse results weighted by softmax-normalized retrieval scores
- **Key difference from NSA**: HSA chunk selection IS end-to-end learnable (NSA's is not)

### 2. Three Key Properties for Ultra-Long Context
1. **Sparsity**: Selective activation like human memory
2. **Random-Access Flexibility**: Intrinsic retrieval mechanism optimized end-to-end
3. **Length Generalization**: Generalize retrieval ability from short to long contexts

### 3. Model Architecture (HSA-UltraLong)
- **Type**: 8B-parameter MoE model (1B activated params)
- **Architecture**: 
  - L layers split into upper and lower decoder
  - Lower decoder: L/2 standard Transformer layers with Sliding Window Attention (SWA)
  - Upper decoder: G groups, each with one SWA+HSA layer followed by SWA-only layers
  - HSA layers share KV cache from intermediate layer (L/2 output)
  - Bi-directional encoder + [CLS] token for chunk summary representation
- **MoE config**: Follows Ling-2.0 design, 64 experts / 4 activated (modified from 32/2), shared expert like DeepSeek V3
- **NoPE**: No Positional Encoding for HSA (critical for length extrapolation)

### 4. Training Methodology (5 stages)
1. **Warm-up**: SWA=512, full HSA (top-k covers full sequence), 16K context, synthetic RULER tasks (1%)
2. **Pre-training**: SWA=4K, sparse HSA, 16K context, 8T tokens (MoE), 4T tokens (dense)
3. **Long-context mid-training**: Long effective context data (>32K), top-k raised to full, 32K context, 175B tokens
4. **Annealing**: High-quality reasoning data, 32K context, 400B tokens
5. **SFT**: Supervised fine-tuning, 8K context

### 5. Key Experimental Results
- **RULER benchmark**: Near-perfect accuracy on Single-NIAH at 16M context
- **MQ-NIAH (Multi-Query)**: >90% accuracy up to 16M
- **Variable Tracking**: Strong performance showing reasoning+retrieval capability
- **Standard benchmarks**: Comparable to Qwen2.5-0.5B (despite 4.5x less data) and Qwen3-1.7B (despite 4.5x fewer training tokens)
- **PG19 perplexity**: 15.96 at 16K (comparable to full-attention baseline)

### 6. Hardware & Efficiency
- **Hardware**: NVIDIA H800 GPUs
- **Implementation**: HSA kernel in TileLang (not CUDA)
- **Training**: FSDP2 distributed training
- **Efficiency finding**: HSA only beats FlashAttention-3 at longer sequences; at short seqs, FA3 wins (better CUDA optimization, less memory access overhead from sparsity)
- **Head ratio constraint**: 16:1 query:KV heads ratio (information bottleneck)

### 7. Critical Findings
1. **Effective context length of training data is critical**: Models pretrained on standard corpora (short effective context) show progressive decline; training on long effective context data (>32K) yields much better extrapolation
2. **Seesaw effect between HSA and SWA**: Smaller SWA window (512) → better HSA extrapolation; larger SWA (4K) → HSA doesn't learn short-range dependencies → worse generalization
3. **HSA capability scales with parameter size**: MoE-8B > Dense-0.5B on reasoning-retrieval tasks
4. **NoPE is essential**: RoPE hurts length extrapolation
5. **Warm-up is necessary**: Training from scratch with 4K SWA fails to generalize

## Specific Answers to User Questions

### Prefill Stage Handling
- The paper does NOT explicitly discuss prefill vs. decode stage separately for HSA
- HSA operates the same way during prefill: each token retrieves top-k chunks from all past chunks
- The sparsity IS maintained during prefill — each token only attends to top-k retrieved chunks (k=64 chunks × 64 tokens/chunk = 4096 tokens) plus sliding window
- For 16M context: each token performs full attention within SWA window (4K) + attention with 64 retrieved chunks (64×64=4096 tokens) = ~8K effective attention per token (vs 16M for full attention)
- **Key insight**: The retrieval is content-based (landmark dot-products), so sparsity is inherent at both prefill and generation time
- The shared KV cache design means chunk representations are computed once and reused across all HSA layers

### Business Context / Motivation
- "Machines that Can Remember" - long-term memory for AI agents
- Personalized agents that accumulate unique experiences over time
- World knowledge retrieval from context vs. compression into parameters
- Skills/information acquisition via in-context learning vs. costly retraining

## Limitations Acknowledged
1. HSA/SWA seesaw problem: SFT on short data degrades extrapolation
2. Head ratio constraint (16:1) creates information bottleneck
3. No clear efficiency advantage over FlashAttention-3 at short sequences
4. Need kernel-level optimizations (currently TileLang, not CUDA)

## Key Insights for AI Infra Researchers
1. **Length generalization ≠ just sparsity**: Must combine chunk-wise attention + retrieval-score fusion + NoPE
2. **Data effective context length matters more than window size**: Training on data with longer dependencies enables better extrapolation
3. **MoE + Sparse Attention is a viable scaling path**: 8B/1B model achieves 16M context with competitive standard benchmarks
4. **Retrieval as attention, not pre-processing**: HSA's key insight is that chunk selection gradients flow through the attention computation
5. **Warm-up curriculum is critical**: Progressive sparsity (full→sparse) enables model to learn meaningful retrieval patterns
