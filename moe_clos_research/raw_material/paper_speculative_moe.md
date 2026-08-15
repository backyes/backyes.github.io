# Speculative MoE: Communication Efficient Parallel MoE Inference

> arXiv 2503.04398v1, 2025
> Authors: Yan Li, Pengfei Zheng, Shuang Chen, et al.
> URL: https://arxiv.org/abs/2503.04398v1

## Core Thesis
EP (all-to-all) is the primary bottleneck in MoE inference. Speculative pre-scheduling can reduce EP's communication volume.

## Key Techniques
1. **Speculative token shuffling**: Predict token's expert routing paths and pre-schedule tokens
2. **Speculative expert grouping**: Pre-schedule experts across devices to minimize cross-device communication

## Results
- Significantly boosts DeepSpeed-MoE and SGLang inference
- Works on both fast (NVLink) and slow (Ethernet) interconnects

## Significance
Demonstrates that predicting routing patterns can _losslessly_ reduce A2A communication volume, effectively making the network "sparser" without modifying topology.
