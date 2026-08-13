# vLLM Key Architecture PRs (from release notes v0.16-v0.25 & roadmaps)

## PagedAttention / Attention Backend
- #47361 - PagedAttention removed (legacy attention deleted, V1/MRv2 standard) [v0.25]
- #32974 - FlashAttention 4 integration [v0.17]
- #38819/#38835 - FA4 default MLA prefill, head-dim 512 + paged-KV on SM90+ [v0.20]
- #38479/#40092 - TurboQuant 2-bit KV cache backend (4× capacity) [v0.20]
- #41778 - TOKENSPEED_MLA backend on Blackwell [v0.21]
- #41228/#41445/#39571 - KV Offload + HMA integration [v0.21]
- #41286 - Model Runner V2 (roadmap issue)

## Model Runner V2 & Flat Model
- #44443 - MRV2 default ALL dense models [v0.25]
- #44446 - MRV2 quantized models default [v0.24]
- #43458 - MRV2 default Llama+Mistral [v0.23]
- #44050 - Breakable CUDA graphs [v0.23]
- #42187 - PP bubble elimination [v0.23]
- #37588 - Eagle prefill full-CUDA-graph [v0.20]
- #32936 - auto-resolve cudagraph mode/sizes from attn backend [v0.20]
- #33960 - MRV2 Pipeline Parallel [v0.17]
- #34179 - MRV2 Decode Context Parallel [v0.17]
- #32771 - MRV2 piecewise & mixed CUDA graph [v0.17]
- #42770 - [RFC] Flat Model Migration (design doc) [roadmap]

## Disaggregated Prefill / PD分离
- #17751 - NIXL integration core P/D disagg framework [base]
- #43097 - Bidirectional KV Transfer [roadmap]
- #43099 - KV Cache block Lease mechanism [roadmap]
- #33377 - Async scheduling + request abort + async KV transfer bugfix
- #27648 - Async scheduling support
- #18833/#20189/#27274 - Heterogeneous TP (P≠D TP sizes)
- #26759/#30275 - Heterogeneous block_size
- #27743/#30275 - Heterogeneous KV layout (HND↔NHD)
- #33702 - PD Disaggregation Roadmap (NixlConnector)

## KV Cache Manager / Offloading
- #40020 - Multi-tier KV cache offloading framework [v0.22]
- #41968 - Object-store secondary tier [v0.23]
- #41735 - Python filesystem secondary tier [v0.22]
- #43142 - DSv4 offloading support [v0.22]
- #42689 - Mooncake disk offloading [v0.22]
- #41847 - HMA enabled by default for connectors [v0.23]
- #44287 - Tiering support for HMA models [v0.23]
- #43205 - Per-request offloading policy [v0.23]
- #37160/#37874/#34805/#36642 - General CPU KV cache offloading [v0.19]
- #35342 - Smart CPU offloading (frequent blocks) [v0.18]
- #34328 - FlexKV offloading backend [v0.18]
- #33689 - [RFC] KV Offloading Roadmap

## Scheduler
- #32618 - Async scheduling + PP (30.8% throughput) [v0.16]
- #32951 - Zero-bubble async scheduling + spec decode [v0.19]
- #34668 - Spec decode + thinking budget [v0.21]
- #29184 - NGram GPU spec decode + async scheduler [v0.18]
- #44794 - FlowPrefill: adaptive sub-chunk preemption (mentioned in Q3 roadmap)

## Parallelism (TP/EP/DP/PP)
- #41183 - DeepEP v2 for expert parallelism [v0.24]
- #35627 - NIXL-EP integration (Elastic EP M2) [v0.18]
- #34861 - Elastic Expert Parallelism initial [v0.17]
- #37351 - enable-ep-weight-filter for EP loading [v0.18]
- #28782 - Proxy server for high concurrency / DP [base]
- #33960 - MRV2 Pipeline Parallel [v0.17]
- #45810 - MiniMax-M3 pipeline parallelism [v0.25]

## Speculative Decoding
- #32887 - Unified Parallel Drafting [v0.16]
- #38174 - Universal spec decode heterogeneous vocabularies (TLI) [v0.25]
- #46995 - DSpark drafter [v0.25]
- #46770/#46853 - DFlash drafter [v0.25]
- #45953 - Dynamic spec decode + full CUDA graphs [v0.25]
- #33736 - Hidden states extraction [v0.17]
- #35029/#35040 - Eagle3 + CUDA graphs [v0.17]
- #36658/#36361 - Eagle3 for Qwen3.5/Kimi K2.5 [v0.18]

## Quantization
- #41566 - Online quantization refactor (flexible/memory-efficient) [v0.22 Q2 roadmap]
- #40835/#40177 - INT8 dynamic per-token KV-cache quantization [Q2 roadmap]
- #38138/#39736 - Online quantization frontend [v0.20]
- #38463 - experts_int8 → FP8 online path [v0.20]
- #40152 - MXFP8 online quant [v0.20]
- #41652 - W{1-8}A{16/8/4} broader bitwidth + humming-kernel [Q2 roadmap]
- #42997 - Efficient transforms/rotations for low-bit quantization [Q2 roadmap]
- #38032 - Weight reloading for RL [Q2 roadmap]

## Rust Frontend
- #40848 - Rust frontend integration [v0.22]
- #43283 - Rust implementation moved in-tree [v0.22]
- #40841 - DP Supervisor for DP serving [v0.22]
- #44280 - Rust Frontend Feature Parity roadmap
- #45890 - HTTPS/mTLS [v0.25]
- #47076 - DP supervisor [v0.25]

## vLLM IR & Compilation
- #33825 - vLLM IR skeleton (rms_norm op) [v0.20]
- #38807 - OOT-platform kernel imports [v0.20]
- #39014 - gemma_rms_norm on IR [v0.20]
- #40167 - IR op testing/benchmarking [v0.20]

## MoE / Hardware
- #36286 - MoE unquantized → Full Oracle Flow [v0.20]
- #39187 - CT W8A8 to Oracle [v0.20]
- #35153 - SharedExperts class [v0.20]
- #43004 - DeepSeek V4 model package [v0.22]
- #35466 - CPU AVX2/AVX-512/VNNI/AMX multi-ISA dispatcher [v0.17]
