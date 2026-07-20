
# Ascend 90-Day Activity Pulse (2026-04 - 2026-07)

Source: `vllm-ascend` git log `--since 2026-04-01` classified by topic keyword.
Total commits in window: ~1060 across all areas.

## Commits by topic (90-day)
| Topic | Commits | Significance |
|---|---|---|
| kv | 97 | KV-cache management dominates: PD disagg pool (Memcache/Yuanrong/Mooncake layerwise), KV CPU offload, layerwise KV pooling |
| A3 | 64 | 800T A3 hardware bring-up, fabric memory, graph full-mode |
| quant | 61 | W4A4/W4A8/W8A8/PDmix/MXFP4/FlatQuant — heavy quantization investment |
| spec | 53 | speculative decoding (Eagle3, DFlash, DSpark, MTP, ngram, PCP proposer metadata H2D avoid) |
| 310 | 53 | Ascend 310P 适配 (dual-chip 310P is a *separate* hardware line from A2/A3!) |
| PCP | 35 | Prefill Context Parallel — long context splitting |
| eplb | 24 | Expert-Parallel Load Balancing — Ascend original MoE contribution |
| MLA | 15 | Multi-head Latent Attention (DeepSeek V3/V4 style compression) |
| recompute | 14 | recompute scheduler — roofline-optimal swap-vs-recompute |
| Flashcomm | 11 | CANN fused communication (AllGather+Matmul+ReduceScatter fusion) |
| 910 | 9 | Ascend 910 / 950 系列 |
| disagg | 8 | Prefill-Decode disaggregation (Mooncake pull + layerwise push) |
| memcache | 2 | Memcache KV Pool backend |

## Feature Cross-Compatibility Matrix (Ascend, key constraints)
Full matrix at `docs/source/user_guide/support_matrix/feature_matrix.md`.
Key exclusions (✅=compat, 🟠=partial, ❌=incompat):
- ACLGraph Piecewise ⟷ Full: **mutually exclusive** (like CUDA graph)
- DP + CP: only DCP supports DP; PCP does not
- Flashcomm: enabled on prefill stage only
- W4A4 quant ⟷ EPLB/TP: ❌
- MLAPO: decode-stage only
- multimodal inputs + ACLGraph Full: ❌ (only piecewise/eager)
- DCP 嵌入 ASM (Shared Expert DP) 仅 🟠 on CP

## Supported Hardware
- 800I/T A2 系列 (910B)
- 800I/T A3 系列 (910C)
- Ascend 950 (新高端)

## Core Supported Models
| Model | W8A8 | TP | PP | EP | DP | PD Disagg | Graph |
|---|---|---|---|---|---|---|---|
| DeepSeek V4-Flash/Pro | ✅ | ✅ | - | ✅ | ✅ | ✅ | Full |
| DeepSeek V3/3.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Full+Piecewise |
| Qwen3-235B-A3B | ✅ | ✅ | - | ✅ | ✅ | ✅ | Full+Piecewise |
| Qwen3-Dense | ✅ | ✅ | - | - | ✅ | - | Full+Piecewise |
| GLM-5/5.2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Full+Piecewise |
| Gemma4 | - | ✅ | - | - | ✅ | - | Full+Piecewise |

## Recent Headline PRs (2026-06+)
- `#12112` Remove get_masked_input_and_mask ops (cleanup)
- `#11444` Support Layerwise KV Pooling with Memcache Backend
- `#12079` Snapshot query start locations before async H2D copies (w/ the bugfix)
- `#12036` Fix DCP + DP services hang
- `#12020` Adapt to vLLM main (85c09e98) — keeping up with vLLM mainline
- `#11672` Initialize AscendStore after first real decode request (KV Pool startup fix)
- `#11496` [Performance][SpecDecode] Avoid H2D sync in CP proposer metadata
- `#11914` Update torch_npu to 2.10.0.post2
- `#11921` Fix AllGather MoE finalize order with DP and PCP
