Feature × Feature

The table below shows mutually exclusive features and the support on Ascend hardware, extended from the vLLM table.

The symbols used have the following meanings:

- ✅ = Full compatibility
- 🟠 = Partial compatibility
- ❌ = No compatibility
- ❔ = Unknown or TBD

| Feature | ACLGraph Full_Decode_Only | ACLGraph Piecewise | Async Scheduling | <abbr title="Automatic Prefix Caching">APC</abbr> | Chunked Prefill | Context Parallel | Cpu Binding | <abbr title="Data Parallel">DP</abbr> | Disaggregated Prefill | Eagle3 | Eplb | <abbr title="Expert-Parallel">EP</abbr> | Flashcomm1 | KV Cache Pool | Lmhead TP | Mlapo | <abbr title="Multimodal Inputs">mm</abbr> | Multistream Moe | Shared Expert DP | Quantization W4A4 | Quantization W4A8 | Quantization W8A8 | <abbr title="Tensor Parallel">TP</abbr> | Weight nz |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| ACLGraph Full_Decode_Only | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ACLGraph Piecewise | ❌ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Async Scheduling | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| <abbr title="Automatic Prefix Caching">APC</abbr> | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Chunked Prefill | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Context Parallel | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Cpu Binding | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| <abbr title="Data Parallel">DP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>1</sup> | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Disaggregated Prefill | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Eagle3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Eplb | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |
| <abbr title="Expert-Parallel">EP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |
| Flashcomm1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>2</sup> | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |
| KV Cache Pool | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |
| Lmhead TP | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | 🟠<sup>3</sup> | ✅ | ✅ | ✅ | ✅ | ❌ | ❔ | ✅ |  |  |  |  |  |  |  |  |  |
| Mlapo | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>4</sup> | ✅ | ✅ | ✅ | ❌ | ❔ | ✅ | ✅ |  |  |  |  |  |  |  |  |
| <abbr title="Multimodal Inputs">mm</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| Multistream Moe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |
| Shared Expert DP | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>1</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ❔ | ✅ |  |  |  |  |  |
| Quantization W4A4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❔ | ❔ | ✅ | ✅ | ❔ | ❌ | ❔ | ❔ | ✅ |  |  |  |  |
| Quantization W4A8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❔ | ❌ | ✅ | ✅ | ❔ | ✅ |  |  |  |
| Quantization W8A8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ |  |  |
| <abbr title="Tensor Parallel">TP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| Weight nz | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 🟠 | ✅ | ✅ | ✅ |

- <sup>1</sup> Only dcp supports dp while pcp does not support dp.
- <sup>2</sup> Flashcomm is only enabled on the prefill stage.
- <sup>3</sup> Lmhead TP is only enabled in the pure dp scenarios.
- <sup>4</sup> MLAPO is only supported on the decode stage.
