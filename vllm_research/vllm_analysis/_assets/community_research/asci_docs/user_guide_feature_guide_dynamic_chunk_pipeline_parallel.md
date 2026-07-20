Dynamic Chunked Pipeline Parallel

!!! note

    For design details and mathematical models, see Design Document. For deployment tutorial, see Dynamic Chunked Pipeline Parallel Tutorial.

Overview

Dynamic Chunked Pipeline Parallel (CPP) is a profiling-based dynamic chunking strategy that optimizes prefill performance for long sequences in Pipeline Parallelism (PP) scenarios. CPP is designed to be used on the Prefiller (P) node in Prefill-Decode (PD) disaggregation deployments. By dynamically calculating the optimal chunk size based on profiling data, CPP significantly reduces Time-To-First-Token (TTFT) for long sequences on P nodes.

:::{important}
CPP should be configured on the P (Prefiller) node in a PD disaggregation setup. The D (Decoder) node does not require CPP configuration. For PD disaggregation deployment guidance, refer to the tutorials below:

- PD Disaggregation Single Node (Qwen2.5-VL)
- PD Disaggregation Multi Node (Deepseek)
:::

When to Use

- PD disaggregation P node: Enables CPP on the Prefiller node to optimize long-sequence prefill with Pipeline Parallelism. The Decoder node does not need CPP.
- Variable-length sequence serving: PP does not introduce degradation on short sequences, and gains benefits through dynamic chunks on long sequences.
- Ultra-long sequence inference: For sequences exceeding single-machine memory capacity (e.g., 1M tokens), dynamic chunking significantly reduces pipeline idle time.

Supported Scenarios

CPP focuses on optimization during the prefill phase on the P node in PD disaggregation scenarios. It is better to be used in PD disaggregation scenarios. Supported features are as follows:

|         | Eager | Graph | Prefix <br> Cache | Chunked <br> Prefill |
| ------- | ----- | ----- | ------ | ------ |
| CPP | ✅    | ✅     | ✅      | ✅       |

How to Enable

PD Disaggregation Deployment Example

In a PD disaggregation setup, enable CPP only on the P (Prefiller) node. Below is a complete example using the MooncakeConnector for a 1P1D architecture.

Note:

- It is currently known that async-scheduling may cause performance degradation in the prefill stage of PP, and async-scheduling provides minimal benefit to prefill. Therefore, it is currently recommended not to enable asynchronous scheduling on P nodes of PP.
- It is recommended to use MooncakeConnectorV1 as the kv_connector, as it provides more comprehensive support for PP.

:::::{tab-set}

::::{tab-item} P Node (Prefiller — with CPP)



::::

::::{tab-item} D Node (Decoder — without CPP)



::::{tab-item} Example Proxy for Deployment

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: load\_balance\_proxy\_server\_example.py



| Parameter | Meaning |
| --- | --- |
| --port | Port of proxy |
| --prefiller-port | All ports of prefill |
| --decoder-ports | All ports of decoder |

::::

::::{tab-item} Verification

Check service health using the proxy server endpoint.



::::

:::::

> Key points for PD disaggregation with CPP:
>
> - CPP (profiling_chunk_config.enabled, --pipeline-parallel-size > 1) is configured only on the P node.
> - The D node runs without pipeline parallelism — it focuses on low-latency token-by-token decoding.
> - For complete PD disaggregation setup instructions (environment verification, Mooncake installation, proxy deployment), see:
>     - PD Disaggregation Single Node
>     - PD Disaggregation Multi Node

Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | False | Enable/disable Dynamic Chunked Pipeline Parallel |
| smooth_factor | float | 1.0 | Smoothing factor (0 < x ≤ 1.0). Higher values trust dynamic prediction more |
| min_chunk | int | 4096 | Minimum chunk size for dynamic calculation |
| need_timing | bool | True | Enable/disable Online Calibration |
| max_fit_chunk | int | 30 | Number of chunk-time data for Online Calibration |

Parameter Tuning

- smooth_factor: Controls trust level in dynamic prediction
    - 1.0: Strictly follow model prediction
    - 0.6~0.85: Balance dynamic adjustment and scheduling overhead
    - 0.0: No dynamic adjustment (degrades to fixed chunking)
- min_chunk: Generally doesn't need adjustment. Should be smaller than max-num-batched-tokens

Recommended Settings

max-num-batched-tokens

Notably, the TTFT of CPP is very sensitive to max-num-batched-tokens (considered the initial chunksize for dynamic solving). Because if it is too large, it will introduce significant computational waste, and if it is too small, it will lead to a decrease in operator efficiency. To leave enough room for dynamic adjustments, we recommend that the longer the sequence being processed, the larger the max-num-batched-tokens should be set. Recommended values:

| Sequence Length | max-num-batched-tokens |
|-----------------|--------------------------|
| 64k             | 20480                    |
| 128k            | 32768                    |

Online Calibration

For optimal performance, online calibrate with real data before production:

You can use aisbench to generate fixed-length random datasets. Refer to Using AISBench for performance evaluation for details.

1. Modify <YOUR_AISBENCH_PATH>/benchmark/ais_bench/datasets/synthetic/synthetic_config.py:

    

2. Run for online calibration:

    

Configure online calibration data length to match your max-model-len. Use batch_size=1 and ensure data differs to avoid cache hits if prefix caching is enabled.

Performance

Refer to Using AISBench for performance evaluation for details.

To evaluate the effectiveness of Dynamic Chunked Pipeline Parallel in long sequence LLM inference scenarios, we use DeepSeek-V3.1-W8A8 and Qwen3-235B, deploy P instance in Ascend Atlas A3 inference products*64G (A3), the configuration and performance data are as follows.

Fixed-length requests, concurrency=1:

- DeepSeek-V3.1-W8A8:

    | Configuration | CPP <br> (Dynamic Chunk, <br> chunksize=32k) | PP <br>(Static Chunk, <br> chunksize=32k) |
    | ----------------------------- | ------------------------- | ------------------------- |
    | Input length  128k    | TTFT: 22.5s | TTFT: 27.0s |

- Qwen3-235B:

    | Configuration | CPP <br> (Dynamic Chunk, <br> chunksize=32k) | PP <br>(Static Chunk, <br> chunksize=32k) |
    | ----------------------------- | ------------------------- | ------------------------- |
    | Input length  256k    | TTFT: 53.5s | TTFT: 61.4s |

Variable-length requests, concurrency=4:

- DeepSeek-V3.1-W8A8:

    | Configuration | 4k~64k Input, mean=32k, std=32k <br> prefix hit rate=99% |
    | ----------------------------- | ------------------------- |
    |  CPP2TP8   | Input throughput: 22424 tps/card |
    |  DP2TP8   | Input throughput: 16150 tps/card |
    |  PCP2TP8   | Input throughput: 18197 tps/card |
    |  TP16   | Input throughput: 18875 tps/card |

Constraints

- Pipeline Parallelism Required: --pipeline-parallel-size > 1
- Chunked Prefill Required: --enable-chunked-prefill
- Incompatible with Balance Scheduling: Cannot enable VLLM_ASCEND_BALANCE_SCHEDULING
- Startup Overhead: Profiling adds ~64 forward passes (tens of seconds)
