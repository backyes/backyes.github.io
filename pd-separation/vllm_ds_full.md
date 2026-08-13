User Guide
Getting Started
Quickstart
Installation
Examples
General
vLLM V1
Frequently Asked Questions
Production Metrics
Reproducibility
Security
Troubleshooting
Usage Stats Collection
Inference and Serving
Offline Inference
Online Serving
Context Parallel Deployment
Data Parallel Deployment
Troubleshooting distributed deployments
Expert Parallel Deployment
Parallelism and Scaling
Integrations
Deployment
Using Docker
Using Kubernetes
Using Nginx
Frameworks
Integrations
Training
Async Reinforcement Learning
What is Layerwise (Re)loading?
Reinforcement Learning from Human Feedback
Transformers Reinforcement Learning
Weight Transfer
Configuration
Conserving Memory
Engine Arguments
Environment Variables
Model Resolution
Optimization and Tuning
Server Arguments
TPU
Models
Supported Models
Generative Models
Pooling Models
Extensions
Hardware Supported Models
Features
Automatic Prefix Caching
Batch Invariance
Context Extension
Custom Arguments
Custom Logits Processors
Disaggregated Encoder
Disaggregated Prefilling (experimental)
IndexCache
Interleaved Thinking
KV Offloading Usage Guide
LoRA Adapters
MooncakeConnector Usage Guide
MooncakeStoreConnector Usage Guide
MoRIIOConnector Usage Guide
Multimodal Inputs
NixlConnector Compatibility Matrix
NixlConnector Usage Guide
Per-Request Metrics
Prompt Embedding Inputs
Reasoning Outputs
Sleep Mode
Structured Outputs
Tool Calling
Quantization
Speculative Decoding
Table of contents
Why disaggregated prefilling?
Usage example
Development
Third-party contributions
Home
User Guide
Features
Disaggregated Prefilling (experimental)¶

This page introduces you to the disaggregated prefilling feature in vLLM.

Note

This feature is experimental and subject to change.

Why disaggregated prefilling?¶

Two main reasons:

Tuning time-to-first-token (TTFT) and inter-token-latency (ITL) separately. Disaggregated prefilling put prefill and decode phase of LLM inference inside different vLLM instances. This gives you the flexibility to assign different parallel strategies (e.g. tp and pp) to tune TTFT without affecting ITL, or to tune ITL without affecting TTFT.
Controlling tail ITL. Without disaggregated prefilling, vLLM may insert some prefill jobs during the decoding of one request. This results in higher tail latency. Disaggregated prefilling helps you solve this issue and control tail ITL. Chunked prefill with a proper chunk size also can achieve the same goal, but in practice it's hard to figure out the correct chunk size value. So disaggregated prefilling is a much more reliable way to control tail ITL.

Note

Disaggregated prefill DOES NOT improve throughput.

Usage example¶

Now supports 9 types of connectors:

ExampleConnector: refer to 
 examples/disaggregated/example_connector/run.sh for the example usage of ExampleConnector disaggregated prefilling.
LMCacheConnectorV1: refer to 
 examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh for the example usage of LMCacheConnectorV1 disaggregated prefilling which uses NIXL as the underlying KV transmission. LMCache also offers a multi-process (MP) mode via LMCacheMPConnector, where a standalone lmcache server holds the KV cache shared by one or more vLLM instances; see the 
 LMCache examples and the LMCache docs for setup.
NixlConnector: refer to 
 tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh for the example usage of NixlConnector disaggregated prefilling which support fully async send/recv. For detailed usage guide, see NixlConnector Usage Guide. For feature compatibility details, see NixlConnector Compatibility Matrix. You may specify one or multiple NIXL transfer backends, such as:
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_buffer_device":"cuda", "kv_connector_extra_config":{"backends":["UCX", "GDS"]}}'

MooncakeConnector: refer to 
 examples/disaggregated/mooncake_connector/run_mooncake_connector.sh for the example usage of MooncakeConnector disaggregated prefilling. For detailed usage guide, see MooncakeConnector Usage Guide.
MoRIIOConnector (ROCm only): see MoRI-IO Usage Guide for example usage and detailed documentation.
MultiConnector: take advantage of the kv_connector_extra_config: dict[str, Any] already present in KVTransferConfig to stash all the connectors we want in an ordered list of kwargs.such as:
--kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both"},{"kv_connector":"ExampleConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"local_storage"}}]}}'

OffloadingConnector: enable offloading of KV data to CPU memory, customizing the CPU block size (in tokens) and total CPU memory bytes to allocate:
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size": 64, "cpu_bytes_to_use": 1000000000}}'


For multi-tier offloading (e.g., CPU + filesystem tier) and the full configuration reference, see the KV Offloading Usage Guide.

FlexKVConnectorV1: refer to 
 examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py for the example usage of FlexKVConnectorV1. FlexKV is a distributed KV Store and multi-level cache management system for ultra-large-scale LLM inference.
--kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'

Development¶

We implement disaggregated prefilling by running 2 vLLM instances. One for prefill (we call it prefill instance) and one for decode (we call it decode instance), and then use a connector to transfer the prefill KV caches and results from prefill instance to decode instance.

All disaggregated prefilling implementation is under vllm/distributed/kv_transfer.

Key abstractions for disaggregated prefilling:

Connector: Connector allows kv consumer to retrieve the KV caches of a batch of request from kv producer.
LookupBuffer: LookupBuffer provides two API: insert KV cache and drop_select KV cache. The semantics of insert and drop_select are similar to SQL, where insert inserts a KV cache into the buffer, and drop_select returns the KV cache that matches the given condition and drop it from the buffer.
Pipe: A single-direction FIFO pipe for tensor transmission. It supports send_tensor and recv_tensor.

Note

insert is non-blocking operation but drop_select is blocking operation.

Here is a figure illustrating how the above 3 abstractions are organized:

The workflow of disaggregated prefilling is as follows:

The buffer corresponds to insert API in LookupBuffer, and the drop_select corresponds to drop_select API in LookupBuffer.

Now every process in vLLM will have a corresponding connector. Specifically, we have:

Scheduler connector: the connector that locates in the same process as the scheduler process. It schedules the KV cache transfer ops.
Worker connectors: the connectors that locate in the worker processes. They execute KV cache transfer ops.

Here is a figure illustrating how the above 2 connectors are organized:

The figure below shows how the worker connector works with the attention module to achieve layer-by-layer KV cache store and load:

Third-party contributions¶

Disaggregated prefilling is highly related to infrastructure, so vLLM relies on third-party connectors for production-level disaggregated prefilling (and vLLM team will actively review and merge new PRs for third-party connectors).

We recommend three ways of implementations:

Fully-customized connector: Implement your own Connector, and call third-party libraries to send and receive KV caches, and many many more (like editing vLLM's model input to perform customized prefilling, etc.). This approach gives you the most control, but at the risk of being incompatible with future vLLM versions.
Database-like connector: Implement your own LookupBuffer and support the insert and drop_select APIs just like SQL.
Distributed P2P connector: Implement your own Pipe and support the send_tensor and recv_tensor APIs, just like torch.distributed.
June 19, 2026
 Back to top