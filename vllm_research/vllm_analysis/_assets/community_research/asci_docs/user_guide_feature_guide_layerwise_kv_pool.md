Layerwise KV Pool

Layerwise mode is an optimization for the AscendStore KV Pool that saves and
loads KV cache layer by layer instead of as a single bulk copy. By pipelining
the transfer of one layer with the attention computation of the next, it reduces
the stall that occurs when the entire KV cache must arrive before any forward
progress can be made.

Layerwise mode works in both PD-Mixed (kv_role: "kv_both") and PD
disaggregation (kv_role: "kv_producer" / "kv_consumer") scenarios. See the
KV Pool guide for the general KV Pool architecture and backend
setup.

How It Works (Brief)

Without layerwise mode, the KV cache for a request is saved to (or loaded from)
the pool as one bulk operation after the full forward pass completes (or before
it starts). For long prompts this bulk transfer introduces a serialization
stall.

Layerwise mode splits the save/load at the layer granularity:

1. Saving (producer / kv_both): after computing layer *i*'s attention, the
   KV for that layer is immediately sent to the pool backend. The next layer's
   computation proceeds in parallel with the transfer.
2. Loading (consumer / kv_both): before computing layer *i*'s attention, the
   system waits for layer *i*'s KV to arrive from the pool
   (wait_for_layer_load), then proceeds. The transfer of layer *i+1* overlaps
   with the attention computation of layer *i*.

The net effect: save/load latency is amortized across the forward pass rather
than concentrated as a single blocking step.

Prerequisites

Layerwise mode currently requires the memcache backend
(backend: "memcache"). Install and configure memcache_hybrid before
proceeding — see the KV Pool guide for memcache installation,
config files (mmc-meta.conf / mmc-local.conf), and MetaService startup.

Additional setup:



Configuration

Add use_layerwise: true to the AscendStoreConnector extra config:



Change "kv_role" to "kv_producer" or "kv_consumer" for PD disaggregation.

Key Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| use_layerwise | false | Enable layer-by-layer KV save/load. Requires backend: "memcache". |
| backend | "mooncake" | Storage backend. Layerwise currently supports "memcache" only. |
| mooncake_rpc_port | "0" | RPC port for the scheduler↔worker lookup service. Use "0" to auto-assign, or a unique port per instance. |
| layerwise_prefetch_layers | 1 | Number of layers to prefetch ahead of the compute frontier. Higher values improve overlap at the cost of memory. |
| layerwise_max_transfer_blocks | 0 (unlimited) | Maximum number of KV blocks per transfer batch. |
| layerwise_max_transfer_bytes | 0 (unlimited) | Maximum bytes per transfer batch. |
| h2d_stagger_us | 0 | Stagger delay (microseconds) between H2D copies across TP ranks to avoid bus contention. |
| discard_partial_chunks | true (non-layerwise) / false (layerwise) | Whether to discard KV for incomplete chunk boundaries. Layerwise defaults to false to preserve partial layers. |

Usage Scenarios

PD-Mixed (kv_both)

A single vLLM instance acts as both producer and consumer. The pool serves as a
shared prefix cache: completed requests' KV is saved layer by layer, and new
requests with overlapping prefixes load KV layer by layer. No proxy is needed.



Send requests directly to port 8100 — no proxy required.

PD Disaggregation (kv_producer + kv_consumer)

Separate prefiller and decoder instances. The prefiller saves KV layer by layer;
the decoder loads it layer by layer. A layerwise proxy coordinates request
routing and per-layer KV placement via its /v1/metaserver endpoint.

Prefiller:



Decoder:



Layerwise proxy (different from the standard disagg proxy — serves
/v1/metaserver):



> Note: The proxy --host must not be 0.0.0.0 (wildcard). The
> decoder connects back to host:port/v1/metaserver, so use a reachable IP.

Send requests to the proxy:



Tuning

Prefetch Depth

Increase layerwise_prefetch_layers (default 1) to prefetch more layers
ahead of the compute frontier. This increases transfer/compute overlap but uses
more temporary buffers. Typical values: 1–4.

Transfer Batching

Use layerwise_max_transfer_blocks or layerwise_max_transfer_bytes to limit
the size of each transfer batch. This prevents a single large layer from
monopolizing the transfer bus. Set to 0 (default) for unlimited.

H2D Stagger

On multi-TP deployments, H2D (host-to-device) copies for all TP ranks can
contend on the PCIe/HCCS bus. Set h2d_stagger_us to spread them out (e.g.
100 for a 100 µs stagger between ranks).

Supported Models

Layerwise mode integrates with the MLA (mla_v1) and SFA (sfa_v1)
attention backends. DeepSeek-V2/V3 and other MLA-based models are supported.

Basic full attention (attention_v1) and all context-parallel (CP) variants
(mla_cp, sfa_cp, attention_cp) do not yet integrate the layerwise
wait/save calls. Layerwise + CP is future work.

Limitations

* Backend: Only memcache is supported for layerwise mode (mooncake and
  yuanrong do not support use_layerwise).
* Hybrid KV cache: Not supported — layerwise raises
  NotImplementedError when the model has multiple KV cache group families
  (hybrid MLA + sliding-window attention).
* Context parallel: Layerwise is not yet integrated with CP attention
  backends.
* PD disaggregation proxy: When using kv_producer / kv_consumer, the
  dedicated layerwise proxy
  (load_balance_proxy_layerwise_server_example.py) is required — the standard
  disagg proxy does not serve the /v1/metaserver endpoint.
