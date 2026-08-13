# vLLM KV Transfer Code Summary (P/D disaggregation)

Codebase: `/Users/backyes/work/triton/vllm/vllm/distributed/kv_transfer/`
Scope: kv_pipe + kv_lookup_buffer abstractions + kv_connector re-export.

## Three-layer abstraction (README)
vLLM models KV transport as three composable layers:

1. **KVPipe** — FIFO pipe for `torch.Tensor`. API: `send_tensor` / `recv_tensor`.
2. **KVLookupBuffer** — keyed lookup on top of the pipe. Key = (tokens, roi); value = (key KV, value KV, hidden states). API: `insert` / `drop_select` (SL-destructive pop semantics).
3. **KVConnector** — wires pipe+buffer into vLLM. API: `send_kv_caches_and_hidden_states` / `recv_kv_caches_and_hidden_states`.

Why a buffer layer exists: prefill and decode workers do not share request ordering (prefill may emit A,B,C; decode may drain C first). FIFO alone can't serve that — the buffer reorders by token key.

- **Pipe is bypassable**: if your comm layer already supports key/value lookup (Redis, RDMA DB) go straight to lookup_buffer.
- **Both bypassable**: to alter vLLM execution flow (e.g. receive partial KV and prefill the rest) subclass the connector directly — at risk of breakage across vLLM versions.

`kv_connector/base.py` itself is just a stub re-exporting `KVConnectorBase_V1` from `kv_connector/v1`.

---

## kv_pipe/base.py — KVPipeBase
Abstract base `KVPipeBase(ABC)`:
- `send_tensor(tensor: Optional[torch.Tensor])` — must support `None` (error propagation).
- `recv_tensor() -> Optional[torch.Tensor]`.
- `close()`.
Notable **TODO**: add a `key` argument so a traditional KV datastore can back the pipe directly.

## kv_pipe/mooncake_pipe.py — MooncakeTransferEngine + MooncakePipe
Two classes:
- **`MooncakeTransferEngine`** wraps `mooncake.engine.TransferEngine` + ZMQ. Rank 0 = kv prefill side. Config `MooncakeTransferEngineConfig` from `MOONCAKE_CONFIG_PATH` JSON, with `protocol`/`device_name`/`metadata_backend` (etcd or redis). Key methods:
  - `allocate_managed_buffer / free_managed_buffer` — mooncake-managed GPU/CPU buffers.
  - `transfer_sync(peer_buffer_address, length)` → RDMA `transfer_sync_read` read from remote URL.
  - `send_bytes(user_data)` — alloc buffer, write, ZMQ PUSH metadata (buffer ptr + length), async ACK cleanup via `ThreadPoolExecutor(1)`.
  - `recv_bytes()` — ZMQ PULL, remote-sourced RDMA read into local buffer, ACK, free.
  - Uses `transfer_sync_read` (pull semantic).
- **`MooncakePipe(KVPipeBase)`**:
  - Serializes tensors with `safetensors.save({tensor}).tobytes()` → `engine.send_bytes`.
  - `recv` deserializes safetensors and `.to(self.device)`.
  - None-sent via a sentinel tensor `[-150886311]` (int, `NONE_INT`).
  - Rank-direction picked in ctor by `kv_rank`: prefill_url binds, decode_url connects for rank 0; inverted for rank 1.

**Limitations**: tensors serialized via safetensors and shipped through ZMQ control + one-sided RDMA — serial per rank, no pipelining; pull-based only; metadata sockets need offset bookkeeping for TP/PP.

## kv_pipe/pynccl_pipe.py — PyNcclPipe
`PyNcclPipe(KVTransferConfig)`:
- Topology: default `kv_rank` 0/1, target = `(rank ± 1) % kv_parallel_size`.
- Backed by `StatelessProcessGroup` (TCP store bootstrap) + `PyNcclCommunicator` for CUDA; CPU path is an explicitly documented **control-plane-only** path (not KV transfer).
- Protocol per tensor: send `Metadata = {dtype, shape}` via `group.send_obj`, then `device_send_func(tensor, target)`; recv is symmetric.
- **Backpressure** via `buffer_size` + `buffer_size_thresh` (lock-protected). `block_if_full()` polls 50 ms. `send_tensor` submits through `ThreadPoolExecutor(1)`.
- Constants `METADATA_LENGTH=16`, `MAX_TENSOR_DIMENSIONS=14`, `METADATA_DTYPE=torch.int64` reserved for future binary metadata.
- `close()` shuts down the transport thread.

**Limitations**: single-pair topology; explicit `TODO` implied comment that CPU send/recv must not be repurposed for KV; backpressure is coarse polling.

---

## kv_lookup_buffer/base.py — KVCacheBufferBase / KVLookupBufferBase / KVStoreBufferBase
Three ABCs under `KVCacheBufferBase` (abstract `close()`):

- **`KVLookupBufferBase`** — token-keyed store.
  - `insert(input_tokens, roi, key, value, hidden)` — `roi` is a binary mask over tokens (which tokens this KV applies to; extension for TP/PP-sharded KV is marked "not implemented").
  - `drop_select(input_tokens=None, roi=None) -> list[Optional[Tensor]]` — destructive pop. Passing `None` means "pop any entry" (offload to external KV service).
  - FIXMEs in docstring: future should narrow to `(key_tensor_dict, value_tensor_dict)` and transmit both hidden *and* sampler outputs.

- **`KVStoreBufferBase`** — flat string-keyed store.
  - `put(key: str, value: Optional[Tensor])`, `get(key: str) -> Optional[Tensor]`.
  - Models a distributed KV store (Redis-like), enables arbitrary-granularity KV transfer.

## kv_lookup_buffer/mooncake_store.py — MooncakeStore
`MooncakeStore(KVStoreBufferBase)` wraps `mooncake.store.MooncakeDistributedStore`. Config `MooncakeStoreConfig` from `MOONCAKE_CONFIG_PATH`, defaults 3.125 GiB global segment / 1 GiB local buffer, `metadata_backend` etcd/redis implied by server.
- `_put_impl(key, value)` — saves `{tensor, device_id}` via safetensors, calls `store.put`.
- `_get_impl(key)` — `store.get`, safetensors load, restores to original cuda device via stored `device_id`.
- `close` — no-op; C++ dtor handles teardown.
- Explicit TODO in places: "needs a message queue before it can be made async" — currently synchronous blocking put/get.

---

## Connector-Pipe-Buffer pattern
The intended composition:
```
worker ──► KVConnector
               │ owns
               ▼
        KVLookupBuffer (token-ordered cache)
               │ backed by (if needed)
               ▼
         KVPipe (tensor transport between ranks)
```
and the variant where the lookup buffer *is* the transport:
```
worker ──► KVConnector
               │ owns
               ▼
        KVStoreBuffer (flat KV store ↔ RDMA via mooncake store)
              (no pipe — true KV datastore behind)
```

concretely: `MooncakeConnector` historically pairs `MooncakePipe` + a `SimpleBuffer` (a concrete `KVLookupBufferBase` — not present in the reviewed set, lives in `kv_lookup_buffer/simple_connector_buffer.py` or similar). For pure store-style transfers it pairs `MooncakeStore` directly with the connector.

**Summary cheat sheet**
| Class / base              | Role                          | Direction       |
|---------------------------|-------------------------------|-----------------|
| `KVPipeBase`              | FIFO tensor pipe              | push pull       |
| `PyNcclPipe`              | NCCL tensor pipe              | push            |
| `MooncakePipe`            | RDMA (safetensors) pipe       | pull (read)     |
| `KVCacheBufferBase`       | common buffer root            | —               |
| `KVLookupBufferBase`      | token-keyed KV cache          | insert/pop      |
| `KVStoreBufferBase`       | string-keyed distributed KV   | put/get         |
| `MooncakeStore`           | mooncake-backed KV store      | put/get         |
| `KVConnectorBase` (V1)    | top-level connector           | send/recv        |

Key recurring TODOs/issues:
- Pipe lacks a `key` argument — blocks cleaner KV-backed pipes.
- NCCL pipe CPU path is control-plane only.
- Lookup buffer key/value shape is slated for refactor to tensor-dicts; sampler output transfer is missing.
- Mooncake store/pipe are sync; a message queue is needed for async.
- roi-based TP/PP sharding awareness is flagged but not implemented.
