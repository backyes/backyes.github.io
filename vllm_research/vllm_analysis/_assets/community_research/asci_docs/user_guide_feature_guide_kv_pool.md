KV Cache Pool（Ascend Store）Deployment Guide

Contents

* Environmental Dependencies
* Example of using Mooncake as a KV Pool backend
* Example of using Memcache as a KV Pool backend
* Example of using Yuanrong as a KV Pool backend
* FAQ

Environmental Dependencies

* Software:
    * CANN >= 8.5.0
    * vLLM：main branch
    * vLLM-Ascend：main branch
    * mooncake：>= 0.3.11.post1

KV Pool Parameter Description

kv_load_failure_policy: KV Load Failure Handling Policy

kv_load_failure_policy is a top-level field in kv-transfer-config.

* recompute: When KV loading fails, vLLM rolls the request back to the last valid prefix and reschedules it to recompute the failed KV blocks. Hybrid attention models (e.g. DeepSeekV4, Qwen 3.5) are not supported yet.
* fail: When KV loading fails, the affected request is terminated directly with an error.

The default value in vLLM is fail. If you want the request to fall back to recomputation after a KV load failure, set it to recompute.

When MultiConnector is used, configure kv_load_failure_policy on the MultiConnector top-level kv-transfer-config instead of the child connectors.

kv_connector_extra_config: Additional Configurable Parameters for Pooling

| Parameter | Description |
| :--- | :--- |
| lookup_rpc_port | Port for RPC Communication Between Pooling Scheduler Process and Worker Process: Each Instance Requires a Unique Port Configuration. |
| load_async | Whether to Enable Asynchronous Loading. The default value is false. |
| backend | Set the storage backend for kvpool (mooncake, memcache, yuanrong), with the default being mooncake. |
| consumer_is_to_put | Whether Decode node put KV Cache into KV Pool. The default value is false. |
| consumer_is_to_load | Whether Decode node load KV cache from KV Pool. The default value is false. |
| prefill_pp_size | Prefill PP size, needs to be set when Prefill node enables PP. |
| prefill_pp_layer_partition | Prefill PP layer partition, needs to be set when Prefill node enables PP. |

Environment Variable Configuration

To guarantee uniform hash generation, it is required to synchronize the PYTHONHASHSEED environment variable across all nodes upon enabling KV Pool.



Example of using Mooncake as a KV Pool backend

* Software:
    * Check NPU HCCN Configuration:

        Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

        

    * Install Mooncake

        Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI.
        The Mooncake wheel requires glibc 2.35 or later. Check the installed glibc version before installation:

        

        Install Mooncake with pip:

        

Environment Variables Description

| Hardware | Dependencies | Export Command | Description |
| :--- | :--- | :--- | :--- |
| 800 I/T A3 series | HDK >= 26.0<br>or HDK >= 25.5 with mooncake >= v0.3.11<br>CANN >= 9.0.0<br>LingQu Computing Network >= 1.5 | export ASCEND_ENABLE_USE_FABRIC_MEM=1 | Recommended. Enables unified memory address direct transmission scheme. With SSD offload, see Fabric memory size alignment — memory sizes must be aligned to 1GB. |
| 800 I/T A3 series | If any dependency above is not met | export ASCEND_BUFFER_POOL=4:8 | Configures the number and size of buffers on the NPU Device for aggregation and KV transfer (e.g., 4:8 means 4 buffers of 8MB). |
| 800 I/T A2 series | HDK >= 25.5 is recommended | export HCCL_INTRA_ROCE_ENABLE=1 | Required by direct transmission scheme on 800 I/T A2 series|

Run Mooncake Master

Note: Before proceeding, review the following Mooncake guides:

* Mooncake Store Deployment Guide
* SSD Offload

1. Configure mooncake.json

The environment variable MOONCAKE_CONFIG_PATH is configured to the full path where mooncake.json is located.



metadata_server: Configured as P2PHANDSHAKE.  
protocol: Must be set to 'Ascend' on the NPU.
device_name: ""
master_server_address: Configured with the IP and port of the master service. It can also be set via the MOONCAKE_MASTER environment variable, which takes precedence over this configuration item (useful for injecting the master address through Kubernetes).  
global_segment_size: Registered memory size per card to the KV Pool. Needs to be aligned to 1GB. It can also be set via the MOONCAKE_GLOBAL_SEGMENT_SIZE environment variable, which takes precedence over this configuration item.  
preferred_segment: Whether to prefer storing KV on the local segment when putting objects to the KV Pool. Defaults to false.  
prefer_alloc_in_same_node: Whether to prefer allocating KV on the same node. Defaults to true.

2. Start mooncake_master

Under the mooncake folder:



eviction_high_watermark_ratio determines the watermark where Mooncake Store will perform eviction, and eviction_ratio determines the portion of stored objects that would be evicted.
default_kv_lease_ttl controls the default lease TTL for KV objects (milliseconds); configure it via --default_kv_lease_ttl and keep it larger than ASCEND_CONNECT_TIMEOUT and ASCEND_TRANSFER_TIMEOUT.

PD Disaggregation Scenario

1. Run prefill Node and decode Node

Using MultiConnector to simultaneously utilize both MooncakeConnectorV1 and AscendStoreConnector. MooncakeConnectorV1 performs kv_transfer, while AscendStoreConnector serves as the prefix-cache node.

prefill Node:



The content of the multi_producer.sh script:



decode Node：



The content of multi_consumer.sh:



Currently, the key-value pool in PD Disaggregate only stores the kv cache generated by the Prefill node by default. In models using MLA, it is now supported that the Decode node stores the kv cache for use by the Prefill node, enabled by adding consumer_is_to_put: true to the AscendStoreConnector. If the Prefill node enables PP, prefill_pp_size or prefill_pp_layer_partition also needs to be set. Example as follows:



2. Start proxy_server



Change localhost to your actual IP address.

3. Run Inference

Configure the localhost, port, and model weight path in the command to your own settings.

Short question:



Long question:



PD-Mixed Inference

1. Run Mixed Deployment Script



Content of pd_mix.sh:



2. Run Inference

Configure the localhost, port, and model weight path in the command to your own settings. The requests sent will only go to the port where the mixed deployment script is located, and there is no need to start a separate proxy.

Short question:



Long question:



Note: For MooncakeStore with ASCEND_BUFFER_POOL enabled, it is recommended to perform a warm-up phase before running actual performance benchmarks.

This is because HCCL one-sided communication connections are created lazily after the instance is launched when Device-to-Device communication is involved. Currently, full-mesh connections between all devices are required. Establishing these connections introduces a one-time time overhead and persistent device memory consumption (4 MB of device memory per connection).

For warm-up, it is recommended to issue requests with an input sequence length of 8K and an output sequence length of 1, with the total number of requests being 2–3× the number of devices (cards/dies).

Enable MooncakeStore SSD Offload with Embedded Real Client Mode

* Requires mooncake >= v0.3.11.

Start the master

Start Mooncake master as described in Run Mooncake Master. To enable SSD offload, add --enable_offload=true to the same master startup command. For example:



| Field | Description |
| :--- | :--- |
| enable_offload | Set to true to enable SSD offload in Mooncake master. Keep the master port aligned with master_server_address in mooncake.json. |
| client_ttl | Seconds a client stays alive after the last Ping. CLI default is 10; see SEGMENT_NOT_FOUND with SSD offload. |

Configuration

Starting from the mooncake.json configured in Run Mooncake Master, add the following SSD offload fields:



| Field | Description |
| :--- | :--- |
| enable_ssd_offload | Set to true to enable SSD offload. Environment variables are not supported. |
| ssd_offload_path | Required when enable_ssd_offload is true. Absolute path to a local directory where Mooncake stores offloaded KV data (for example, /nvme/mooncake_offload). The directory must exist and be writable by the vLLM process; create it before startup (mkdir -p <path>). Relative paths, symbolic links, and paths containing .. are rejected by Mooncake. Passed to MooncakeDistributedStore.setup() as the SSD storage root (equivalent to MOONCAKE_OFFLOAD_FILE_STORAGE_PATH in standalone clients). Configure this field in mooncake.json only; environment variables are not supported. |

Running the Embedded Real Client

With Mode A (Embedded Real Client), Mooncake is embedded in vLLM. When the vLLM service starts, AscendStoreConnector / MooncakeBackend automatically calls MooncakeDistributedStore.setup() using the settings in mooncake.json (including enable_ssd_offload and ssd_offload_path when SSD offload is enabled). No separate mooncake_client process is required.

SSD Disk Usage Control

The following environment variables control disk space usage for SSD offload (bucket backend):

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES | 1342177280 (1280 MB) | Per-rank SSD read/write buffer size in bytes. Not configurable in mooncake.json. If you hit BUFFER_OVERFLOW, increase this value — see Sizing MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES. On A3 with ASCEND_ENABLE_USE_FABRIC_MEM=1, must be aligned to 1GB and counts toward per-rank fabric mem quota (see Fabric memory size alignment). |
| MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE | 0 | Eviction threshold in bytes. When set to 0, the backend uses 90% of the physical disk capacity as the quota. Set an explicit value to control disk usage precisely. |
| MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY | none | Eviction policy: none (writes fail when full), fifo, or lru. |
| MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES | 2199023255552 (2 TB) | Per-rank maximum disk usage reported to Mooncake master. Master aggregates this across clients (roughly 2 TB × rank count in the SSD Storage total). Always override to match real disk capacity — the default often exceeds available space. |

MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES risk: If left at the 2 TB default, master shows a total SSD quota far larger than the physical disk (e.g. 16 ranks → ~32 TB displayed on a 1 TB NVMe). Offload still fails when the disk fills, while monitoring looks healthy. Set this to your actual per-rank budget before production use.

Since each TP rank uses an independent SSD subdirectory (rank_0/, rank_1/, ...) under ssd_offload_path, all ranks share the same physical disk. To prevent a single rank from consuming excessive space, set an explicit per-rank quota. For example, with an 800 GB disk and 8 TP ranks:



Example of using Memcache as a KV Pool backend

Installing Memcache

MemCache depends on MemFabric. Therefore, MemFabric must be installed. Installing the memcache after the memfabric is installed.



Configuring the memcache Config File

mmc-meta.conf：



mmc-local.conf：



Key Focuses：

| Parameter | Description |
| :--- | :--- |
| ock.mmc.meta_service_url | Configure the IP address and port number of the master node. The IP address and port number of the P node and D node can be the same. |
| ock.mmc.local_service.config_store_url | Configure the IP address and port number of the master node. The IP address and port number of the P node and D node can be the same. |
| ock.mmc.local_service.world_size | Total count of local service, including services that will be added in the future. |
| ock.mmc.local_service.protocol | device_rdma (supported for A2 and A3 when device ROCE available, recommended for A2), device_sdma (supported for A3 when HCCS available, recommended for A3). Currently does not support heterogeneous protocol setting.|
| ock.mmc.local_service.dram.size | Sets the size of the memory occupied by the master. The configured value is the size of the memory occupied by each card. |

Run Memcache Master

Starting the MetaService service.

Run pip show memcache_hybrid and find the Location value in the output. Use that value as {INSTALL_PATH} below.





PD Disaggregation Scenario

1. Run prefill Node and decode Node

Using MultiConnector to simultaneously utilize both MooncakeConnectorV1 and AscendStoreConnector. MooncakeConnectorV1 performs kv_transfer, while AscendStoreConnector enables KV Cache Pool

800I A2/800T A2/800I A3/800T A3 Series

run_prefill.sh/run_decode.sh:



2. Start proxy_server

Refer to Start proxy_server in the MooncakeStore deployment section.

3. Run Inference

Refer to Run Inference in the MooncakeStore deployment section.

PD-Mixed Scenario

1. Run Mixed Deployment Script

800I A2/800T A2/800I A3/800T A3 Series

Run_pd_mix.sh:



2. Run Inference

Example of using Yuanrong as a KV Pool backend

* Software:
    * Install openyuanrong-datasystem on all nodes (yr.datasystem must be importable).

Install Yuanrong Datasystem



If the prebuilt package does not match the CANN or Ascend driver version in
your environment, build Yuanrong Datasystem from source in the vLLM Ascend
image. Follow the official Yuanrong Datasystem build instructions:
<https://atomgit.com/openeuler/yuanrong-datasystem>

Start etcd

Yuanrong Datasystem uses etcd for service discovery. The following example
starts a single-node etcd cluster:



For production environments, refer to the official etcd clustering
documentation: <https://etcd.io/docs/v3.7/op-guide/clustering/>

Start Datasystem Worker

Start a Datasystem worker on each node by using dscli. The following
configuration is a recommended starting point for high-throughput KV Pool
workloads:



The --worker_address value is consumed later by DS_WORKER_ADDR, so keep
the host and port identical on the same node.

The tuning parameters above have the following effects:

| Parameter | Description |
| :--- | :--- |
| log_dir | Sets the Datasystem worker log directory. Create the directory and grant the worker process write permission before startup. |
| arena_per_tenant=1 | Uses one shared-memory arena per tenant as a conservative starting point for memory and file-descriptor usage. |
| enable_huge_tlb=true | Backs worker shared memory with HugeTLB pages. Reserve enough 2 MiB huge pages before starting the worker. |
| enable_fallocate=false | Disables fallocate for the shared-memory file; use this setting with the HugeTLB configuration above. |
| rpc_thread_num=64 | Sets the RPC/ZMQ service concurrency. |
| oc_thread_num=64 | Sets the Object Cache business-thread pool size. |
| enable_worker_worker_batch_get=true | Enables batched Object Cache reads between Datasystem workers. |
| sc_regular_socket_num=0, sc_stream_socket_num=0 | Disables the Stream Cache service. Both values must be greater than zero to enable it; keep them at zero when KV Pool does not use Stream Cache. |

For shared_memory_size_mb=40960, reserve at least 20480 2 MiB huge pages and
verify that they are available before starting the worker:



Worker logs, including files whose base name is normally
datasystem_worker, are written under the --log_dir directory. Use an
absolute path so the log location does not depend on the worker process's
current directory.

These thread counts are tuning starting points rather than universal defaults.
Adjust them according to the available CPU cores and measured request
throughput. Because -w consumes the remaining command-line arguments, place
any dscli start options such as --timeout before -w.

For more parameters, refer to the dscli usage documentation on the Yuanrong
Datasystem official site:
<https://atomgit.com/openeuler/yuanrong-datasystem>

To stop the worker:



Environment Variable Configuration

Set the following environment variables on each node before starting vLLM:

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| PYTHONHASHSEED | Yes | 0 | Must be consistent across all nodes to guarantee uniform hash generation. |
| DS_WORKER_ADDR | Yes | N/A | Datasystem worker address in <host>:<port> format. This must match the local dscli start --worker_address value. |
| DATASYSTEM_CLIENT_LOG_DIR | No | ~/.datasystem/logs | Directory for Yuanrong client SDK logs created by the vLLM process. Use a directory separate from the worker logs. |
| DS_ENABLE_EXCLUSIVE_CONNECTION | No | 0 | Passed to Yuanrong HeteroClient.enable_exclusive_connection. Use 1 to enable the exclusive connection mode when required by your deployment. |
| DS_ENABLE_REMOTE_H2D | No | 0 | Passed to Yuanrong HeteroClient.enable_remote_h2d. Use 1 only after the Remote H2D requirements below are met. |



Set DATASYSTEM_CLIENT_LOG_DIR before starting vLLM because the Yuanrong
client reads it during logging initialization. Client SDK logs, whose base
name is normally ds_client, are written to this directory.

Remote H2D Requirements

Set DS_ENABLE_REMOTE_H2D=1 only when Remote Host-to-Device transfer is
enabled and verified in the Yuanrong Datasystem deployment:

* Reserve enough 2 MiB HugeTLB pages before starting the worker. For 40 GiB
  shared memory, reserve at least 20480 2 MiB huge pages.
* Start each Datasystem worker with Remote H2D enabled. The worker start
  command must include --remote_h2d_device_ids, --enable_huge_tlb true,
  --arena_per_tenant 1, and --enable_fallocate false. Using multiple
  available NPU device IDs is recommended, for example "0,1,2,3,4,5,6,7" on
  an 8-NPU node.



* Make sure the NPU driver, firmware, and CANN toolkit required by Yuanrong
  Remote H2D are installed and visible to the worker process. In containers,
  mount the Ascend driver path, npu-smi, hccn_tool, /etc/hccn.conf,
  /etc/ascend_install.info, and the required /dev/davinci* devices.
* Verify the NPU and RoCE environment before enabling the client flag:



If these checks fail, keep DS_ENABLE_REMOTE_H2D=0 and use the default
Datasystem transfer path.

Run AscendStoreConnector with Yuanrong backend

Use AscendStoreConnector with backend: "yuanrong":



lookup_rpc_port is the RPC port used between the pooling scheduler process
and the worker process. Each instance must use a unique port value.

Notes

* The Yuanrong backend normalizes KV keys before calling Datasystem. Supported
  ASCII keys up to 1024 bytes are preserved. Longer keys or keys containing
  unsupported characters are rewritten to a maximum of 1024 characters with a
  hash suffix, so do not rely on the raw key string when debugging backend
  storage.
* No extra buffer pre-registration step is required for Yuanrong. The backend
  uses device pointers directly when building blob lists.

2. Run Inference

FAQ

1. Mooncake FAQ

1.1 failed to put/get key

When vLLM reports failed put or get operations, first check whether the error is reported by Mooncake itself.

* If the error is reported by Mooncake:
    * For put failures, check whether the Mooncake log contains NO_AVAILABLE_HANDLE or BatchPut failed ... due to insufficient space. This usually means the remaining space after eviction is not enough for one BatchPut request. Ensure the space left by the eviction policy (for example, the capacity implied by 1 - eviction_ratio) can hold one batch put, or consider increasing the available capacity, increasing eviction headroom, or reducing the batch size.
    * For get failures, check whether the Mooncake log contains lease_expired_before_data_transfer_completed key=... or returns LEASE_EXPIRED. This means the KV object lease expired before the data transfer completed. Increase --default_kv_lease_ttl for mooncake_master as needed, and keep it larger than ASCEND_CONNECT_TIMEOUT and ASCEND_TRANSFER_TIMEOUT.
* If the error is not reported by Mooncake, it is likely an HIXL (ascend_direct) transfer-layer issue. Collect plog files under /root/ascend/log/debug/plog and check whether the issue matches a known HIXL problem.

For common troubleshooting and issue localization guidance for HIXL (ascend_direct), see:
<https://gitcode.com/cann/hixl/wiki/HIXL%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D%E6%89%8B%E5%86%8C.md>

1.2 SSD FAQ

1.2.1 SEGMENT_NOT_FOUND with SSD offload

If client logs show OffloadObjectHeartbeat failed, error code is SEGMENT_NOT_FOUND, Master has unmounted the rank's LOCAL_DISK segment (usually after client_expired when Ping stops refreshing TTL). SSD offload on that rank stops until the segment is registered again.

Typical trigger (with enable_cpu_binding=true): Mooncake starts Ping during init, then vLLM-Ascend bind_cpus() runs migratepages/IRQ binding; the Ping thread is not pinned and can miss beats under the default client_ttl=10.

| Mitigation | Notes |
| :--- | :--- |
| Temporary: raise Master TTL | e.g. mooncake_master ... --client_ttl=120. Tune to your init/warmup window (often 60–120 is enough). Does not fix the root cause. |
| Recovery: upgrade Mooncake | Versions > v0.3.11 (main branch) can remount LOCAL_DISK and rescan metadata after SEGMENT_NOT_FOUND. This recovers after cleanup; it does not prevent expiry or in-flight request failures while metadata is gone. |
| Root fix: Mooncake Ping CPU affinity | Pin the storage Ping thread to a release/isolated CPU (Mooncake-side change). Optional vLLM-Ascend cooperation to pass the release CPU per rank. |

Also restart Master together with vLLM to avoid stale segment_already_exists state when debugging restarts.

1.2.2 Fabric memory size alignment (A3 + ASCEND_ENABLE_USE_FABRIC_MEM=1)

On A3 with fabric memory enabled, each fabric mem allocation must be an integer multiple of 1 GB (1073741824 bytes). Mooncake does not round sizes up automatically.

| Parameter | Config source | Alignment |
| :--- | :--- | :--- |
| global_segment_size | mooncake.json or export MOONCAKE_GLOBAL_SEGMENT_SIZE | Each rank's segment size must be aligned to 1GB (e.g. "1GB", "20GB"). |
| MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES | export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES (only when enable_ssd_offload=true) | Must be aligned to 1GB. Default is 1280 MB (1.25 GB), which is not aligned and is too small for long-context SSD loads — size with Sizing MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES. |

local_buffer_size in mooncake.json is not used under fabric mem (vLLM-Ascend passes 0 to setup()).

Risk if misaligned: adxl MallocMem / aclrtMapMem fails with Invalid_Argument. With SSD offload enabled, a failed MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES allocation can segfault during FileStorage init and abort vLLM startup. Avoid values such as "1280MB", "512MB", or "1.5GB".

Fabric mem quota: Both global_segment_size and MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES are separate fabric mem allocations per rank. Their sizes add up against the HIXL fabric mem limit configured via ASCEND_GLOBAL_RESOURCE_CONFIG (e.g. "fabric_memory.max_capacity":32, unit GB per process — see HIXL docs). Rough budget per rank:



Risk if quota is too low: Some ranks fail with Memory_Allocation_Failure(EL0004) after global_segment_size succeeds but MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES allocation fails. Increase fabric_memory.max_capacity, reduce global_segment_size or MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES, or ensure the node has enough host memory.

Example (add to your vLLM startup script when SSD offload is on):



set ASCEND_GLOBAL_RESOURCE_CONFIG only if fabric mem is too low.



1.2.3 Sizing MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES

When enable_ssd_offload=true, Mooncake allocates a separate per-rank SSD read/write buffer sized by MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES. This buffer is independent of global_segment_size in mooncake.json — increasing the segment does not fix BUFFER_OVERFLOW caused by an undersized MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES.

If the buffer is too small, SSD reads fail with BUFFER_OVERFLOW (error_code=-10) during FileStorage::AllocateBatch, and vLLM may fail when kv_load_failure_policy=fail.

If you encounter BUFFER_OVERFLOW during use, try increasing MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES. Do not set it higher than the Available KV cache memory value shown in vLLM worker logs:



Example:



Use byte literals only (10737418240). 10G / 10GB are ignored and fall back to the 1280 MB default.

<details>
<summary>Notes</summary>

* --max-num-batched-tokens only chunks prefill compute; it does not reduce the memory required by MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES.

</details>

Host memory budget (single node)

MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES is allocated per rank, in addition to global_segment_size:



Ensure free -h available on the host exceeds this sum plus vLLM overhead. MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES does not need to fit inside global_segment_size.

Verify after tuning

1. Startup: each rank logs AlignedClientBufferAllocator: allocated <N> bytes with your configured size.
2. Under load: no BUFFER_OVERFLOW / Failed to get ... keys out of ... error_codes=[-10].
3. If failures persist with a large buffer, check overlapping loads (load_async).

2. Memcache FAQ

For Memcache troubleshooting, see:
<https://gitcode.com/Ascend/memcache/wiki/FAQ.md>

3. DSv4 known issue (temporary)

For the temporary DSv4 known issue, see:
<https://github.com/vllm-project/vllm-ascend/issues/9975>
