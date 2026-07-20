Optimization and Tuning

This guide aims to help users improve vLLM Ascend performance at the system level. It includes OS configuration, library optimization, deployment guide, and so on. Any feedback is welcome.

Preparation

Run the container:



Configure your environment:



Install vLLM and vLLM Ascend:



Please follow the Installation Guide to make sure vLLM and vLLM Ascend are installed correctly.

!!! note

    Make sure your vLLM and vLLM Ascend are installed after your Python configuration is completed, because these packages will build binary files using python in current environment. If you install vLLM and vLLM Ascend before completing section 1.1, the binary files will not use the optimized python.

Optimizations

1. Memory Allocator Optimization

1.1. jemalloc

jemalloc is a memory allocator that improves performance for multi-threaded scenarios and can reduce memory fragmentation. jemalloc uses a local thread memory manager to allocate variables, which can avoid lock competition between threads and can hugely optimize performance.



1.2. Tcmalloc

TCMalloc (Thread Caching Malloc) is a universal memory allocator that improves overall performance while ensuring low latency by introducing a multi-level cache structure, reducing mutex contention and optimizing large object processing flow. Find more details.



2. torch_npu Optimization

Some performance tuning features in torch_npu are controlled by environment variables. Some features and their related environment variables are shown below.

Memory optimization:



or



Scheduling optimization:



3. CANN Optimization

3.1. HCCL Optimization

There are some performance tuning features in HCCL, which are controlled by environment variables.

You can configure HCCL to use "AIV" mode to optimize performance by setting the environment variable shown below. In "AIV" mode, the communication is scheduled by AI vector core directly with RoCE, instead of being scheduled by AI CPU.



Plus, there are more features for performance optimization in specific scenarios, which are shown below.

- HCCL_INTRA_ROCE_ENABLE: Use RDMA link instead of SDMA link between two 8Ps as the mesh interconnect link. Find more details.
- HCCL_RDMA_TC: Use this var to configure traffic class of RDMA NIC. Find more details.
- HCCL_RDMA_SL: Use this var to configure service level of RDMA NIC. Find more details.
- HCCL_BUFFSIZE: Use this var to control the cache size for sharing data between two NPUs. Find more details.

4. Kernel Optimization

This section describes operating system–level optimizations applied on the host machine (bare metal or Kubernetes node) to improve performance stability, latency, and throughput for inference workloads.

!!! note

    These settings must be applied on the host OS and with root privileges. Not inside containers.

4.1

Set CPU Frequency Governor to performance



Purpose

- Forces all CPU cores to run under the performance governor
- Disables dynamic frequency scaling (e.g., ondemand, powersave)

Benefits

- Keeps CPU cores at maximum frequency
- Reduces latency jitter
- Improves predictability for inference workloads

4.2 Disable Swap Usage



Purpose

- Minimizes the kernel’s tendency to swap memory pages to disk

Benefits

- Prevents severe latency spikes caused by swapping
- Improves stability for large in-memory models

Notes

- For inference workloads, swap can introduce second-level latency
- Recommended values are 0 or 1

4.3 Disable Automatic NUMA Balancing



Purpose

- Disables the kernel’s automatic NUMA page migration mechanism

Benefits

- Prevents background memory page migrations
- Reduces unpredictable memory access latency
- Improves performance stability on NUMA systems

Recommended For

- Multi-socket servers
- Ascend / NPU deployments with explicit NUMA binding
- Systems with manually managed CPU and memory affinity

4.4 Increase Scheduler Migration Cost



Purpose

- Increases the cost for the scheduler to migrate tasks between CPU cores

Benefits

- Reduces frequent thread migration
- Improves CPU cache locality
- Lowers latency jitter for inference workloads
  
Parameter Details

- Unit: nanoseconds (ns)
- Typical recommended range: 50000–100000
- Higher values encourage threads to stay on the same CPU core
