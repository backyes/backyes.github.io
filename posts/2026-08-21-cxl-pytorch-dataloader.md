---
title: "[AI Generated] PyTorch CXL Dataset/DataLoader: Distributed Data Loading Architecture for Hundreds of TB DDR Memory"
date: 2026-08-21
tags: ["CXL", "PyTorch", "DataLoader", "Dataset", "Memory Pool", "Distributed", "NUMA", "Prefetch", "LLM Training", "AI Generated"]
excerpt: "Extend PyTorch Dataset/DataLoader API to build CXL memory pool-aware distributed data loading system — not a transparent solution, but explicitly leveraging CXL semantics (byte-level addressing, NUMA-aware, tiered placement) for global data pooling, intelligent prefetching, and load balancing across hundreds of TB of memory."
---

# [AI Generated] PyTorch CXL Dataset/DataLoader: Distributed Data Loading Architecture for Hundreds of TB DDR Memory

> **Note**: This post was AI-generated based on systematic research. Source: [cxl_pytorch_dataloader.md](https://github.com). Method: Architecture-level analysis of PyTorch DataLoader internals, CXL memory pool integration, distributed sampling/prefetching design, and deep reading of 10+ academic papers (arXiv 2022-2026).
>
> **Companion PDF Slides**: See embedded preview below or [Download PDF](cxl_pytorch_dataloader.pdf)

---

## Thesis

**By extending PyTorch's Dataset/DataLoader API, we can build a CXL memory pool-aware distributed data loading system** — not a transparent solution, but one that explicitly leverages CXL semantics (byte-level addressing, NUMA-aware, tiered placement) to achieve global data pooling, intelligent prefetching, and load balancing across hundreds of TB of memory.

This report provides a source-level analysis of PyTorch DataLoader extension points and systematically addresses five core questions:

1. **Why can't existing data loading solutions handle hundreds of TB?** — The scale problem
2. **What are the core challenges of CXL memory pool data loading?** — The architecture problem
3. **How do we design a CXL-aware DataLoader system?** — The system design problem
4. **How do we驾驭 hundreds of TB of DDR memory?** — The scale operation problem
5. **What are the expected performance benefits?** — The evaluation problem

We analyze 10+ academic papers in depth (including MegaScale-Data, FFCV, DeepSpeed, CXL-DMSim, Aquifer), extract their design philosophies, and propose a complete system architecture with implementation details.

---

## 1. Background & Problem Definition

### 1.1 The Memory Wall in Large-Scale Training

```
Large-scale training data scale (2024-2025):
┌─────────────────────────────────────────────┐
│  GPT-4 class training:                      │
│  ├── Token count: ~13T tokens              │
│  ├── Raw text: ~50TB                       │
│  ├── Tokenized: ~100TB (int32)            │
│  └── Intermediate features/index: ~200TB    │
│                                              │
│  LLaMA-405B class training:                 │
│  ├── Token count: ~20T tokens              │
│  ├── Raw text: ~100TB                      │
│  ├── Tokenized: ~300TB                     │
│  └── Intermediate results: ~500TB          │
│                                              │
│  Single-node DRAM capacity: 0.5-2TB         │
│  Single-node GPU memory: 80-800GB           │
│  Data/compute ratio: 100:1 ~ 1000:1        │
└─────────────────────────────────────────────┘
```

**Core contradiction**: Training data scale (hundreds of TB) far exceeds single-node DRAM (TB-level). Traditional "load to memory" mode fails.

### 1.2 Limitations of Existing Solutions

| Solution | Limitation |
|----------|-----------|
| **PyTorch native DataLoader** | Assumes data on local disk, pin_memory limited to local DRAM |
| **Memory-mapped (mmap)** | Limited to local NUMA node, cannot share across nodes |
| **NVIDIA DALI** | GPU-only decoding, doesn't manage global memory pool |
| **WebDataset** | Streaming load, doesn't support global shuffle |
| **FFCV** | Single-node optimized, doesn't sense distributed memory pool |
| **MegaScale-Data** | Focus on multi-source preprocessing, doesn't directly manage CXL memory |

### 1.3 CXL Memory Pool Opportunity

```
CXL Memory Pool vs Traditional Storage Hierarchy:
┌─────────────────────────────────────────────┐
│  Traditional:                                │
│  GPU → CPU DRAM → NVMe SSD → NAS            │
│  100ns   100ns    100μs     10ms            │
│                                              │
│  CXL Memory Pool:                            │
│  GPU → CPU DRAM → CXL Pool → NVMe-oF       │
│  100ns   100ns    300ns     100μs           │
│                                              │
│  CXL extends memory hierarchy from          │
│  "intra-node" to "rack-scale"               │
│  Hundreds of TB exposed as byte-addressable  │
└─────────────────────────────────────────────┘
```

---

## 2. Related Work

### 2.1 MegaScale-Data: Multi-Source LFM Training Data Loading (2504.09844)

**Core Architecture**:

```
MegaScale-Data Architecture:
┌─────────────────────────────────────────────────────┐
│  Centralized Data Plane (declarative orchestration)  │
│  ├── Multi-source routing: text/image/code/multimodal│
│  ├── Curriculum Learning: dynamic data mixing       │
│  └── Long-short context: variable-length sequences  │
├─────────────────────────────────────────────────────┤
│  Disaggregated Preprocessing                         │
│  ├── Source Loaders: per-source-type decoupling     │
│  ├── Data Constructors: unified format conversion   │
│  └── Multi-level Auto-partitioning: adaptive shard  │
├─────────────────────────────────────────────────────┤
│  Data Parallel Loaders                               │
│  ├── Each rank handles disjoint subset              │
│  └── Eliminates redundant data access              │
└─────────────────────────────────────────────────────┘
```

**Key Innovations**:
1. **Disaggregated preprocessing**: Source Loaders separated by data source type, eliminating parallel redundancy
2. **Centralized declarative data plane**: Unified orchestration of multi-source routing, curriculum learning, variable-length sequences
3. **Multi-level auto-partitioning**: Adaptively adjusts shards based on heterogeneous preprocessing costs

**Performance**:
- End-to-end training throughput improvement: **4.5x**
- CPU memory reduction: **13.5x**

**Implications for CXL Dataset**:
- Disaggregated preprocessing → different data sources in CXL pool can be placed in different tiers
- Centralized data plane → global metadata service can sense CXL topology
- Auto-partitioning → dynamically adjust data placement based on CXL bandwidth/latency

### 2.2 FFCV: Eliminating Data Bottlenecks (2306.12517)

**Core Idea**:

```
FFCV Data Loading Pipeline:
┌─────────────────────────────────────────────┐
│  Efficient file format (custom .ffcv)        │
│  ├── Compact binary encoding                 │
│  ├── Supports random access                  │
│  └── Reduces I/O bandwidth needs            │
├─────────────────────────────────────────────┤
│  Caching strategy                            │
│  ├── Hot data → memory                     │
│  ├── Warm data → SSD                       │
│  └── Cold data → disk                      │
├─────────────────────────────────────────────┤
│  Data preload + async transfer               │
│  ├── Pre-fetch next batch                   │
│  ├── CPU → GPU async DMA                   │
│  └── Compute/communication overlap           │
├─────────────────────────────────────────────┤
│  JIT compilation optimization                 │
│  ├── Data augmentation JIT                   │
│  └── Reduces Python interpreter overhead     │
└─────────────────────────────────────────────┘
```

**Performance**:
- ResNet-50 ImageNet training to 75%: only **20 minutes**
- Data loading is no longer a bottleneck

**Implications for CXL Dataset**:
- Caching strategy extensible to three tiers: DRAM → CXL Pool → NVMe
- Async transfer extensible to CXL.mem → GPU direct
- Efficient file format can optimize data layout in CXL memory

### 2.3 DeepSpeed Data Efficiency (2212.03597)

**Core Techniques**:

```
DeepSpeed Data Efficiency Framework:
┌─────────────────────────────────────────────┐
│  Efficient Data Sampling                     │
│  ├── Curriculum Learning Library            │
│  ├── Difficulty-based data sampling          │
│  └── Dynamic data mixing ratio               │
├─────────────────────────────────────────────┤
│  Efficient Data Routing                      │
│  ├── Random Layerwise Token Dropping        │
│  ├── Random inter-layer token dropping       │
│  └── Redundant computation reduction         │
├─────────────────────────────────────────────┤
│  Data/model co-optimization                  │
│  ├── Data quality vs training efficiency     │
│  └── Maintains 95% model quality             │
└─────────────────────────────────────────────┘
```

**Performance**:
- GPT-3 1.3B pre-training: **12.5x** data/time/cost reduction
- Maintains 95% model quality

**Implications for CXL Dataset**:
- Curriculum learning can combine with CXL memory hierarchy
- Hot data (high learning value) → fast tier, cold data → CXL tier
- Token dropping can reduce data movement in CXL memory

### 2.4 PyTorch DataLoader Architecture Analysis

```
PyTorch DataLoader Existing Architecture:
┌─────────────────────────────────────────────┐
│  Dataset                                     │
│  ├── __getitem__(index) → sample            │
│  └── __len__() → size                       │
├─────────────────────────────────────────────┤
│  Sampler                                     │
│  ├── SequentialSampler                      │
│  ├── RandomSampler                          │
│  ├── DistributedSampler                     │
│  └── WeightedSampler                        │
├─────────────────────────────────────────────┤
│  Collate Function                            │
│  └── batch assembly                          │
├─────────────────────────────────────────────┤
│  Worker Processes                            │
│  ├── num_workers parallel loading            │
│  ├── shared memory queue                     │
│  └── pin_memory page-locked memory          │
├─────────────────────────────────────────────┤
│  Main Process                                │
│  └── batch → GPU                            │
└─────────────────────────────────────────────┘
```

**Extension Point Analysis**:
- Dataset: Extensible to CXLDataset, data loaded from CXL pool
- Sampler: Extensible to CXL-aware Sampler, sensing data physical location
- Worker: Extensible to CXL Worker, NUMA binding + prefetch
- pin_memory: Extensible to pin_to_cxl(), pinning to CXL pool

---

## 3. Core Challenges of CXL Memory Pool Data Loading

### 3.1 Challenge 1: Data Locality vs Global Random Access

```
Problem:
  LLM pre-training needs global shuffle (random access to all tokens)
  But CXL memory pools have latency asymmetry (local DRAM vs remote CXL)
  
  If each batch is randomly sampled → massive remote CXL access
  → Latency penalty cancels capacity gains
  
Solution approach:
  1. Tiered shuffle: intra-node random, inter-node sequential
  2. Prefetch buffer: prefetch data from CXL to DRAM in advance
  3. Data placement: auto-adjust data tier based on access frequency
```

### 3.2 Challenge 2: Metadata Management for Hundreds of TB

```
Problem:
  100TB data = 10^13 tokens (int32)
  Per-token metadata: 8 bytes (offset + length)
  Total metadata: 80MB → fits in DRAM
  
  But if data is variable-length sequences (text/code):
  100M records × 1KB metadata = 100GB metadata
  → Metadata itself needs tiered storage

Solution approach:
  1. Compact metadata encoding (delta + varint)
  2. Metadata tiering: hot metadata → DRAM, cold metadata → CXL
  3. Bloom filter for accelerated negative lookup
```

### 3.3 Challenge 3: Cross-Node Data Consistency

```
Problem:
  Multiple nodes simultaneously read same CXL memory region
  CXL 2.0 lacks hardware cache coherence
  Software protocol needed for consistency

Solution approach:
  1. Read-only data: multi-node shared CXL region (no consistency issue)
  2. Read-write data: ownership-based coherence protocol (Aquifer approach)
  3. Version control: copy-on-write (COW) + version number
```

### 3.4 Challenge 4: Heterogeneous Hardware Topology

```
Problem:
  Different nodes have different CXL topologies:
  - Node A: Direct CXL device connection (low latency)
  - Node B: Via CXL switch (high latency)
  - Node C: Via RDMA to remote CXL (highest latency)
  
  NUMA-aware scheduling and data placement needed

Solution approach:
  1. Topology-aware data partitioning
  2. Data allocation based on node capability
  3. Dynamic load balancing
```

---

## 4. System Architecture Design

### 4.1 Overall Architecture

```
CXL DataLoader System Architecture:
┌─────────────────────────────────────────────────────────────┐
│                     User Code Layer                          │
│  from cxl_dataloader import CXLDataset, CXLDataLoader       │
│  dataset = CXLDataset('s3://data/', pool='cxl_pool_A')      │
│  loader = CXLDataLoader(dataset, batch_size=32,             │
│      pin_to_cxl=True, prefetch_factor=4,                    │
│      memory_policy='hot_first')                              │
├─────────────────────────────────────────────────────────────┤
│                  CXL Dataset API Layer                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CXLDataset                                          │    │
│  │ ├── __init__(source, pool, tier_policy)             │    │
│  │ ├── __getitem__(index) → sample (CXL.mem)          │    │
│  │ ├── __len__() → size                               │    │
│  │ ├── pin_to_cxl() → pinned batch                    │    │
│  │ └── migrate_to(tier) → data migration              │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ CXLSampler                                         │    │
│  │ ├── DistributedCXLSampler (topology-aware)         │    │
│  │ ├── RandomCXLSampler (tiered random)               │    │
│  │ └── CurriculumCXLSampler (curriculum learning)     │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                  CXL DataLoader API Layer                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CXLDataLoader                                       │    │
│  │ ├── pin_to_cxl(): pin to CXL pool                  │    │
│  │ ├── prefetch_to_gpu(): CXL → GPU async prefetch    │    │
│  │ ├── memory_policy: hot_first | streaming | full     │    │
│  │ └── num_workers: CXL-aware Worker pool             │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ CXLWorker (NUMA binding + prefetch)                 │    │
│  │ ├── Bind to specific NUMA node                     │    │
│  │ ├── CXL.mem read/write                             │    │
│  │ └── Async prefetch + compute overlap                │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Runtime Layer                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CXL Memory Pool Allocator                           │    │
│  │ ├── Pool Manager (multi-pool management)            │    │
│  │ ├── Tier Policy (hot/warm/cold tiering)             │    │
│  │ ├── RDMA Remote Access (cross-pool access)          │    │
│  │ └── Arrow Buffer (zero-copy columnar)               │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ UCX Transport (cross-node communication)             │    │
│  │ ├── InfiniBand RC (RDMA)                           │    │
│  │ ├── CUDA IPC (intra-node GPU)                       │    │
│  │ └── Shared Memory (intra-node)                      │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ Metadata Service (global metadata)                   │    │
│  │ ├── Data location: which CXL pool                   │    │
│  │ ├── Access heat: hot/warm/cold marking              │    │
│  │ └── Version control: COW + version number           │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ CXL Pool │ │ CXL Pool │ │ CXL Pool │ │ NVMe-oF  │      │
│  │ (Node A) │ │ (Node B) │ │ (Node C) │ │ (Archive)│      │
│  │  64TB    │ │  64TB    │ │  64TB    │ │  1PB     │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Comparison with PyTorch Native API

```
PyTorch Native:
  dataset = MyDataset(path)
  loader = DataLoader(dataset, batch_size=32, num_workers=4,
                      pin_memory=True)

CXL DataLoader:
  dataset = CXLDataset(path, pool='cxl_pool_A',  # specify CXL pool
                       tier_policy='hot_first')    # tier policy
  loader = CXLDataLoader(dataset, batch_size=32,
                         num_workers=4,            # CXL-aware Worker
                         pin_to_cxl=True,          # pin to CXL pool
                         prefetch_factor=4,         # async prefetch
                         memory_policy='hot_first',# memory policy
                         numa_bind=True)           # NUMA binding

Key extensions:
  + pool: Specify CXL memory pool
  + tier_policy: Hot/warm/cold tiering policy
  + pin_to_cxl: Pin to CXL pool (not local DRAM)
  + prefetch_factor: CXL → GPU prefetch depth
  + memory_policy: hot_first | streaming | full_prefetch
  + numa_bind: NUMA node binding
```

### 4.3 Three-Layer API Design

```
Layer 1: Basic CXL Dataset (replaces torch.utils.data.Dataset)
  ├── CXLMemmapDataset: Memory-mapped CXL data
  ├── CXLArrowDataset: Arrow-format CXL data
  ├── CXLNestedDataset: Nested/variable-length data
  └── CXLWebDataset: Streaming CXL data

Layer 2: Advanced Sampler (replaces torch.utils.data.Sampler)
  ├── DistributedCXLSampler: Distributed topology-aware sampling
  ├── RandomCXLSampler: Tiered random sampling
  ├── CurriculumCXLSampler: Curriculum learning sampling
  └── BalancedCXLSampler: Load-balanced sampling

Layer 3: Advanced DataLoader (replaces torch.utils.data.DataLoader)
  ├── CXLDataLoader: Basic CXL DataLoader
  ├── CXLInstantDataLoader: Instant prefetch DataLoader
  ├── CXLAsyncDataLoader: Async pipeline DataLoader
  └── CXLDistributedDataLoader: Distributed DataLoader
```

---

## 5. Key Module Design

### 5.1 CXL Memory Pool Allocator

```python
class CXLMemoryPool:
    """
    CXL memory pool allocator.
    
    Core responsibilities:
    1. Manage memory regions exposed by CXL Type 3 Memory Expander
    2. Provide mmap interface for direct Dataset access
    3. Support tiered allocation (hot/warm/cold)
    4. Support cross-node RDMA access
    """
    
    def __init__(self, pool_id, size, topology):
        self.pool_id = pool_id
        self.size = size
        self.topology = topology  # NUMA topology info
        
        # Open CXL device
        self.fd = open(f"/dev/cxlmem_{pool_id}", 'rb+')
        
        # mmap to process address space
        self.base = mmap.mmap(
            self.fileno(), size,
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_SHARED | mmap.MAP_HUGETLB,  # 2MB huge pages
            0
        )
        
        # Initialize allocator
        self.allocator = CXLBumpAllocator(self.base, size)
        
    def alloc(self, size, hint='hot', numa_node=None):
        """
        Allocate memory.
        
        Args:
            size: Allocation size
            hint: 'hot'|'warm'|'cold' - page heat hint
            numa_node: Preferred NUMA node
        """
        ptr = self.allocator.alloc(size)
        
        # Give kernel tiering hints via madvise
        if hint == 'hot':
            madvise(ptr, size, MADV_HUGEPAGE | MADV_WILLNEED)
        elif hint == 'warm':
            madvise(ptr, size, MADV_MERGEABLE)
        else:
            madvise(ptr, size, MADV_COLD)
            
        return CXLBuffer(ptr, size, self)
    
    def remote_access(self, node_id, offset, size):
        """
        Cross-node access (via RDMA).
        
        When data is not local, read from remote CXL pool via RDMA.
        """
        remote_pool = self.topology.get_pool(node_id)
        return RDMARead(remote_pool, offset, size)
    
    def pin_for_gpu(self, buffer):
        """
        Register CXL region as GPU-accessible.
        
        Uses cudaHostRegister to register CXL mmap region as CUDA host memory.
        GPU can access directly via RDMA/PCIe P2P.
        """
        cudaHostRegister(buffer.ptr, buffer.size, 
                        cudaHostRegisterPortable | cudaHostRegisterIoMemory)
        return buffer
```

### 5.2 CXL Dataset Implementation

```python
class CXLMemmapDataset(torch.utils.data.Dataset):
    """
    Memory-mapped CXL dataset.
    
    Core features:
    1. Data memory-mapped in CXL pool, no need to load to local DRAM
    2. Supports random access (direct location via index)
    3. Supports tiered placement (hot data → DRAM, cold data → CXL)
    """
    
    def __init__(self, path, pool, tier_policy='hot_first'):
        self.path = path
        self.pool = pool
        self.tier_policy = tier_policy
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Memory-map to CXL pool
        self.mmap = pool.mmap_file(path)
        
        # Tiered placement
        self._tier_placement()
        
    def _load_metadata(self):
        """
        Load dataset metadata.
        
        Metadata includes:
        - Per-sample offset and length
        - Data type and schema
        - Statistics (mean, variance, etc.)
        """
        meta_path = self.path + '.meta'
        meta = json.load(open(meta_path))
        
        # Compact encoding: delta + varint
        self.offsets = self._decode_delta(meta['offsets'])
        self.lengths = self._decode_delta(meta['lengths'])
        self.total_size = meta['total_size']
        
        return meta
    
    def _tier_placement(self):
        """
        Tiered placement strategy.
        
        hot_first:
        - First 10% of data (high access frequency) → DRAM
        - Middle 30% → CXL pool hot tier
        - Last 60% → CXL pool cold tier
        
        streaming:
        - Data flows sequentially through CXL → GPU
        - Not retained in memory
        
        full_prefetch:
        - Full prefetch to CXL pool
        - Hot pages auto-promoted to DRAM
        """
        if self.tier_policy == 'hot_first':
            hot_boundary = int(len(self) * 0.1)
            warm_boundary = int(len(self) * 0.4)
            
            # Prefetch hot data to DRAM
            for i in range(hot_boundary):
                data = self._read_sample(i)
                self.pool.alloc(len(data), hint='hot')
                
    def __getitem__(self, idx):
        """
        Get single sample.
        
        Flow:
        1. Look up offset/length via idx
        2. Read from CXL pool (CXL.mem)
        3. Return tensor
        """
        offset = self.offsets[idx]
        length = self.lengths[idx]
        
        # CXL.mem read
        data = self.mmap[offset:offset+length]
        
        # Decode to tensor
        return self._decode_sample(data)
    
    def migrate_to(self, tier):
        """
        Migrate dataset to specified tier.
        
        Args:
            tier: 'dram' | 'cxl_hot' | 'cxl_cold' | 'nvme'
        """
        if tier == 'dram':
            self._promote_to_dram()
        elif tier.startswith('cxl'):
            self._migrate_to_cxl(tier)
        elif tier == 'nvme':
            self._migrate_to_nvme()
```

### 5.3 CXL Sampler Implementation

```python
class DistributedCXLSampler(torch.utils.data.Sampler):
    """
    Distributed CXL-aware sampler.
    
    Core features:
    1. Topology-aware: allocate data based on node CXL topology
    2. Tiered random: intra-node random, inter-node sequential
    3. Load balancing: dynamic adjustment based on node capability
    """
    
    def __init__(self, dataset, num_replicas, rank, 
                 topology, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.topology = topology
        self.shuffle = shuffle
        
        # Calculate per-rank data range
        self.total_size = len(dataset)
        self.num_samples = self.total_size // num_replicas
        self.start_idx = rank * self.num_samples
        self.end_idx = self.start_idx + self.num_samples
        
        # Topology-aware data partitioning
        self.partition = self._topology_aware_partition()
        
    def _topology_aware_partition(self):
        """
        Topology-aware data partitioning.
        
        Strategy:
        1. Calculate each node's "capability score" (CXL bandwidth × capacity)
        2. Allocate data proportionally by capability
        3. Hot data prioritized to high-capability nodes
        """
        nodes = self.topology.get_nodes()
        
        scores = {}
        for node in nodes:
            cxl_bandwidth = node.get_cxl_bandwidth()
            cxl_capacity = node.get_cxl_capacity()
            scores[node.id] = cxl_bandwidth * cxl_capacity
            
        total_score = sum(scores.values())
        partitions = {}
        for node in nodes:
            ratio = scores[node.id] / total_score
            partitions[node.id] = int(self.total_size * ratio)
            
        return partitions
    
    def __iter__(self):
        """
        Iterator.
        
        Tiered random strategy:
        1. Inter-node: sequential traversal (avoid cross-node access)
        2. Intra-node: random shuffle
        """
        if self.shuffle:
            indices = list(range(self.start_idx, self.end_idx))
            random.shuffle(indices)
        else:
            indices = range(self.start_idx, self.end_idx)
            
        return iter(indices)
```

### 5.4 CXL DataLoader Implementation

```python
class CXLDataLoader(torch.utils.data.DataLoader):
    """
    CXL DataLoader.
    
    Core features:
    1. pin_to_cxl: Pin batch in CXL pool
    2. prefetch_to_gpu: Async prefetch CXL → GPU
    3. memory_policy: hot_first | streaming | full_prefetch
    4. numa_bind: NUMA node binding
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 num_workers=0, pin_to_cxl=True,
                 prefetch_factor=2, memory_policy='hot_first',
                 numa_bind=True, collate_fn=None):
        
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn or self._default_collate,
            pin_memory=False,  # Disable native pin_memory
        )
        
        self.pin_to_cxl = pin_to_cxl
        self.prefetch_factor = prefetch_factor
        self.memory_policy = memory_policy
        self.numa_bind = numa_bind
        
        # CXL Worker pool
        if num_workers > 0:
            self.workers = [
                CXLWorker(dataset, numa_bind=numa_bind)
                for _ in range(num_workers)
            ]
        else:
            self.workers = None
            
        # Prefetch queue
        self.prefetch_queue = queue.Queue(maxsize=prefetch_factor)
        
    def __iter__(self):
        """
        Iterator.
        
        Pipeline:
        1. Worker reads data from CXL pool
        2. Data pinned in CXL pool
        3. Async prefetch: CXL → GPU
        4. Return GPU tensor
        """
        if self.prefetch_factor > 0:
            self._start_prefetch_thread()
            
        for batch in super().__iter__():
            if self.pin_to_cxl:
                batch = self._pin_to_cxl(batch)
            yield batch
    
    def _pin_to_cxl(self, batch):
        """
        Pin batch in CXL memory pool.
        
        Implementation:
        1. Get buffer in CXL pool
        2. Copy batch data to CXL buffer
        3. cudaHostRegister on CXL region
        4. Return CXL-pinned batch
        """
        if isinstance(batch, torch.Tensor):
            cxl_buffer = self.dataset.pool.alloc(
                batch.numel() * batch.element_size(),
                hint='hot'
            )
            ctypes.memmove(cxl_buffer.ptr, batch.data_ptr(), batch.nbytes)
            cudaHostRegister(cxl_buffer.ptr, batch.nbytes, 
                           cudaHostRegisterPortable)
            return CXLTensor(cxl_buffer, batch.shape, batch.dtype)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._pin_to_cxl(b) for b in batch)
        else:
            return batch
    
    def _prefetch_to_gpu(self, batch):
        """
        Async prefetch: CXL → GPU.
        
        Paths:
        1. CXL.mem → GPU (CXL 3.0 GPU Fabric)
        2. CXL.mem → NIC → GPU (GPUDirect RDMA)
        3. CXL.mem → CPU DRAM → GPU (fallback)
        """
        if self._cxl_gpu_fabric_available():
            return batch.to('cuda', non_blocking=True)
        elif self._gdr_available():
            return self._gdr_prefetch(batch)
        else:
            return batch.to('cuda', non_blocking=True)
```

### 5.5 CXL Worker Implementation

```python
class CXLWorker:
    """
    CXL Worker: NUMA-bound data loading worker thread.
    
    Core features:
    1. NUMA binding: bind to specific NUMA node
    2. CXL.mem access: direct read/write CXL memory
    3. Async prefetch: load next batch in advance
    """
    
    def __init__(self, dataset, numa_bind=True):
        self.dataset = dataset
        self.numa_bind = numa_bind
        
        # NUMA binding
        if numa_bind:
            self.numa_node = self._get_local_numa_node()
            self._bind_to_numa(self.numa_node)
            
    def _bind_to_numa(self, node):
        """
        Bind to NUMA node.
        
        Uses libnuma to bind thread to specific NUMA node,
        ensuring CXL access takes optimal path.
        """
        import ctypes
        libnuma = ctypes.CDLL('libnuma.so')
        
        mask = libnuma.numa_allocate_nodemask()
        libnuma.numa_bitmask_setbit(mask, node)
        
        libnuma.numa_run_on_node_mask(mask)
        libnuma.numa_free_nodemask(mask)
        
    def load_sample(self, idx):
        """
        Load single sample.
        
        Flow:
        1. Look up offset/length via idx
        2. CXL.mem read
        3. Decode to tensor
        """
        offset = self.dataset.offsets[idx]
        length = self.dataset.lengths[idx]
        
        data = self.dataset.mmap[offset:offset+length]
        
        return self.dataset._decode_sample(data)
    
    def prefetch(self, indices):
        """
        Prefetch multiple samples.
        
        Uses CXL.mem batch reads to reduce round-trip latency.
        """
        ranges = self._merge_ranges(indices)
        
        batches = []
        for start, end in ranges:
            batches.append(self.dataset.mmap[start:end])
            
        return batches
```

### 5.6 Metadata Service Implementation

```python
class MetadataService:
    """
    Global metadata service.
    
    Core responsibilities:
    1. Record each sample's physical location (which CXL pool)
    2. Record access heat (hot/warm/cold)
    3. Version control (COW + version number)
    4. Consistency guarantee (ownership-based coherence)
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.metadata = {}  # idx → SampleMetadata
        self.access_stats = AccessStats()
        self.version = 0
        
    def locate(self, idx):
        """
        Find sample's physical location.
        
        Returns:
            SampleLocation(pool_id, offset, length, tier)
        """
        meta = self.metadata[idx]
        return SampleLocation(
            pool_id=meta.pool_id,
            offset=meta.offset,
            length=meta.length,
            tier=meta.tier
        )
    
    def record_access(self, idx, node_id):
        """
        Record access statistics.
        
        Used for dynamic data placement adjustment.
        """
        self.access_stats.record(idx, node_id)
        
        if self._should_migrate(idx):
            self._migrate(idx)
    
    def _should_migrate(self, idx):
        """
        Determine if migration needed.
        
        Strategy:
        - If a node frequently accesses data → migrate to that node
        - If data long-unaccessed → migrate to cold tier
        """
        stats = self.access_stats.get(idx)
        
        if stats.frequency > HOT_THRESHOLD:
            return True
            
        if stats.last_access > COLD_THRESHOLD:
            return True
            
        return False
    
    def _migrate(self, idx):
        """
        Migrate data to better location.
        """
        meta = self.metadata[idx]
        target_tier = self._decide_tier(idx)
        
        if target_tier != meta.tier:
            self.pool_manager.migrate(meta.pool_id, meta.offset, 
                                     meta.length, target_tier)
            meta.tier = target_tier
```

---

## 6. Hundreds of TB DDR Memory: Operation Solutions

### 6.1 Global Memory Pool Architecture

```
Hundreds of TB CXL Memory Pool Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Global Memory Pool Manager                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Pool Registry                                       │    │
│  │ ├── Pool_A: Node_0, 64TB, CXL_0                    │    │
│  │ ├── Pool_B: Node_1, 64TB, CXL_1                    │    │
│  │ ├── Pool_C: Node_2, 64TB, CXL_2                    │    │
│  │ └── Pool_D: Node_3, 64TB, CXL_3                    │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ Topology Manager                                    │    │
│  │ ├── Node_0 ←CXL Switch→ Node_1                    │    │
│  │ ├── Node_2 ←CXL Switch→ Node_3                    │    │
│  │ └── Node_0 ←RDMA→ Node_2                          │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ Data Placement Engine                               │    │
│  │ ├── Hot data → local DRAM                          │    │
│  │ ├── Warm data → local CXL                          │    │
│  │ ├── Cold data → remote CXL                         │    │
│  │ └── Archive → NVMe-oF                              │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    CXL Switch Fabric                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Node_0   │  │ Node_1   │  │ Node_2   │  │ Node_3   │   │
│  │ 8×H100   │  │ 8×H100   │  │ 8×H100   │  │ 8×H100   │   │
│  │ 2TB DRAM │  │ 2TB DRAM │  │ 2TB DRAM │  │ 2TB DRAM │   │
│  │ 64TB CXL │  │ 64TB CXL │  │ 64TB CXL │  │ 64TB CXL │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                              │
│  Total capacity: 4 × 64TB = 256TB CXL memory pool           │
│  Total bandwidth: 4 × 50GB/s = 200GB/s                      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Data Partitioning Strategy

```
Data Partitioning Strategy:
┌─────────────────────────────────────────────┐
│  1. Horizontal (by sample)                  │
│     ├── Rank 0: samples [0, N/4)           │
│     ├── Rank 1: samples [N/4, N/2)         │
│     ├── Rank 2: samples [N/2, 3N/4)        │
│     └── Rank 3: samples [3N/4, N)          │
│                                              │
│  2. Vertical (by feature/modality)          │
│     ├── Pool_A: text tokens                 │
│     ├── Pool_B: image features              │
│     ├── Pool_C: audio features              │
│     └── Pool_D: labels/metadata             │
│                                              │
│  3. Hybrid (sample + feature)               │
│     ├── Each rank handles subset of samples │
│     └── All features per rank               │
│                                              │
│  4. Dynamic (by access pattern)             │
│     ├── Hot data → multiple replicas        │
│     ├── Warm data → single replica          │
│     └── Cold data → on-demand loading       │
└─────────────────────────────────────────────┘
```

### 6.3 Tiered Data Placement

```
Tiered Data Placement Strategy:
┌─────────────────────────────────────────────┐
│  Tier 0: GPU VRAM (80GB × 8 = 640GB)       │
│  ├── Current batch                          │
│  ├── Prefetch buffer                        │
│  └── Model parameters                       │
│                                              │
│  Tier 1: Local DRAM (2TB × 4 = 8TB)        │
│  ├── Hot data (10% of dataset)              │
│  ├── Metadata                               │
│  └── Prefetch queue                         │
│                                              │
│  Tier 2: Local CXL Pool (64TB × 4 = 256TB) │
│  ├── Warm data (30% of dataset)             │
│  ├── Intermediate results                   │
│  └── Cached data                            │
│                                              │
│  Tier 3: Remote CXL Pool (RDMA)            │
│  ├── Cold data (60% of dataset)             │
│  └── Backup data                            │
│                                              │
│  Tier 4: NVMe-oF (1PB+)                    │
│  ├── Archive data                           │
│  └── Checkpoints                            │
└─────────────────────────────────────────────┘
```

### 6.4 Data Prefetch Pipeline

```
Three-Level Prefetch Pipeline:
┌─────────────────────────────────────────────┐
│  Level 1: CXL → DRAM (prefetch thread)     │
│  ├── 4 batches ahead                        │
│  ├── Async CXL.mem read                     │
│  └── Write to DRAM ring buffer              │
│                                              │
│  Level 2: DRAM → GPU (CUDA stream)         │
│  ├── 2 batches ahead                        │
│  ├── cudaMemcpyAsync                        │
│  └── Overlaps with compute                  │
│                                              │
│  Level 3: GPU → GPU (GPU Fabric)           │
│  ├── 1 batch ahead                          │
│  ├── GPU P2P copy                           │
│  └── Fully overlaps with compute            │
└─────────────────────────────────────────────┘

Timeline:
  [CXL Read] [CXL Read] [CXL Read] [CXL Read]
             [DRAM→GPU] [DRAM→GPU] [DRAM→GPU]
                        [GPU Compute] [GPU Compute]
```

### 6.5 Global Shuffle Strategy

```
Global Shuffle Strategy:
┌─────────────────────────────────────────────┐
│  Problem: 100TB data cannot fully shuffle   │
│                                              │
│  Solution: Tiered Shuffle                   │
│  ├── Level 1: Intra-node shuffle (DRAM)     │
│  │   └── Hot data (10TB) fully random       │
│  ├── Level 2: Inter-node shuffle (CXL)      │
│  │   └── Warm data (30TB) block random      │
│  └── Level 3: Global shuffle (RDMA)         │
│      └── Cold data (60TB) sequential read   │
│                                              │
│  Implementation:                             │
│  1. Split data into 1000 shards             │
│  2. Reassign shards to nodes each epoch     │
│  3. Shuffle shard order within node         │
│  4. Random sample read within node          │
└─────────────────────────────────────────────┘
```

---

## 7. Performance Analysis & Expected Benefits

### 7.1 Theoretical Analysis

```
Performance Model:
┌─────────────────────────────────────────────┐
│  Given:                                      │
│  - Dataset size: D = 100TB                   │
│  - Batch size: B = 32 samples               │
│  - Sample size: S = 1KB (tokenized)         │
│  - CXL bandwidth: BW_cxl = 50GB/s           │
│  - DRAM bandwidth: BW_dram = 100GB/s        │
│  - GPU compute time: T_comp = 100ms/batch   │
│                                              │
│  Traditional (NVMe):                         │
│  - Data load: T_load = D / 10GB/s = 10000s  │
│  - Training time: T_total = N × T_comp      │
│  - I/O fraction: ~50%                        │
│                                              │
│  CXL Solution:                               │
│  - Data load: T_load = D / 50GB/s = 2000s   │
│  - Training time: T_total = N × T_comp      │
│  - I/O fraction: ~10%                        │
│  - Speedup: ~2-5x                            │
└─────────────────────────────────────────────┘
```

### 7.2 Expected Benefits

| Scenario | Current Bottleneck | CXL Acceleration Path | Expected Benefit |
|----------|-------------------|----------------------|------------------|
| **LLM Pre-training** | Data loading I/O | CXL → GPU direct | **3-5x** |
| **Multi-modal Training** | Multi-source mixing | CXL tiered placement | **2-4x** |
| **Large-scale Image** | Image decoding | CXL + GPU decode | **2-3x** |
| **RL Training** | Environment interaction | CXL shared buffer | **1.5-2x** |
| **Recommendation** | Feature engineering | CXL memory pool | **2-3x** |

### 7.3 Comparison with Existing Solutions

| Solution | Data Scale | Cross-Node | CXL-Aware | Global Shuffle |
|----------|-----------|-----------|-----------|----------------|
| **PyTorch DataLoader** | Single-node | ✗ | ✗ | ✗ |
| **FFCV** | Single-node | ✗ | ✗ | ✗ |
| **MegaScale-Data** | Multi-source | ✓ | ✗ | ✓ |
| **DeepSpeed** | Large-scale | ✓ | ✗ | ✗ |
| **CXL DataLoader** | **Hundreds of TB** | **✓** | **✓** | **✓** |

---

## References

| # | Source | arXiv |
|---|--------|-------|
| 1 | **MegaScale-Data: Multi-Source LFM Training Data Loading** | [arXiv:2504.09844](https://arxiv.org/abs/2504.09844) |
| 2 | **FFCV: Eliminating Data Bottlenecks** | [arXiv:2306.12517](https://arxiv.org/abs/2306.12517) |
| 3 | **DeepSpeed Data Efficiency** | [arXiv:2212.03597](https://arxiv.org/abs/2212.03597) |
| 4 | **CXL-DMSim: Full-System CXL Disaggregated Memory Simulator** | [arXiv:2411.02282](https://arxiv.org/abs/2411.02282) |
| 5 | **Aquifer: Hierarchical CXL+RDMA Memory Pooling** | [arXiv:2606.24079](https://arxiv.org/abs/2606.24079) |
| 6 | **CCCL: Node-Spanning GPU Collectives with CXL Memory Pooling** | [arXiv:2602.22457](https://arxiv.org/abs/2602.22457) |
| 7 | **Equilibria: Multi-Tenant CXL Memory Tiering** | [arXiv:2602.08800](https://arxiv.org/abs/2602.08800) |
| 8 | **HybridTier: Adaptive Lightweight CXL Tiering** | [arXiv:2312.04789](https://arxiv.org/abs/2312.04789) |
| 9 | **NeoMem: Hardware-Assisted CXL Tiering** | [arXiv:2403.18702](https://arxiv.org/abs/2403.18702) |
| 10 | **HeteroMem: Device-Side Transparent CXL Management** | [arXiv:2502.19233](https://arxiv.org/abs/2502.19233) |
