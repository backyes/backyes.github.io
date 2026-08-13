kvcache-ai
/
Mooncake
Public
Notifications
Fork 961
 Star 5.8k
Code
Issues
241
Pull requests
259
Discussions
Actions
Projects
Security and quality
Insights
kvcache-ai/Mooncake
 main
97 Branches
31 Tags
Code
Folders and files
Name	Last commit message	Last commit date

Latest commit
3 people
[TENT] Wire live RDMA bandwidth into admission queue degradation poli…
930cb62
 · 
History
1,578 Commits


.claude-plugin
	
docs: publish built-in skills and add plugin marketplace (#2497)
	


.claude/skills
	
[Doc]: clarify Mooncake Store quick start and Transfer Engine guidance (
	


.devcontainer
	
[CI] change dependency libboost-all-dev to libboost-dev (#2129)
	


.github
	
[CI] Migrate tone_tests to CUDA 13: pull cu130 wheel, adapt sglang/vl…
	


FAST25-release
	
[Doc] clarify FAST25 trace release (#2727)
	


benchmarks
	
[Bench] Enable replay speedup and multi-threading in SSD Benchmarking (…
	


docker
	
[CI] publish master image to Docker Hub via manual workflow (#2678)
	


docs
	
docs: add Kubernetes Deployment Guide for Mooncake Store and Transfer…
	


extern
	
[Build] Upgrade yalantinglibs to 6a0e067d9a43492cf8e4e280b531924fbd72…
	


image
	
[Doc] docs: update README hardware partners table and logos (#2654)
	


monitoring
	
[Store] Add Prometheus and Grafana example (#1335)
	


mooncake-common
	
[Bugfix][TENT] Fix silent TPU data corruption for transfers larger th…
	


mooncake-ep
	
[EP] add DeepEP V2 elastic buffer (#2503)
	


mooncake-integration
	
[Store] feat: expose client metrics HTTP config to Python (#2822)
	


mooncake-p2p-store
	
[Build] Bump golang.org/x/net in /mooncake-p2p-store/src/p2pstore (#2708
	


mooncake-pg
	
[EP]:Stabilize MACA Expert Parallelism P2P Fast Path (#2592)
	


mooncake-rl/examples
	
[RL] Add dummy example of RL training on mooncake store (#810)
	


mooncake-store
	
[Store] Tune Master defaults based on RPC scaling results (#2871)
	


mooncake-transfer-engine
	
[TENT] Wire live RDMA bandwidth into admission queue degradation poli…
	


mooncake-wheel
	
[Store] feat: expose client metrics HTTP config to Python (#2822)
	


scripts
	
[CI] Migrate tone_tests to CUDA 13: pull cu130 wheel, adapt sglang/vl…
	


.clang-format
	
code format & enable code format checking in ci (#677)
	


.dockerignore
	
[Build] update dockerfile and install mooncake from scratch (#1214)
	


.gitignore
	
[Doc] Add agent guidance (#2712)
	


.gitmodules
	
[Build] add yalantinglibs submodule (#1781)
	


.pre-commit-config.yaml
	
[TE] Reject empty RDMA completion resources during context setup (#2892)
	


.typos.toml
	
[EP] add DeepEP V2 elastic buffer (#2503)
	


AGENTS.md
	
[Doc] Add agent guidance (#2712)
	


CLAUDE.md
	
[Doc] Add agent guidance (#2712)
	


CMakeLists.txt
	
[EP]:Stabilize MACA Expert Parallelism P2P Fast Path (#2592)
	


CODE_OF_CONDUCT.md
	
[Chore] Add Contributor Covenant Code of Conduct (#1056)
	


CONTRIBUTING.md
	
[Doc] Add agent guidance (#2712)
	


LICENSE-APACHE
	
Squashed commits related to transfer engine
	


MAINTAINERS.md
	
[Doc] chore: Add Hygon logo to supported hardware and contributors (#…
	


README.md
	
[Doc] Update README with LightX2V deployment details (#2767)
	


dependencies.sh
	
[Store] feat: add optional RFC #1527 KV events publisher on master (#…
	


requirements.txt
	
[Doc]: clarify Mooncake Store quick start and Transfer Engine guidance (
	
Repository files navigation
README
Code of conduct
Contributing
Apache-2.0 license
A KVCache-centric Disaggregated Architecture for LLM Serving
Paper | Slides | Traces | Documentation | Blog | Slack



    


    




Mooncake is the serving platform for  Kimi, a leading LLM service provided by  Moonshot AI. Under real workloads, Mooncake’s innovative architecture enables Kimi to handle 75% more requests while adhering to SLOs.

🔄 Updates
May 7, 2026: 🚀 vLLM officially features Mooncake Store — a deep dive into how Mooncake's distributed KVCache engine supercharges vLLM inference with high-throughput, memory-efficient, cross-instance KV cache sharing!
Apr 29, 2026: SGLang introduces RDMA-based P2P weight transfer for large-scale distributed RL using Mooncake TransferEngine, achieving 7x faster weight updates for the 1T-parameter Kimi-K2 model (53s → 7.2s) with zero-copy RDMA transfer across thousands of GPUs.
Mar 19, 2026: TorchSpec: Speculative Decoding Training at Scale is open sourced, using Mooncake to decouple inference and training via efficient hidden states management.
Mar 5, 2026: LightX2V now supports disaggregated deployment based on Mooncake, enabling encoder/transformer service decoupling with Mooncake Transfer Engine for high-performance cross-device and cross-machine data transfer. Details in blog.
Feb 25, 2026: SGLang merged Encoder Global Cache Manager, introducing a Mooncake-powered global multimodal embedding cache that enables cross-instance sharing of ViT embeddings to avoid redundant GPU computation.
More
🎉 Overview

Mooncake is an infrastructure project for large-scale LLM inference and training. It features a KV cache-centric disaggregated architecture that separates prefill and decode clusters, while leveraging otherwise underutilized CPU, DRAM, and SSD resources in GPU clusters to build a disaggregated KV cache pool.

Mooncake includes a high-performance Transfer Engine for low-latency data movement across heterogeneous networks and accelerators; Mooncake Store for distributed KV cache and model-weight management; and Mooncake EP & PG for elastic MoE serving. Deeply integrated with ecosystems such as SGLang and vLLM, Mooncake helps LLM systems improve cache reuse, reduce serving latency, and scale efficiently across multi-node clusters.

🔥 Show Cases
Transfer Engine (TE)

The core of Mooncake is the Transfer Engine (TE), a high-performance data transfer framework. TE offers a unified interface for batched data movement across diverse storage, network, and accelerator environments. By supporting multiple transport protocols, topology-aware routing, multi-NIC bandwidth aggregation, and automatic failover, TE delivers low-latency, scalable, and robust data transmission for distributed AI workloads. See the Transfer Engine guide for details.

Highlights
Mooncake Store

Mooncake Store is a high-performance distributed key-value cache storage engine designed for LLM inference. Built on the Transfer Engine, it stores and manages reusable KV caches and model weights across inference clusters, with support for efficient object storage, replication, eviction, and high-bandwidth data transfer. See the Mooncake Store guide for details.

Highlights
Mooncake EP and Process Group (PG)

Mooncake EP and Mooncake PG extend Mooncake from high-performance data movement to fault-tolerant distributed execution for large-scale MoE inference. Mooncake EP adapts DeepEP-style expert-parallel dispatch and combine operations with rank activeness awareness, while Mooncake PG provides a PyTorch distributed process-group backend with collective communication primitives that can detect failed ranks, report failures to upper layers, and recover ranks without restarting the entire inference service. See the Mooncake EP & Backend guide for details.

Highlights
Tensor-Centric Ecosystem

Mooncake establishes a full-stack, Tensor-oriented AI infrastructure where Tensors serve as the fundamental data carrier. The ecosystem spans from the Transfer Engine, which accelerates Tensor data movement across heterogeneous storage (DRAM/VRAM/NVMe), to Mooncake Store for distributed management of Tensor objects (e.g., KVCache and model weight), up to the Mooncake Backend enabling Tensor-based elastic distributed computing. This architecture is designed to maximize Tensor processing efficiency for large-scale model inference and training.

SGLang Integration (Guide)

Mooncake is deeply integrated into SGLang as a high-performance communication and storage backend. These integrations enable efficient KV cache transfer in PD-disaggregated serving, scalable multi-level KV caching through HiCache, fault-tolerant expert-parallel inference, high-performance multimodal pipeline data movement, and fast RDMA-based weight synchronization for large-scale RL training. Together, Mooncake and SGLang provide a production-oriented foundation for building elastic, high-throughput, and resource-efficient LLM and multimodal serving systems.

Details
vLLM Integration (Guide)

Mooncake integrates with vLLM to accelerate large language model serving through high-performance KV cache transfer and distributed KV cache storage. The integration supports both disaggregated prefill-decode serving and cross-instance KV cache sharing, helping vLLM deployments reduce TTFT, improve cache reuse, and scale more efficiently across multi-node inference clusters.

Details
🖥️ Supported Hardware

Mooncake supports hardware backends across accelerator vendors, cloud fabrics, and standard datacenter interconnects, as listed below. See the supported protocols and Transfer Engine design docs for details.

					
					
🚀 Getting Started

Install Mooncake using pip. The mooncake-transfer-engine package includes Mooncake Transfer Engine, Mooncake Store, Mooncake EP and PG:

CUDA < 13.0
pip install mooncake-transfer-engine
CUDA >= 13.0
pip install mooncake-transfer-engine-cuda13

In addition to CUDA, Mooncake also supports other accelerator backends, along with flexible installation and deployment options. See the guides below for details:

Quick Start
Build from Source
Deployment Guide
Skills for AI Assistants

Mooncake ships a set of built-in skills under .claude/skills — reusable, task-focused playbooks that an AI coding assistant (such as Claude Code) invokes automatically when your request matches, or that you can run as a slash command.

Details
📦 Open Source Traces and Tools

We open-source anonymized request traces containing request arrival times, input and output token counts, and remapped block hashes. These traces are designed to support reproducible simulation and evaluation of caching behavior while preserving user privacy. The released traces and related details are available in FAST25-release.

Together with the released traces, we also provide two KV cache analysis tools: a KV Cache Size Calculator for calculating cache capacity across popular LLM model families, and a KV Cache Hit Rate Simulator for analyzing KV cache hit rates and planning cache capacity under different workloads and models. These tools help users better understand KV cache storage costs and caching effectiveness when analyzing or reproducing serving workloads. The tools are open-sourced here.

📑 Citation
Please kindly cite our papers if you find the papers or the traces are useful:
@inproceedings{qin2025mooncake,
  author    = {Ruoyu Qin and Zheming Li and Weiran He and Jialei Cui and Feng Ren and Mingxing Zhang and Yongwei Wu and Weimin Zheng and Xinran Xu},
  title     = {Mooncake: Trading More Storage for Less Computation {\textemdash} A {KVCache-centric} Architecture for Serving {LLM} Chatbot},
  booktitle = {23rd USENIX Conference on File and Storage Technologies (FAST 25)},
  year      = {2025},
  isbn      = {978-1-939133-45-8},
  address   = {Santa Clara, CA},
  pages     = {155--170},
  url       = {https://www.usenix.org/conference/fast25/presentation/qin},
  publisher = {USENIX Association},
  month     = {feb},
}
More
About

Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI.

kvcache-ai.github.io/Mooncake/
Topics
reinforcement-learning inference rdma disaggregation llm vllm sglang kvcache trt-llm tokenspeed
Resources
 Readme
License
 Apache-2.0 license
Code of conduct
 Code of conduct
Contributing
 Contributing
 Activity
 Custom properties
Stars
 5.8k stars
Watchers
 46 watching
Forks
 961 forks
Report repository


Releases 27
v0.3.11.post1
Latest
+ 26 releases


Packages
No packages published



Contributors
272
+ 258 contributors


Languages
C++
80.4%
 
Python
11.2%
 
Cuda
2.9%
 
Shell
1.3%
 
CMake
1.3%
 
C
1.2%
 
Other
1.