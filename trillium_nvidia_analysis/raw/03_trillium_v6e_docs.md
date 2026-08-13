# SOURCE: Google Cloud Documentation — TPU v6e (Trillium)
URL: https://docs.cloud.google.com/tpu/docs/v6e
Retrieved: 2026-07-07
Type: PRIMARY (vendor official docs), live as of 2026-07-07

NOTE: "On all technical surfaces, such as the API and logs, and throughout this document, Trillium will be referred to as v6e."

## SYSTEM ARCHITECTURE
- Each v6e chip contains ONE TensorCore.
- Each TensorCore has **2 matrix-multiply units (MXU)**, a vector unit, and a scalar unit.
- Optimized for transformer, text-to-image, and CNN training, fine-tuning, and serving.

## KEY SPECIFICATIONS (authoritative)
| Specification | Value |
|---|---|
| Peak compute per chip (BF16) | 918 TFLOPs |
| Peak compute per chip (Int8) | 1836 TOPs |
| HBM capacity per chip | 32 GB |
| HBM bandwidth per chip | 1638 GBps |
| Bidirectional ICI bandwidth (per chip) | 800 GBps |
| ICI ports per chip | 4 |
| DRAM per host | 1536 GiB |
| Chips per host | 8 |
| TPU Pod size | 256 chips |
| Interconnect topology | 2D torus |
| BF16 peak compute per Pod | 234.9 PFLOPs |
| All-reduce bandwidth per Pod | 102.4 TB/s |
| Bisection bandwidth per Pod | 3.2 TB/s |
| Per-host NIC | 4 × 200 Gbps |
| DCN bandwidth per Pod | 25.6 Tbps |
| Special features | SparseCore |

## SUPPORTED 2D SLICE TOPOLOGIES
1x1 (1 chip), 2x2 (4), 2x4 (8 single/multi-host), 4x4 (16), 4x8 (32), 8x8 (64), 8x16 (128), 16x16 (256 = full pod). Machine type ct6e-standard-4t/8t. 256 chips = 32 hosts = 64 VMs.

## VM TYPES
- 1-chip VM: 44 vCPU, 176 GB RAM, 1 NUMA
- 4-chip VM: 180 vCPU, 720 GB RAM, 1 NUMA
- 8-chip VM (v6e-8): 360 vCPU, 1440 GB RAM, 2 NUMA — optimized for inference (all 8 chips in single serving workload)
- Multi-host inference via Pathways.

## DISCREPANCY TO INVESTIGATE (analyst-relevant)
- Launch blog said "2X ICI bandwidth vs v5e". v5e ICI was 800 Gbps (100 Gbps/lane × 4 links × 2 directions). Docs say v6e ICI = 800 GB/s bidirectional = 6.4 Tbps. Secondary SERP source claimed "3,200 Gbps per chip". Need to reconcile units (GBps vs Gbps) — likely confusion between bytes and bits.
- MXU count: docs = 2 MXU per TensorCore. Some secondary sources cite 4. Likely v6e (cloud SKU, 2 MXU) vs the originally-announced/Hot Chips Trillium die (which may pack more). The Next Platform article to clarify.

## DERIVED NUMBERS
- Per pod: 256 × 918 TFLOPs = 234.9 PFLOPs BF16 ✓ (matches doc)
- Per chip HBM: 32 GB × 256 = 8.2 TB HBM per pod
- ICI bisection 3.2 TB/s for 256-chip 2D torus.
