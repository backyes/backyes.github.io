# SOURCE: The Next Platform — "Google Shows Off Its Inference Scale And Prowess"
URL: https://www.nextplatform.com/ai/2025/09/17/google-shows-off-its-inference-scale-and-prowess/1642358
Retrieved: 2026-07-07
Type: SECONDARY (expert analyst), AI Infra Summit Sept 2025

## GOOGLE INFERENCE SCALE (cluster-level, the demand driver)
- Google inference token rate trajectory (across all Google products):
  - Apr 2024: 9.7T tokens/month
  - Aug 2024: ~25T/month
  - Dec 2024: ~160T/month
  - Feb 2025: 160T/month
  - Apr 2025: >480T/month (49.5X growth vs Apr 2024)
  - Jun 2025: 980T/month
  - Aug 2025 (est): ~1,460T/month
- MLPerf: a Trillium TPU v6e generates ~800 tokens/sec on Llama 2 70B → ~2.07B tokens/month per Trillium TPU.
- Fleet estimate: ~704,090 Trillium-equivalents in Aug 2025 (very rough).
- Quote: "the rate of inference across all Google products... has gone exponential."

## IRONWOOD (TPU v7) — successor, trend context
- 5X peak performance and 6X HBM memory capacity of Trillium.
- Ironwood cluster via Google's OCS (optical circuit switch): **9,216 Ironwood TPUs, 1.77 PB HBM** on training+inference.
- vs rackscale Nvidia 144 Blackwell chiplets, 20.7 TB HBM ("looks like a joke").
- OCS: dynamic reconfiguration, heals around TPU failures WITHOUT restarting training/inference jobs. ("This latter bit is huge.")
- Full Ironwood system = 144 racks, 9,216 TPU v7e, **36 pods in a 4D torus**. Base pod = 256 TPUs in 3D torus.
- TNP row-counting analysis: 7 racks/row × 16 systems/rack × 4 TPUs = 448 TPUs/row; suspects 3 pods per 2 rows with 1 hot-spare rack/row → full physical system maybe 10,752 TPU v7e across 168 racks/24 rows, 1,536 spares.

## LIQUID COOLING (Google's infra moat)
- "Google has been working on liquid pooling since 2014... now in our fifth generation cooling distribution unit, planning to distribute that spec to Open Compute Project later this year." — Mark Lohmeyer, GM AI & compute infra.
- "as of 2024, we had around a gigawatt of total liquid cooled capacity, which was 70 times more than any other fleet at that point in time. We created this first for TPUs and now we will replicate it for GPUs." (Lohmeyer)

## GOOGLE INFERENCE STACK (cluster software)
- GKE (managed K8s ≈ internal Borg/Omega), **vLLM** at heart (like Nvidia Dynamo).
- **Anywhere Cache** (new flash caching): cuts read latency 70% within-region, 96% cross-region.
- Managed Lustre for feeding data to GPU/TPU clusters.
- **GKE Inference Gateway**: AI-infused load balancing/routing across compute pools — find device that already has needed context in memory. Breaks prefill from decode (like Nvidia "Rubin CPX" long-context GPU).
- **GKE Inference Quickstart** tool (GA).
- Customer results claimed: inference latency down up to 96%, throughput up 40% higher, token cost down up to 30%.
- **Speculative decoding**: boosted Gemini performance, dropped energy ~33X.

## GOOGLE CLOUD GPU FLEET (hybrid)
- Blackwell RTX 6000 Pro (G4), 8-way B200 (A4), 72-way B200 rackscale (A4X).
- GB300 NVL72 (aimed at lowering inference cost) — not on GCP yet.
- Nvidia Dynamo added as option for custom inference stack.
- Quote: "We strongly suspect that Google prefers to use its own inference stack... if that Google inference stack has not been ported to both Nvidia and AMD GPUs, we would be surprised."

## ANALYST FRAMING (critical)
- Hyperscalers "masters of driving scale up and costs down so a new tech can be cheap enough to be widely deployed."
- Google invented GFS (2003), MapReduce (2004), Borg/Omega→Kubernetes (2014), Bigtable (2006), Dremel/BigQuery+Colossus (2010), Spanner (2012), Dataflow (2014), TPUs concurrent with inventing the Transformer.
