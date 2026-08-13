#!/usr/bin/env bash
# download_papers.sh — fetch PDF papers from arxiv, rate-limited friendly
# Usage: ./download_papers.sh
# Dir layout:
#   papers/<arxiv_id>.pdf
#   papers/metadata.tsv (id | title | url)

set -euo pipefail
mkdir -p papers
cd papers

META="metadata.tsv"
echo -e "arxiv_id\turl\ttitle" > "$META"

# ID<TAB<TITLE pairs (preserved from report)
declare -a PAPERS=(
"2401.09670\tDistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving"
"2407.00079\tMooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
"2412.12488\tA System for Microserving of LLMs"
"2501.14743\tKVDirect: Distributed Disaggregated LLM Inference"
"2510.09665\tLMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference"
"2510.13223\tBanaServe: Unified KV Cache and Dynamic Module Migration"
"2511.20982\tDOPD: A Dynamic PD-Disaggregation Architecture"
"2512.03416\tTokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving"
"2512.18194\tTraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale"
"2601.11822\tRAPID-Serve: Resource-efficient and Accelerated P/D Intra-GPU Disaggregation"
"2602.18755\tDualScale: Energy-Efficient Disaggregated LLM Serving"
"2602.21548\tDualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference"
"2603.13358\tNot All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving"
"2603.17456\tMulti-stage Flow Scheduling for LLM Serving"
"2605.01708\tSplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving"
"2605.16637\tHexAGenT: Efficient Agentic LLM Serving via Workflow- and Heterogeneity-Aware Scheduling"
"2605.22850\tObjectCache: Layerwise Object-Storage Retrieval for KV Cache Reuse"
"2606.01839\tObservation, Not Prediction: Conversation-Level Disaggregated Scheduling"
"2606.03910\tNetKV: Network-Aware Decode Instance Selection for Disaggregated LLM Inference"
"2606.07684\tSemantic Cache Distillation: Efficient State Transfer via Reuse and Selective Patching"
"2606.08635\tSpectrumKV: Per-Token Mixed-Precision KV Cache Transfer"
"2606.24506\tCrossPool: Efficient Multi-LLM Serving for Cold MoE Models"
"2606.29986\tHBM Is Not All You Need: Efficient Disaggregated LLM Serving across Memory-heterogeneous Accelerators"
"2607.01617\t3DLS: A 3D Logic-Stacked Architecture for Disaggregated LLM Serving"
"2607.01831\tLynx: Progressive Speculative Quantization for accelerating KV Transfer"
"2607.02043\tTowards Load-Aware Prefill Deflection for Disaggregated LLM Serving"
"2604.15039\tPrefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter"
)

UA="Mozilla/5.0 (ResearchBot/1.0; research-use)"
DELAY=8   # seconds between requests (arxiv rate-limit friendly)
RETRY=3

fetch_one() {
  local id="$1" title="$2"
  local pdf="${id}.pdf"
  local url="https://arxiv.org/pdf/${id}.pdf"

  if [[ -s "$pdf" ]]; then
    echo "SKIP  $id  (already exists, $(stat -f%z "$pdf" 2>/dev/null || stat -c%s "$pdf" 2>/dev/null) bytes)"
    return 0
  fi

  for attempt in $(seq 1 $RETRY); do
    local http
    http=$(curl -s -o "$pdf" -w "%{http_code}" -A "$UA" -L --max-time 60 "$url" 2>/dev/null || echo "000")
    if [[ "$http" == "200" && -s "$pdf" ]]; then
      local size
      size=$(stat -f%z "$pdf" 2>/dev/null || stat -c%s "$pdf" 2>/dev/null)
      echo "OK    $id  ${size} bytes  $title"
      echo -e "${id}\t${url}\t${title}" >> "$META"
      return 0
    else
      rm -f "$pdf"
      echo "RETRY $id  attempt $attempt (http=$http)"
      sleep "$DELAY"
    fi
  done
  echo "FAIL  $id  $title"
  echo -e "${id}\t${url}\tFAILED: ${title}" >> "$META"
  return 1
}

echo "=== Downloading ${#PAPERS[@]} papers ==="
echo "Rate limit: 1 req / ${DELAY}s"
echo "Start: $(date)"
echo ""

ok=0; skip=0; fail=0
for entry in "${PAPERS[@]}"; do
  id="${entry%%$'\t'*}"
  title="${entry#*$'\t'}"
  if fetch_one "$id" "$title"; then
    if [[ -s "${id}.pdf" ]]; then
      ((ok++)) || true
    else
      ((skip++)) || true
    fi
  else
    ((fail++)) || true
  fi
  sleep "$DELAY"
done

echo ""
echo "=== Summary ==="
echo "OK: $ok   SKIP: $skip   FAIL: $fail"
echo "End: $(date)"
echo "Papers in: $(pwd)"
echo "Metadata: $META"
ls -la *.pdf 2>/dev/null | wc -l
echo "PDF files total"
