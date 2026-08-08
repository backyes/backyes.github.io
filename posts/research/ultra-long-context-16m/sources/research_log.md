# DeepSeek V4 KVCache Offloading Research Log

## Date: 2026-08-08

## Sources Accessed

### Primary Sources (Successfully Downloaded)
1. **Official DeepSeek V4 Technical Report** (arXiv:2606.19348)
   - Title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
   - Authors: DeepSeek-AI team
   - Downloaded: HTML (794KB) + PDF (4.7MB)
   - URL: https://arxiv.org/abs/2606.19348

2. **FlashMemory-DeepSeek-V4 Paper** (arXiv:2606.09079)
   - Title: "FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention"
   - Authors: Yan Wang et al. (Tencent/libertywing)
   - Downloaded: HTML (157KB) + PDF (661KB)
   - URL: https://arxiv.org/abs/2606.09079

3. **FlashMemory GitHub README**
   - URL: https://github.com/libertywing/FlashMemory-Deepseek-V4
   - Downloaded: README.md (324 lines)
   - Key data: Performance tables, architecture details, bandwidth constraints

### Search Queries Performed
- arxiv search: "DeepSeek V4" → 83 results
- arxiv search: "DeepSeek V4 technical report" → found 2606.19348
- arxiv search: "DeepSeek V4 KV cache offloading" → found 2606.09079 (FlashMemory)
- arxiv search: "DeepSeek V4 on-disk" → confirmed 2606.19348
- Google/Bing search: "20GB 512K KVCache offload bandwidth" → no direct match found

### Key Findings
- The "~20GB KVCache with 512K historical sequences" claim does NOT appear verbatim in any paper
- The official DeepSeek V4 report mentions "On-Disk KV Cache Storage" (Section 3.5.2)
- FlashMemory paper provides concrete numbers: 3.73 GB full KV at 1M context
- The 20GB number likely refers to aggregate batch offload, not single sequence

## Research Process
1. Created research directory
2. Searched arxiv for DeepSeek V4 papers
3. Found and downloaded official technical report (2606.19348)
4. Found and downloaded FlashMemory paper (2606.09079)
5. Extracted "On-Disk KV Cache Storage" section from official report
6. Extracted KV Cache Structure details
7. Downloaded FlashMemory README with concrete performance numbers
8. Performed KVCache size calculations
9. Verified user's bandwidth scaling calculation
10. Searched for source of specific "20GB/512K" claim (not found in papers)

## Key Numbers Verified
- DeepSeek-V4-Flash: 284B params, 13B activated, 43 layers
- KV cache per token: ~3.73 KB
- Full KV at 1M context: 3.73 GB
- Full KV at 512K context: ~1.91 GB
- FlashMemory GPU KV at 1M: 0.37 GB (90% offloaded to CPU)
- Throughput gain: 2.8× at 1M context
- Concurrency gain: 2.7× at 1M context
