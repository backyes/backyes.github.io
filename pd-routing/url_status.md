# URLs accessed and status

| # | URL | Status | Notes |
|---|-----|--------|-------|
| 1 | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/nixl_kv_push_connector.md | ✅ OK | 8500+ chars saved locally |
| 2 | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/nixl_kv_cache_lease.md | ✅ OK | Full design doc captured |
| 3 | https://docs.vllm.ai/en/latest/features/disagg_prefill.html | ❌ 404 / Rerouted | GitHub raw path missing |
| 3b | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/disagg_prefill.md | ✅ OK (curl) | Markdown fetched successfully |
| 4 | https://github.com/ai-dynamo/dynamo | ✅ OK | 7.5K stars, README loaded via Playwright + curl |
| 5 | https://docs.sglang.ai/references/disagg.html | ❌ 404 | Not found |
| 5b | https://raw.githubusercontent.com/sgl-project/sglang/main/docs_new/docs/advanced_features/pd_disaggregation.mdx | ✅ OK (curl) | User-guide |
| 5c | https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/{prefill,decode,nixl/conn,mooncake/conn,decode_hicache_mixin,common/conn}.py | ✅ OK | Source-level PD routing internals |
| 6 | https://github.com/MoonshotAI/Mooncake | ❌ 404 / Gone | Repo deleted/renamed. Fallbacks from Mooncake-rs / vLLM / SGLang integration docs |
| 7 | https://docs.lmcache.ai/ | ✅ OK | Main landing |
| 7b | https://raw.githubusercontent.com/LMCache/LLCache/dev/docs/source/mp/{index,disaggregated_prefill,p2p}.rst | ✅ OK (curl, dev branch) | MP-mode PD internals |
| 7c | https://raw.githubusercontent.com/LMCache/LMCache/dev/docs/source/disaggregated_prefill/nixl/1p1d.rst | ✅ OK | 1P1D example |
