---
title: "MTP 論 — 多符預測對算力、芯片、互連之結構性影響"
date: 2026-07-31
tags: ["DeepSeek", "MTP", "文言文", "芯片", "算力", "互連"]
excerpt: "以文言譯英文原文，論多符預測（MTP）如何移推理之負載從訪存密集轉向計算密集，及其對芯片、互連、超節點之結構性衝擊。"
---

# MTP 論

## 要旨

**多符預測（Multi-Prediction Prediction, MTP）者，非止推理之微調也，乃重構計算-存儲-通信三角之根本術也。** 傳統自回歸解碼，每符計算甚微，而 KV-Cache 訪存極繁，此訪存密集之典範也。MTP 反是：以同一 KV-Cache 攤派於 k 符之預測，遂使算力強度（FLOPs/Byte）隨 k 線性增長。

> DeepSeek 者，高端之選手也，有定力，守高性價比，擊成本之最高突破口，秉普惠人類之術。

---

## 一、算力芯片：計算與 KV-Cache 之比反轉

傳統解碼，每符一算，KV 一訪。MTP 行，則 k 符共享一 KV，算力 k 倍而訪存不變。

| 指標 | 傳統解碼 | MTP（k 步） |
|---|---|---|
| 每符算力 | 1× | ≈k× |
| KV-Cache 訪存 | 1× | 1×（共享） |
| 算力強度 | 低 | ==隨 k 線性增長== |

**然則 k× 乃理論之上限也**，實得與否，繫於**接受率**（acceptance rate）。DSpark（DeepSeek 與北大，2026）實證：純堆疊之深層預測，接受率隨深度急速衰減（"suffix decay"）。此正所以 DeepSeek 棄"更深 MTP"之徑，改行半自回歸草擬 + 置信度調度驗證之故。生產部署（DeepSeek-V3）僅用 1-2 層輔助預測，loss scaling 0.1，**k=5+ 之"大 MTP"至今乃理論推測，非生產驗證也**。

**業界脈絡**：NVIDIA GPU 迭代，帶寬增長落後於算力增長——此 MTP 所利用之結構缺口也。

| GPU | 架構 | 顯存 | 帶寬 | FP8 稠密 | 帶寬/算力 | 出處 |
|---|---|---|---|---|---|---|
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 989 TFLOPS | 3.4 | [NVIDIA](https://www.nvidia.com/en-us/data-center/h100/) |
| H200 SXM | Hopper | 141 GB HBM3e | 4.8 TB/s | 989 TFLOPS | 4.9 | [NVIDIA](https://www.nvidia.com/en-us/data-center/h200/) |
| B200 | Blackwell | 192 GB HBM3e | 8 TB/s | 2.25 PFLOPS | 3.6 | [NVIDIA](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) |
| B300 (NVL72) | Blackwell | 288 GB HBM3e | 16 TB/s | ~4.5 PFLOPS | 3.6 | [NVIDIA](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) |
| Rubin (R100) | Rubin | HBM4 (待定) | ~36 TB/s (估) | 待定 | 待定 | [NVIDIA](https://www.nvidia.com/en-us/data-center/technologies/rubin/) |

> ⚠️ 規格待 review — 建議讀者點出處鏈接人工核實。

**分歧昭然**：H100→B200，算力增約 2.3×，帶寬僅增約 2.4×。然算術強度上限（每 Byte 之 FLOPs）方為訪存密集推理之真正約束。MTP 直擊此約束：以同一 KV-Cache 攤 k 符計算，等效於將 compute-per-byte 比率乘以 k，**遂使推理負載從訪存密集推入計算密集之域**。

---

## 二、互連：低延遲 vs 高帶寬之分叉

MTP 於互連之影響，非一端也，隨規模而分：

| MTP 規模 | 生產狀態 | 利 | 害 |
|---|---|---|---|
| **小 MTP**（k=1-2） | 已部署（DeepSeek-V3） | 大超節點、低延遲語義（如 LPX 類） | — |
| **深推測**（DSpark 式） | 已部署（2026.06） | 半自回歸草擬 + 置信度驗證 | 純深層堆疊 |
| **"大 MTP"**（k=5+ 純堆疊） | ==僅理論== — 接受率衰減使不經濟 | 高帶寬結構（若得解） | 低延遲機制 |

**析之**：
- **小 MTP（k=1-2）** 乃當前生產現實。此規模下，低延遲互連仍有價值——LPX 類 SRAM 中心芯片於延遲敏感之解碼場景依舊佔優。
- **深推測（DSpark）** 乃 DeepSeek 對接受率衰減之回答：不堆更深 MTP，改行半自回歸草擬 + 置信度調度驗證，得 60-85% 提速。此乃"k>2 增益"之實際生產機制，非 MTP 之簡單堆疊也。
- **"大 MTP"（k=5+ 純堆疊）** 至今純理論，蓋接受率隨深度急速衰減故也。k× 算力之利，僅於接受率得解時方為現實——此正所以 DSpark 棄此徑也。

> **拐點何在？取決於 MTP 之兩級部署規模**——小 MTP 保留低延遲價值，DSpark 式深推測則將價值轉向驗證帶寬。

**業界對比**：NVIDIA Groq 3 LPX 以片上 SRAM 得 150 TB/s 每 LPU——約為 Rubin HBM4 22 TB/s 之 6.8×。然此優勢**僅於帶寬為瓶頸時方有意義**。MTP 轉向計算密集，削弱 HBM 帶寬天花板之間接懲罰，遂弱化 SRAM 中心架構之價值主張。

---

## 三、超節點域：從"攤銷 Weight"到"通信瓶頸"

**推導鏈**：

```
MTP ↑ → 算力強度 ↑ → 單節點 HBM 壓力 ↓
→ 大 EP 之"攤銷 Weight"收益被抑制
→ 瓶頸轉向通信（大塊傳輸）
→ 利小超節點域之高帶寬
```

**大 EP 之本質**：將專家居於多節點，每節點僅載部分 Weight，以攤銷 HBM 帶寬。MTP 瓦解之——若 HBM 帶寬已非約束，則大 EP 之通信開銷遂為不償失。

**啟示**：瓶頸轉至通信，然乃**大塊通信**——利緊湊域（機櫃級）內之高帶寬，非廣域互連也。

---

## 四、專用芯片：SRAM 獨立架構之擠壓

**無 GPU 夥伴之純 SRAM 架構，市場窗口收窄——然 LPX 乃例外，非通則也。**

| 芯片 | 架構 | SRAM | 帶寬 | GPU 夥伴？ | MTP 影響 |
|---|---|---|---|---|---|
| Cerebras WSE-3 | 晶圓級 SRAM | 44 GB | ~21 PB/s | ✗ | 顯存優勢被稀釋 |
| Groq Trillium | 確定性數據流 | 大量片上 | 80 TB/s | ✗ | SRAM 溢價更難支撐 |
| Etched Sohu | 硬接線 Transformer ASIC | 片上權重 | 極高 | ✗ | 計算密集友好 |
| **NVIDIA LPX** (Groq 3) | SRAM 中心 LPU | 500 MB/LPU | 150 TB/s/LPU | ==✓ (Rubin NVL72)== | ==價值保留==——利低延遲分支 |
| NVIDIA B200 | HBM 平衡 GPU | 極少 | 8 TB/s | — | ==更優定位==——算力優先 |

**LPX 之例外**：LPX 乃 Rubin NVL72 之解碼夥伴——LPU 管低延遲解碼，GPU 管 prefill/attention。此正對應 MTP 之小 MTP/低延遲分支（第二節），故 LPX 之價值主張**被 MTP 強化而非侵蝕**。其所號稱 35× 每瓦吞吐量，正因其瞄準 SRAM 優勢依舊有效之延遲敏感域。

**受擠壓者**：無 GPU 夥伴之純 SRAM 架構——Cerebras WSE-3（~21 PB/s 晶圓帶寬）、Groq Trillium（確定性數據流，80 TB/s）——面臨更嚴苛之算計。其設計假設曰：消除片外存儲即最優解。MTP 使計算為瓶頸時，SRAM 帶寬優勢被稀釋，而其面積/功耗劣勢依舊。Cerebras 基準測試示其超 Groq 6× 以上（晶圓級），然此優勢以訪存密集負載衡量——隨負載轉移，差距收窄。

> **純 SRAM 假設——消除片外存儲即足夠——正在瓦解。** 無 GPU 夥伴之架構（Cerebras、Trillium）窗口收窄。有 GPU 夥伴者（LPX）則佔據 MTP 所保留之低延遲分支。

---

## 五、片上介質：群雄並起之機

MTP 轉計算密集，對芯片供應鏈有==去高端化==之效：

- **利國產/替代存儲介質**：降低對 HBM 極致帶寬之依賴，國產 HBM3E、CXL 掛載存儲、乃至先進 DDR 配置皆得以上桌。
- **利中端工藝節點**：計算密集負載少賴 HBM 帶寬（高端工藝之差異化處），多賴原始 FLOPs（4-5nm 可達）。
- **害高端 HBM 帶寬**：HBM3E（SK Hynix 市佔 62%）之壟斷溢價削弱，蓋帶寬已非約束故也。

> **人人皆得入席。** 此結構性轉向，侵蝕 HBM 生態之定價權——數年之趨勢逆轉也。

**數據**：HBM3E 單堆疊 1.2 TB/s；中端 DDR5 單通道約 50 GB/s。24 倍帶寬差距，於計算密集負載意義大減。全球 HBM 市場 2026 年預計 $58B——MTP 不縮此市場，然將壓縮其**溢價**。

---

## 六、算力密度：加速堆疊，再逢新壁

**兩階段動態**：

**階段一（1-2 年）**：訪存牆暫緩，算力密度加速堆疊。Chiplet、3D 封裝、晶圓級集成所面臨之每計算單元帶寬約束減少。

**階段二（2-3 年）**：算力翻倍 → 存儲再成瓶頸 → 觸髮 HBM 新一輪增長週期。

```
訪存牆破 → 密度堆疊 → 算力翻倍
→ 新存儲壓力 → HBM 入新增長週期
**

**業界平行**：此猶如 2022-2024 週期——HBM3（819 GB/s）→ HBM3E（1.2 TB/s）→ HBM4（2.0 TB/s/堆疊）之採用加速，正因 GPU 算力超前存儲帶寬。MTP 壓縮此週期。

---

## 七、"存儲非瓶頸矣"——常見之謬

謂 MTP 消除存儲壓力者，**謬也**。

- MTP 暫減存儲壓力（同 KV-Cache，更多計算）。
- 然算力密度增長更速——應用層迅速吸納釋放之算力。
- **1-2 年後，存儲需求復強**——非 MTP 失效，乃算力增長超前 HBM 帶寬增長故也。

> MTP 不毀訪存牆——**將其推遲至更高算力基線**。

---

## 八、片外介質：DDR 之反直覺利

一結構性可能：**DDR 或反受其利**。

邏輯：MTP 降 HBM 帶寬依賴 → 然模型容量需求日增（更長序列、更大模型）。若 HBM **容量**成瓶頸，DDR 作為容量層遂得更重要。推理系統或從"HBM 獨大"轉向"HBM + DDR"分層存儲。

此非定論，乃**結構性開口**——MTP 改變存儲層級之間邊際替代率。

---

## 九、Token 成本與序列長度：加速演進

| 維度 | 當前（2026 中） | 預測（2027） |
|---|---|---|
| 推理 token 成本 | 基準 | ==2-4× 下降== |
| 主流序列長度 | 256K | ==2M+== |

MTP 以更高算力利用率直接壓低成本。DSpark（2026.06）稱較 MTP-1 基線快 60-85%，SGLang 基準示 MTP 推測解碼得 1.4× 吞吐量。

**長期趨勢不改，唯節奏加快**：
- UB 低延遲 ✓
- 高帶寬結構 ✓
- 大容量 HBM ✓
- 超低延遲互連 ✓
- 超長序列 ✓

> **此等方向不變——唯更快爾。**

---

## 十、路線之爭：唯稀疏可擴展

通向百萬序列者，兩路焉：

| 路線 | 代表 | 存儲成本 | 計算成本 | 多層介質友好 |
|---|---|---|---|---|
| **線性注意力** | Kimi3 (KDA) | 線性（然指數衰減） | 線性 + Full Attn | ✗ |
| **稀疏注意力 (DSA)** | DeepSeek V4, GLM52 | 稀疏可控 | 稀疏可控 | ✓ |

**Kimi3 之根本問題**：線性 + Full Attention 混合，仍意味著==存儲與計算成本之不良擴展==。當前遺忘門機制下，線性注意力之有效存儲容量隨序列長度指數衰減（遺忘門信息損失累積）——"線性"之斷言僅適用於短上下文質量，非百萬 token 之有效保留。Kimi3 保留之 25% Full Attention 於大序列下仍屬平方級成本。

> **於當前遺忘門範式下，唯稀疏注意力可亞線性擴展存儲/成本。** 線性注意力中信息衰減之累積乃門控遞歸之數學性質，非工程差距——可緩解（如更優門設計、混合比例），然無法在不實質上轉為稀疏之情況下根除。

Kimi 去歲末之技術報告亦承認此點：彼等認可後續需融進稀疏路線。Kimi3 之產品化徑擇線性注意力先行而已。

**預測**：至 2027 下半年，多數領先模型將匯於 DeepSeek 稀疏 + MTP 路線。

---

## 總結矩陣

| 維度 | 近期（1 年） | 中期（2-3 年） |
|---|---|---|
| **算力芯片** | 計算/KV 比上升 | 強度天花板逼近/超越訓練 |
| **互連** | 分叉：小 MTP→延遲 / 深推測→帶寬 | 拐點取決於 MTP 規模部署 |
| **超節點** | 大 EP 收益受抑 | 小域高帶寬成核心資產 |
| **專用芯片** | 流芯片（LPX）窗口收窄 | SRAM 中心假設過時 |
| **片上介質** | 國產/中端上桌 | HBM 壟斷溢價侵蝕 |
| **算力密度** | 加速堆疊 | 新 HBM 增長週期觸發 |
| **訪存牆** | 壓力緩解（非消除） | 算力翻倍後復強 |
| **片外介質** | DDR 覓得分層角色 | 混合存儲架構 |
| **Token 成本** | 2-4× 下降 | 2M+ 序列成標配 |

---

## 收尾

DeepSeek 之策略非偶然也——乃**系統設計方法論**焉：識成本結構中最貴之環，以算法創新破之，讓市場跟隨。較諸 Kimi3"堆精度"之路線，DeepSeek 之系統思維高出一境。

> **至明年或 2027 下半年，多數領先模型將匯於 DeepSeek 之譜。**

---

## 參考文獻

<a id="ref-1"></a>**[1]** DeepSeek-V3 技術報告 — MTP 機制：1-2 層預測深度，序列因果鏈，0.1 loss scaling。[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

<a id="ref-2"></a>**[2]** NVIDIA Groq 3 LPX 架構 — 500 MB SRAM/LPU, 150 TB/s, 128 GB/機架, 35× 每瓦吞吐量。[NVIDIA Blog](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/)

<a id="ref-3"></a>**[3]** HBM 市場數據 — 2026 年預計 $58B, SK Hynix 市佔 62%。[Introl](https://introl.com/blog/hbm-evolution-hbm3-hbm3e-hbm4-memory-ai-gpu-2025)

<a id="ref-4"></a>**[4]** HBM 代際 — HBM3: 819 GB/s → HBM3E: 1.2 TB/s → HBM4: 2.0 TB/s 每堆疊。GPU: H100 3.35 TB/s → B200 ~8 TB/s → Rubin ~22 TB/s。[Wikipedia](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)

<a id="ref-5"></a>**[5]** DeepSeek DSpark — 60-85% 推測解碼提速（2026.06）。半自回歸草擬 + 置信度調度驗證；明確解決深層 MTP 堆疊之接受率衰減。[arXiv:2607.05147](https://arxiv.org/abs/2607.05147)

<a id="ref-6"></a>**[6]** SGLang 推測解碼 — DeepSeek 模型 MTP 得 1.4× 吞吐量。[HPC-AI](https://company.hpc-ai.com/blog/sglang-speculative-decoding-tutorial)

### 相關閱讀（本站）

- [百萬序列：存儲 vs 計算，誰是真瓶頸？](million-seq-storage-vs-compute.html)
- [Kimi3 架構分析：線性注意力、稀疏注意力與百萬 Token 級的架構戰爭](kimi3-architecture-analysis.html)
- [Kimi3 成本效率：為何線性路線無法 Scaling Cost](kimi3-cost-efficiency.html)
