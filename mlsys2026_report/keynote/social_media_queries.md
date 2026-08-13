# Social Media Queries · 待用户在能联网终端跑

> 本调度环境无法访问 reddit / x.com / youtube / 微信 / 知乎。  
> 下列 query 已按平台分组，**直接复制粘贴**到对应搜索框即可。  
> 搜到的高价值结果请补回各 `0X_*.md` 的「社交媒体讨论」段。

---

## 1. YouTube（官方录像 + 二次上架）

```
MLSys 2026 keynote
MLSys 2026 Saroufim kernel
MLSys 2026 Lidong Zhou system intelligence
MLSys 2026 Vahdat Google
MLSys 2026 Zettlemoyer pretraining
MLSys 2026 Kozyrakis inference
site:youtube.com MLSys 2026
```

特别留意以下频道：
- `@MLSys-Conference`
- `@Google` / `@GoogleCloudTech`
- `@MicrosoftResearch`
- `@AllenSchool`（Luke Zettlemoyer 主页所在）
- `@StanfordOnline` / `@StanfordHAI`
- `@NVIDIADeveloper`
- `@GPUMODE`（Saroufim 自家社区）

---

## 2. Reddit

```
site:reddit.com MLSys 2026
site:reddit.com MLSys 2026 keynote
site:reddit.com Saroufim GPU MODE kernel LLM
site:reddit.com Lidong Zhou system intelligence
site:reddit.com Amin Vahdat MLSys
site:reddit.com Zettlemoyer BLT byte latent transformer
site:reddit.com Kozyrakis inference efficiency
```

主要关注子版：
- r/MachineLearning
- r/LocalLLaMA
- r/cscareerquestions（career-style 讨论 Saroufim 那段"代码成本趋零"建议）
- r/ResearchML
- r/computerscience
- r/MLSys（如存在）

---

## 3. X.com（Twitter） · 高级搜索

时间窗：2026-05-17 ~ 2026-05-25（会议期 + 余波 1 周）

```
("MLSys 2026" OR #MLSys2026) since:2026-05-17 until:2026-05-25
"MLSys 2026" keynote since:2026-05-17 until:2026-05-25
from:msaroufim since:2026-05-17 until:2026-05-25
from:LukeZettlemoyer since:2026-05-17 until:2026-05-25
from:lidong_zhou since:2026-05-17 until:2026-05-25
from:KozyrakisCS since:2026-05-17 until:2026-05-25
from:amin_vahdat since:2026-05-17 until:2026-05-25
"GPU MODE" "MLSys" since:2026-05-17 until:2026-05-25
"system intelligence" Lidong since:2026-05-17 until:2026-05-25
"Byte Latent Transformer" MLSys since:2026-05-17 until:2026-05-25
"FlashInfer" MLSys since:2026-05-17 until:2026-05-25
```

也可以用 `(@msaroufim OR @LukeZettlemoyer) MLSys` 类组合。

---

## 4. HackerNews（多年来 MLSys 顶级 keynote 都会有热帖）

```
https://hn.algolia.com/?q=MLSys+2026
https://hn.algolia.com/?q=Saroufim+GPU+kernel
https://hn.algolia.com/?q=Zettlemoyer+pretraining
https://hn.algolia.com/?q=Lidong+Zhou+system+intelligence
https://hn.algolia.com/?q=Kozyrakis+inference
```

---

## 5. 中文圈（知乎 / 微信公众号 / 小红书）

```
知乎搜索：MLSys 2026 keynote
知乎搜索：周礼栋 MLSys 系统智能 / system intelligence
知乎搜索：Vahdat Google 数据中心
知乎搜索：Saroufim GPU MODE kernel 自动生成
知乎搜索：Zettlemoyer 预训练 / BLT / 字节级
知乎搜索：Kozyrakis 推理效率

微信搜一搜：MLSys 2026 / 机器学习与系统大会 2026 / 周礼栋 MLSys
百度搜：MLSys 2026 综述 -site:mlsys.org
```

中文翻译参考：
- 周礼栋（Lidong Zhou）
- 阿明·瓦德（Amin Vahdat）
- 卢克·泽特尔莫耶（Luke Zettlemoyer）
- 克里斯托斯·科齐拉基斯（Christos Kozyrakis）
- 马克·萨鲁菲姆（Mark Saroufim）

---

## 6. arXiv / Google Scholar（找 keynote 配套论文）

```
arXiv: site:arxiv.org Saroufim KernelBench OR "GPU kernel" benchmark
arXiv: "Byte Latent Transformer" Meta
arXiv: s1 simple test-time scaling
arXiv: Branch-Train-Merge Branch-Train-Mix
arXiv: agentic inference scheduling
Google Scholar：cluster:<doi> for each speaker since 2024
```

---

## 7. 推荐工作流

1. 跑 §1 找官方录像→若已上线，**优先存证 SlidesLive 直链**（不要只存 youtube link，因为 MLSys 录像会双发布）
2. 跑 §3（X 高级搜索）拉时间窗内 live-tweet → 找会议期间的 thread
3. 跑 §2 + §4 拉长篇讨论
4. 跑 §5（中文圈）拉国内复盘
5. §6 用来验证 keynote 中提到的具体工作是否有 arXiv / GitHub 配套发布
6. 把每场 keynote 找到的 ≥3 条最高质量 URL + 一两句精华引用 编进对应 `0X_*.md`

---

最后一条小提醒：**不要把"在某平台找不到讨论"当成"没人在讨论"**——MLSys 这种小众会议常常在垂直社区（GPU MODE Discord、Discord TPU 群、X 私人圈层）有大量 insider 内容，不公开搜不到，必要时联系参会朋友拿一手 notes。
