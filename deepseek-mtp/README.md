# DeepSeek dspark MTP算法研究目录

## 目录结构

```
deepseek_mtp_research/
├── index.html                  # 主研究报告（HTML，适合手机查看）
├── README.md                   # 本文件
├── dspark_paper.pdf            # arxiv原始论文 (2507.18029)
├── sources_and_findings.md     # 关键发现和来源汇总
├── next_platform_article.md    # The Next Platform 报道内容
└── tom_hardware_article.md     # Tom's Hardware 报道内容
```

## 研究概述

研究主题：DeepSeek dspark MTP（Multi-Token Prediction）算法对算力和总线系统行业的影响

核心发现：
1. dspark通过一次解码生成多个token，将LLM推理从内存带宽密集型转向计算密集型
2. HBM内存需求弹性降低30-50%
3. NVLink/NVSwitch在推理场景中的必需性下降
4. ASIC推理芯片竞争力大幅增强
5. 中国算力自主可控获得关键技术杠杆

## 主要来源

- arxiv 2507.18029 - dspark原始论文
- The Next Platform - 深度产业分析
- Tom's Hardware - 技术报道
- 腾讯云开发者社区 - 中文技术分析
- Interconnects Blog - 技术深度解析
- SemiAnalysis - 产业影响分析
