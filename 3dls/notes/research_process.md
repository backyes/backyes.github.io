# 调研过程记录

## 关键阅读与总结
1. 阅读论文全文(HTML提取),理解3DLS核心思想
2. 识别关键问题: PD disaggregation + TP 在2D/2.5D芯片上的混合流量争用
3. 分析3DLS方案: 物理隔离KV Cache传输与decode-side AllReduce
4. 评估方法: 与Naive Planar和PM-Planar对比,iso-bandwidth配置
5. 批判性审视: 3D集成开销、热预算、与晶圆级方案对比

## 关键洞察
- 问题定义清晰: 异构流量共享同一D2D fabric导致decode关键路径延迟
- 解决方案优雅: 利用3D垂直互连物理隔离,而非简单增加带宽
- 实验设计合理: iso-bandwidth对比突出"隔离vs带宽"的trade-off
- 局限性: 仅仿真验证、未考虑实际3D工艺约束、热耦合问题简化

## 检索记录
- arxiv HTML版本下载成功(61KB)
- 本地Python解析HTML为纯文本(24KB)
