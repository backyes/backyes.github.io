# 分析Prompt规划

## 论文核心
3DLS: A 3D Logic-Stacked Architecture for Disaggregated LLM Serving
- 作者: Jaehun Lee, In-Jun Jung, Joo-Young Kim (KAIST)
- 发表: IEEE Computer Architecture Letters 2026
- arXiv: 2607.01617

## 分析视角(多维度专家)
1. 芯片架构专家: 3D集成技术可行性、TSV/热预算/封装
2. 系统专家: PD disaggregation + TP 调度
3. AI推理系统: KV Cache传输 vs AllReduce的关键路径分析
4. 互连架构: D2D互连、UCIe-3D、带宽/延迟权衡
5. 批判性审视: 假设局限性、实际部署gap、与工业界(Groq/Cerebras/Dojo)对比

## 关键分析问题
- 3D集成的margin收益是否值得堆叠开销?
- iso-bandwidth假设是否公平?
- 200W/cm²冷却包络假设实用性?
- 与晶圆级芯片(WSC-LLM, Dojo)路线的优劣
- KV Cache传输是否真的是瓶颈?
- 对LLM架构演进(如SSM/MoE)的适用性
