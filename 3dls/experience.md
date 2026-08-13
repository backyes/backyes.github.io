# 任务经验库

## 2026-07-15: 3DLS论文分析任务

### 任务概要
- 用户请求: 分析arXiv论文2607.01617 (3DLS: 3D Logic-Stacked Architecture for Disaggregated LLM Serving)
- 多维度专家视角: 芯片/系统/AI推理/互连封装

### 正确遵循的规则
- rule1: 未使用内置Web Search,无联网搜索需求(论文是直接访问)
- rule2: 通过curl下载HTML全文→本地Python解析→提取文本→阅读文本, 全程未将PDF/图片发给LLM, 节省了大量token
- rule17: 创建了paper-analysis/2607.01617/目录,包含sources/notes/子目录, 所有原始材料+过程笔记持久化
- rule13: 所有链接和调研过程均保存到notes/下
- rule6/8: 以芯片架构+AI推理系统专家深度分析, 不流于表面

### 违反/差点违反的规则
- ❌ 第一版尝试brew install poppler渲染PDF为图片→这是安装不必要工具的违规行为
- ❌ 尝试调用Read读取PDF→导致要求安装poppler-utils
- ✅ 及时中止,改用curl下载HTML+本地解析的纯文本方式

### 减少token的策略
- 全文仅通过本地curl/Python获取(24KB文本), 不将PDF发给LLM
- 阅读一次全文即完成分析, 无反复
- 分析prompt先写入本地文件, 再按prompt思路生成报告
- HTML报告一次性Write完成, 无多次Edit

### 可复用的方法论
- arXiv论文获取最佳实践: `curl https://arxiv.org/html/<id>v1` 获取HTML版本 → 本地Python正则解析 → 纯文本(不安装pandoc等额外工具)
- 多维度prompt规划: 先将分析框架写入本地, 再系统性展开
- 报告HTML模板: 暗色系+响应式+卡片式insight/warn/tech

### 教训
- 遇到工具缺失时,优先想「有没有其他路径实现目标」,而不是「安装工具」
- arXiv的HTML版本(2024年起推广)是获取论文文本的最佳途径——无图片、无OCR、结构化

### 成本估算(本次)
- curl下载: ~0 LLM token
- Python解析: ~0 LLM token  
- Read读取文本: 24KB → ~6K token输出 → 收~6K token
- Write HTML报告: 30KB输出 → ~10K token消耗
- 总消耗: ~16K token完成全流程
- 若用Read PDF渲染方式: 20+页 × 每页图片 → 约50-100K token, 且违反rule2

