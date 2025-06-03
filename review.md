LLM 基础：大模型是怎么训练出来的？
Transform 的架构，Encoder 和 Decoder 是什么？
Function Call 是怎么训练的？
微调的方案有哪些？自己做过没有？
大模型分词器是什么？
Embedding 是什么？你们用的那个模型？

# 💡 LLM 基础知识详解

## 1. 🧠 大模型是怎么训练出来的？

大语言模型（LLM）的训练通常分为以下几个阶段：

### ① 数据准备

- **文本清洗**：去除低质量、重复、违法内容
- **去重/分词/格式化**：统一编码、语言标识
- **多语种数据处理**：构建语言比例（如中英比 2:8）
- **Tokenization**：使用自定义分词器转为 Token 序列

### ② 预训练（Pretraining）

- 目标：让模型学习语言本身的结构、语义、常识
- 常见任务：
  - **Causal Language Modeling (CLM)**：如 GPT 系列，只看前文预测下一个 token
  - **Masked Language Modeling (MLM)**：如 BERT，随机 mask 输入部分 token 并预测它们
- 数据量：通常需要数百 GB 到数 TB 的文本数据
- 训练过程：
  - 使用分布式并行技术（数据并行、模型并行、流水线并行）
  - FP16 / BF16 混合精度加速训练
  - 使用 AdamW 优化器、warmup + cosine decay 学习率调度

### ③ 微调（Fine-tuning）

- **SFT（监督微调）**：加入人工标注的 QA、指令跟随等数据
- **RLHF（人类反馈强化学习）**：
  - 收集成对偏好数据（A vs B 哪个回答好）
  - 用奖励模型评分，用 PPO（Proximal Policy Optimization）进行强化学习
- **DPO（Direct Preference Optimization）**：一种稳定、无需训练奖励模型的直接优化方法

---

## 2. 🏗️ Transformer 架构与 Encoder / Decoder 区别

### Transformer 核心组件

- **Self-Attention**：每个 token 关注输入序列中所有位置
- **Multi-Head Attention**：多组 attention 并行学习不同子空间信息
- **Feed-Forward Network**：非线性变换增强模型表达力
- **Residual Connection** + LayerNorm：稳定训练过程
- **Positional Encoding**：加入位置信息

### Encoder vs Decoder

| 构件      | Encoder                                 | Decoder                                 |
|-----------|------------------------------------------|------------------------------------------|
| 输入类型  | 全部可见输入                            | 自回归输入（仅当前及之前 token 可见）  |
| 用途      | 理解任务（如分类、摘要）                | 生成任务（如翻译、对话、写作）          |
| 特征      | 双向注意力（BERT）                      | 单向注意力 + Encoder-Decoder 注意力（GPT）|

---

## 3. 📞 Function Call 是怎么训练的？

### 任务定义

Function Call 即 LLM 调用外部工具/函数完成复杂任务，如调用日历、查询数据库、发送请求等。
利用现有的框架进行 Function Call（函数调用）训练，本质是通过 指令微调（Instruction Tuning） 或 函数调用微调（Function Calling Fine-tuning） 方式，让模型学会识别用户意图并输出结构化函数调用格式（JSON)

### 训练方式

- **添加监督数据**：构造输入→输出是函数名 + 参数的 JSON（如 OpenAI Function Call 数据）
- **训练过程**：
  - 基于原始指令微调模型，使其学会输出结构化函数调用格式
  - 引入工具描述作为提示模板（Tool Use Prompting）

### 推理流程

1. 用户发出请求（如“查明天北京天气”）
2. 模型输出函数调用：
   ```json
   {
     "function": "get_weather",
     "arguments": {"city": "北京", "date": "2025-05-29"}
   }
3.系统调用后端 API 获取结果

4.模型继续对响应内容生成最终回复

## 4. 🔧 微调方案有哪些？你是否做过？

### 常见微调方法

| 方法类型            | 描述                                           |
|---------------------|------------------------------------------------|
| Full Fine-tuning     | 全量参数微调，效果最好但资源消耗大               |
| LoRA / QLoRA         | 参数高效微调，仅更新小量 Adapter 参数             |
| PEFT                 | PEFT 是 Parameter-Efficient Fine-Tuning 的统称   |
| Prompt-tuning / P-tuning v2 | 仅训练可学习的 Prompt 嵌入向量                     |

### 实践经验

✅ 使用过 LoRA 对 ChatGLM 微调客服对话任务，包含数据标注、训练、部署全过程。

---

## 5. 🧩 大模型分词器是什么？

### 功能作用

- 将文本编码为 token id（整数），输入模型处理
- 控制序列长度，提高编码效率

### 主流分词算法

| 算法名称               | 基于   | 词表大小 | 多语言支持 | OOV 处理 | 应用模型            |
| ------------------ | ---- | ---- | ----- | ------ | --------------- |
| **BPE**（Byte Pair Encoding）  | 频率合并 | 中    | 一般    | 良好     | GPT-2、RoBERTa   |
| **WordPiece**      | 语言模型 | 中等   | 较差    | 良好     | BERT、DistilBERT |
| **Unigram LM** （SentencePiece） | 概率建模 | 小    | 优秀    | 最佳     | T5、XLNet、mBART  |
| **Byte-level BPE** | 字节处理 | 小    | 极佳    | 最佳     | GPT-3、ChatGPT   |

BPE 是一种数据压缩启发式分词方法。最初被用于图像压缩领域，后被 NLP 借鉴。
初始将文本分割为 字符级别 token；
统计所有相邻字符对的出现频率；
每次选择频率最高的字符对，将其 合并为新 token；
不断迭代，直到达到预设词表大小。

Byte-level BPE 是 GPT 系列使用的分词器，它在标准 BPE 上进行了修改：
对每个字符按字节处理（Unicode 编码），包括空格、标点；
再进行 BPE 合并操作；
支持几乎所有语言字符集，避免编码失败。

---

## 6. 🧬 什么是 Embedding？你们使用哪个模型？

### 定义

Embedding 是将文本/句子转为稠密向量，用于语义计算。

### 应用场景

- 检索（Semantic Search）
- 聚类、分类
- 相似度计算

### 常用模型

- `text-embedding-ada-002`：OpenAI 提供的多语言模型
- `bge-large-zh`：中文语义检索效果佳
- `m3e-base`：多任务中文模型，覆盖分类/搜索/问答等任务

---

> 📝 注：如需生成 PDF 或插入图示、模型结构图，请继续告知。




Lib：
介绍一下 langchian
介绍一下 autogen
有没有用过大模型的网关框架（litellm）
为什么手搓 agent，而不是用框架？
mcp 是什么？和 Function Call 有什么区别？有没有实践过？
A2A 了解吗？

Prompt：
ReAct 是啥？怎么实现的？
CoT 是啥？为啥效果好呢？有啥缺点？
Prompt Caching 是什么？
温度值/top-p/top-k 分别是什么？各个场景下的最佳设置是什么？

RAG：
你介绍一下RAG 是什么？最难的地方是哪？
文档切割策略有哪些？怎么规避语义被切割掉的问题？
多路召回是什么？
文档怎么存的？粒度是多大？用的什么数据库？
为啥要用到图数据库？
向量数据库的对比有没有做过？Qdrant 性能如何？量级是多大？有没有性能瓶颈？
怎么规避大模型的幻觉？
微调和 RAG 的优劣势？
怎么量化你的回答效果？例如检索的效果、回答的效果。

workflow：
怎么做的任务拆分？为什么要拆分？效果如何？怎么提升效果？
text2sql 怎么做的？怎么提高准确率？
如何润色query，目的是什么？
code-generation 是什么做的？如何确保准确性？
现在再让你设计你会怎么做？（replan）
效果是怎么量化的？

Agent：
介绍一下你的 Agent 项目长短期记忆是怎么做的？
记忆是怎么存的？粒度是多少？怎么用的？
Function Call 是什么做的？
你最大的难题是什么？你是怎么提高效果的？怎么降低延迟的？
端到端延迟如何优化的？
介绍一下 single-agent、multi-agent 的设计方案有哪些？
反思机制是什么做的？为什么要用反思？
如何看待当下的 LLM 应用的趋势和方向
为什么要用 Webrtc？它和 ws 的区别是什么？
agent 服务高可用、稳健性是怎么保证的？
llm 服务并发太高了怎么办？

系统设计题：
短链系统
分布式锁的设计
给你一部长篇小说，怎么做文档切割？
怎么做到论文翻译，并且格式尽可能和原来的统一
游戏社区客服助手设计。如何绑定游戏黑话，如何利用好公司内部的文档
结合线上问题快速定位项目工程代码有问题的地方
有很多结构化和非结构化数据，怎么分析，再怎么得出我要的结论

八股：
go 的内存分配策略、GMP、GC
python 的内存分配策略、GC
redis 用过那些？mget 底层什么实现的？、zset 怎么实现的？
mysql 索引怎么设计最好？数据库隔离级别？mvcc 是怎么实现的？
分布式锁是什么实现的？
kafka 的 reblance 是什么？会产生那些问题？怎么保证数据不丢?
fastapi 设计原理？
go 中 net/http 如何处理的 tcp 粘包问题
http2 是什么？比 http1.1 有什么优势？
Linux 网络性能调优的方式
如何定位 Linux 中的 pid、端口号等等
