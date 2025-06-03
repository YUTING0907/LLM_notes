# 🤖 ChatGLM 客服对话微调项目 (LoRA + Function Call)

本项目基于 ChatGLM 模型，使用 LoRA 技术完成客服对话任务的指令微调，支持结构化 Function Call、多轮交互，并可部署为 API 服务。

---

## 🔧 项目结构

```
chatglm-lora-customer-service/
├── data/
│   └── train.json               # 微调训练数据（格式见下）
├── src/
│   ├── dataset.py              # 数据加载与预处理
│   ├── modeling_chatglm.py     # ChatGLM 模型加载（引用官方或 HuggingFace）
│   ├── train.py                # LoRA 微调训练脚本
│   └── infer.py                # 推理脚本（单轮/多轮对话）
├── scripts/
│   └── convert_lora.py         # 合并/保存 LoRA 参数到原模型
├── Dockerfile
├── requirements.txt
└── README.md
```
