
# ChatGLM LoRA 客服对话微调项目

本项目基于 ChatGLM2 + LoRA 实现客服对话的微调与部署，支持数据标注 → LoRA 微调 → 推理部署全流程。

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

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 开始训练
python src/train.py

# 3. 推理验证
python src/infer.py

单轮推理：
python src/infer.py --mode single --query "你们的退货政策是怎样的？"

多轮对话：
python src/infer.py --mode interactive

# 4.用于将 LoRA 微调后的参数合并到原始 ChatGLM 模型中并保存为单一权重：

python scripts/convert_lora.py \
    --base_model THUDM/chatglm2-6b \
    --lora_model ./output/lora \
    --output_dir ./output/chatglm2-lora-merged
```
运行后会在 output_dir 中保存合并了 LoRA 的完整模型，可直接使用 AutoModel.from_pretrained() 加载，无需再加载 LoRA 参数。


