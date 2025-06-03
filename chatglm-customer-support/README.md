# 🤖 ChatGLM 客服对话微调项目 (LoRA + Function Call)

本项目基于 ChatGLM 模型，使用 LoRA 技术完成客服对话任务的指令微调，支持结构化 Function Call、多轮交互，并可部署为 API 服务。

---

## 🔧 项目结构

```
chatglm-customer-support/
├── data/
│   └── customer_service.json       # 训练数据
├── scripts/
│   ├── train_sft.py                # LoRA 微调脚本
│   └── infer.py                    # 推理脚本
├── deploy/
│   └── app.py                      # Flask 接口服务
├── output/
│   └── chatglm_lora/               # 微调输出模型
```
