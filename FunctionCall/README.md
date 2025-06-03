# 🔧 LLM Function Calling with Multi-turn Interaction

本项目展示了如何训练和部署一个支持结构化函数调用（Function Calling）和多轮交互的语言模型。模型能够根据用户输入识别要调用的函数、提取参数，调用后继续对话，完成工具增强问答流程。

## ✨ 项目特色

- ✅ 支持 Function Call 风格训练（兼容 OpenAI JSON Schema）
- ✅ 工具描述提示增强（Tool Use Prompting）
- ✅ 支持多轮调用：调用完函数后根据返回结果继续对话
- ✅ 基于 `transformers` + `ChatGLM`/`Mistral` 等开源模型
- ✅ 示例代码可一键运行

---

## 📁 项目结构

```bash
.
├── configs/
│   └── tools.json               # 函数工具定义（描述、参数等）
├── data/
│   └── train.json               # 监督微调数据，输入→函数名 + 参数
├── examples/
│   └── test_infer.py            # 推理脚本，支持多轮 Function Call
├── scripts/
│   ├── train_sft.py             # 使用 LoRA 或全量 SFT 训练脚本
│   └── prompt_utils.py          # Tool prompting + 模板构造
├── models/
│   └── function_call_model/     # 本地模型权重保存目录
├── requirements.txt             # 项目依赖
└── README.md                    # 当前文件
