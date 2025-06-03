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
```
## 快速开始
### 🚀 1. 准备训练数据
data/train.json
```
{
  "instruction": "请告诉我北京的天气",
  "function_call": {
    "name": "get_weather",
    "parameters": {
      "city": "北京"
    }
  }
}

```
构造方法：
将任意问句（instruction）人工对齐目标函数（name）和参数（parameters），即可组成训练数据。

### 🛠️ 2. 微调模型
使用 LoRA + PEFT + Transformers 对模型进行指令微调。

scripts/train_sft.py 中调用训练数据：

```
# 示例命令
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft.py \
    --base_model "Qwen/Qwen1.5-0.5B-Chat" \
    --train_file ./data/train.json \
    --output_dir ./sft_function_call_model \
    --use_lora True \
    --num_train_epochs 3
```

### 🧪 3. 推理阶段：Function Call 多轮对话
python examples/test_infer.py

流程：

用户输入问题；

构造带 Tool Descriptions 的 Prompt；

模型输出结构化函数调用（如 JSON）；

解析 JSON → 执行工具；

展示执行结果，并继续下一轮问答（如：“还想知道其他城市的天气吗？”）；

### 🧰 4. 工具执行模拟（可替换为真实 API）
```
def fake_tool_executor(call: dict):
    if call["name"] == "get_weather":
        return "北京当前天气：多云，26℃"
```
也可以替换为真实 API，比如天气接口、数据库查询等。

### 🗣️ 5. 多轮对话处理
test_infer.py 中支持上下文维护，可以通过将 历史问答 拼接到 Prompt 实现上下文增强，或者使用 memory 类封装。

### ✅ 6. 输出结果（示意）
```
You are an intelligent assistant. You can call tools in structured JSON format.

### Tool: `get_weather`
Get the current weather of a specified city.

Parameters:
- `city`: string (required). Name of the city.

### Tool: `get_time`
Get the current time for a location.

Parameters:
- `location`: string (required). Target location.

Now, based on the following user request, decide which tool to call and provide the correct parameters in JSON format.

User: 我想知道北京的天气

Your answer must be in the format:
```json
{"name": "get_weather", "parameters": {"city": "北京"}}

```
