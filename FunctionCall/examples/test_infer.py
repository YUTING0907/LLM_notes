# examples/test_infer.py

import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from scripts.prompt_utils import build_instruction_prompt, parse_function_call_output

# 示例工具定义（可换为加载 JSON schema）
TOOLS = [
    {
        "name": "get_weather",
        "description": "获取指定城市当前天气",
        "parameters": {
            "city": {
                "type": "string",
                "description": "城市名称，例如 Beijing",
                "required": True
            }
        }
    },
    {
        "name": "get_time",
        "description": "获取指定城市当前时间",
        "parameters": {
            "city": {
                "type": "string",
                "description": "城市名称，例如 Shanghai",
                "required": True
            }
        }
    }
]

# 模拟工具执行函数
def fake_tool_executor(call: dict):
    name = call.get("name")
    params = call.get("parameters", {})
    if name == "get_weather":
        city = params.get("city", "")
        return f"{city} 当前天气：晴，25°C"
    elif name == "get_time":
        city = params.get("city", "")
        return f"{city} 当前时间：15:42"
    return "未知工具"

def main():
    model_name = "Qwen/Qwen1.5-0.5B-Chat"  # 可换成你自己的 SFT 模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

    # 用 pipeline 简化调用（也可使用 generate 自定义）
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, max_new_tokens=512)

    print(">>> Function Call 多轮问答 Demo，输入 'exit' 退出。")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            break

        prompt = build_instruction_prompt(user_input, TOOLS)
        output = pipe(prompt)[0]['generated_text']

        print("\n[Raw Output]\n", output)

        function_call = parse_function_call_output(output)
        if not function_call:
            print("❌ 无法解析函数调用。")
            continue

        print("\n✅ 函数调用结构:\n", json.dumps(function_call, indent=2, ensure_ascii=False))

        result = fake_tool_executor(function_call)
        print("\n🤖 工具执行结果:\n", result)

        # 模拟继续问答
        follow_up = f"User: {result}，请问还需要获取其他信息吗？"
        print("\n[Follow-up]\n", follow_up)

if __name__ == "__main__":
    main()
