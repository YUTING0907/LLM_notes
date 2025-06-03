# scripts/prompt_utils.py

from typing import List, Dict
import json


def build_tool_description_prompt(tools: List[Dict]) -> str:
    """
    将工具（函数）描述转换为统一格式的文本提示，供模型理解每个函数的功能和参数。
    """
    tool_prompts = []
    for tool in tools:
        name = tool["name"]
        description = tool.get("description", "")
        parameters = tool.get("parameters", {})
        param_descriptions = []

        for param_name, param_info in parameters.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            param_required = param_info.get("required", False)
            param_required_str = " (required)" if param_required else ""
            param_descriptions.append(f"- `{param_name}`: {param_type}{param_required_str}. {param_desc}")

        params_prompt = "\n".join(param_descriptions)
        tool_prompts.append(f"### Tool: `{name}`\n{description}\n\nParameters:\n{params_prompt}")

    return "\n\n".join(tool_prompts)


def build_instruction_prompt(user_query: str, tools: List[Dict]) -> str:
    """
    构造完整的指令微调 prompt，其中包含工具描述和用户查询。
    """
    tool_desc_prompt = build_tool_description_prompt(tools)
    instruction = f"""You are an intelligent assistant. You can call tools in structured JSON format.

{tool_desc_prompt}

Now based on the following user request, decide which tool to call and provide the correct parameters in JSON format.

User: {user_query}

Your answer must be in the format:
```json
{{"name": "<tool_name>", "parameters": {{...}} }}
```"""

    return instruction


def parse_function_call_output(output: str) -> Dict:
    """
    尝试从模型输出中提取函数调用的 JSON 结构，便于执行或评估。
    """
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        return json.loads(json_str)
    except Exception as e:
        print(f"[Warning] Failed to parse function call JSON: {e}")
        return {}

