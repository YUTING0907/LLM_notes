# src/infer.py

import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import argparse

def load_model(base_model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True).half().cuda()
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return tokenizer, model

def chat(model, tokenizer, history, query):
    response, history = model.chat(tokenizer, query, history=history)
    return response, history

def interactive_chat(tokenizer, model):
    print("欢迎使用 ChatGLM-客服助手 (输入 'exit' 退出)")
    history = []
    while True:
        query = input("用户：")
        if query.strip().lower() == "exit":
            break
        response, history = chat(model, tokenizer, history, query)
        print(f"客服：{response}")

def single_turn_chat(tokenizer, model, query):
    response, _ = chat(model, tokenizer, history=[], query=query)
    print(f"客服：{response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="THUDM/chatglm2-6b", help="基础模型路径或名称")
    parser.add_argument("--lora_model", type=str, default="./output/lora", help="LoRA 微调模型路径")
    parser.add_argument("--mode", choices=["interactive", "single"], default="interactive", help="对话模式")
    parser.add_argument("--query", type=str, default=None, help="单轮对话的用户输入")
    args = parser.parse_args()

    tokenizer, model = load_model(args.base_model, args.lora_model)

    if args.mode == "interactive":
        interactive_chat(tokenizer, model)
    elif args.mode == "single":
        if not args.query:
            raise ValueError("请提供 --query 参数用于单轮推理")
        single_turn_chat(tokenizer, model, args.query)
