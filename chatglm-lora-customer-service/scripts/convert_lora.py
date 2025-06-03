# scripts/convert_lora.py

import os
import argparse
from transformers import AutoModel
from peft import PeftModel

def merge_lora_and_save(base_model_path: str, lora_path: str, output_path: str):
    # 加载基础模型
    print(f"Loading base model from: {base_model_path}")
    model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True)
    model = model.half().cuda()

    # 加载 LoRA 权重
    print(f"Loading LoRA weights from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    # 合并权重
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # 保存合并后的模型
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="原始 ChatGLM 模型路径或名称")
    parser.add_argument("--lora_model", type=str, required=True, help="LoRA 权重路径")
    parser.add_argument("--output_dir", type=str, required=True, help="保存合并模型的路径")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    merge_lora_and_save(args.base_model, args.lora_model, args.output_dir)
