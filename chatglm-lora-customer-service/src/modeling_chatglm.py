# src/modeling_chatglm.py

from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch


def load_base_model(model_name_or_path: str, load_in_8bit: bool = False, device: str = "cuda"):
    """
    加载 ChatGLM 原始模型（支持从 HuggingFace 或本地加载）。
    支持 8-bit 量化加载以减少显存占用。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        load_in_8bit=load_in_8bit
    )
    model.eval()
    return tokenizer, model


def load_lora_model(base_model_path: str, lora_path: str, device: str = "cuda"):
    """
    加载基座模型 + LoRA 权重，合并为一个推理用模型。
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )

    # 加载并注入 LoRA 参数
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return tokenizer, model


def save_merged_lora_model(lora_model_path: str, save_path: str):
    """
    合并 LoRA 权重到原始模型，并保存为新模型。
    """
    config = PeftConfig.from_pretrained(lora_model_path)
    model = AutoModel.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_model_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
