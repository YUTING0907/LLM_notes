# src/dataset.py

import json
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ChatGLMDataset(Dataset):
    """
    用于 ChatGLM LoRA 微调的数据集类。
    数据格式为 JSON，每条数据包含：
    {
        "conversation": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
            ...
        ]
    }
    """
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(data_path)

    def _load_data(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]
        conversation = example["conversation"]

        prompt = ""
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                prompt += f"[用户] {content}\n"
            elif role == "assistant":
                prompt += f"[助手] {content}\n"

        # ChatGLM 的输入是拼接的纯文本
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # 标签设置为 input_ids，但用户部分 mask 掉
        labels = input_ids.clone()
        user_prefix = True
        for i, token_id in enumerate(input_ids):
            token_str = self.tokenizer.decode(token_id)
            if "[用户]" in token_str:
                user_prefix = True
            elif "[助手]" in token_str:
                user_prefix = False
            if user_prefix:
                labels[i] = -100  # 忽略 loss 计算

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
