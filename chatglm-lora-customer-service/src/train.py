# src/train.py

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.dataset import ChatGLMDataset
from src.modeling_chatglm import load_chatglm_model

'''
使用 peft 的 LoRA 方式对 ChatGLM 模型微调：
'''

def main():
    model_name_or_path = "THUDM/chatglm2-6b"
    data_path = "./data/train.json"
    output_dir = "./output"

    # 载入分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = load_chatglm_model(model_name_or_path)

    # 量化 + LoRA 配置（可选）
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 数据集和 DataCollator
    dataset = ChatGLMDataset(data_path, tokenizer, max_length=2048)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()

    # 保存模型和 LoRA adapter
    model.save_pretrained(os.path.join(output_dir, "lora"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora"))

if __name__ == "__main__":
    main()

