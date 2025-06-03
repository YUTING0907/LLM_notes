# scripts/train_sft.py
'''
    # 示例命令
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft.py \
    --base_model "Qwen/Qwen1.5-0.5B-Chat" \
    --train_file ./data/train.json \
    --output_dir ./sft_function_call_model \
    --use_lora True \
    --num_train_epochs 3
'''
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType

def format_example(example):
    user_input = example["input"]
    function_call = json.dumps(example["output"]["function_call"], ensure_ascii=False)
    prompt = f"用户：{user_input}\n助手：请调用如下函数：{function_call}"
    return {"text": prompt}

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list([format_example(item) for item in raw_data])
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--use_lora', action='store_true')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    dataset = load_data(args.data_path)
    dataset = dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding='max_length', max_length=512), batched=True)

    if args.use_lora:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none"
        )
        model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    trainer.train()
    trainer.save_model(args.save_path)

if __name__ == '__main__':
    main()
