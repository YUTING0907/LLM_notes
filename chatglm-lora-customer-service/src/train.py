from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import load_dataset
from src.dataset import CustomerDataset

'''
使用 peft 的 LoRA 方式对 ChatGLM 模型微调：
'''

model_name = "THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_key_value"]
)

model = get_peft_model(model, peft_config)

# 加载自定义数据
dataset = CustomerDataset("data/train.json", tokenizer)

# 训练参数
training_args = TrainingArguments(
    output_dir="./lora_ckpt",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir='./logs',
    save_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
