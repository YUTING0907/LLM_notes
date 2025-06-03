from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
model.load_adapter("lora_ckpt", "default")  # 加载LoRA适配器
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

while True:
    query = input("客户: ")
    response, _ = model.chat(tokenizer, query, history=[])
    print("客服:", response)
