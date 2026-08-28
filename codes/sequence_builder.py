import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.add_special_tokens({"additional_special_tokens": ["<EMO>"]})
model.resize_token_embeddings(len(tokenizer))
emo_token_id = tokenizer.convert_tokens_to_ids("<EMO>")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.get_input_embeddings().weight.requires_grad = True


def build_input_embeds(text, visual_tokens, device):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    text_embeds = model.get_input_embeddings()(input_ids).squeeze(0)

    emo_embed = model.get_input_embeddings()(torch.tensor([emo_token_id], device=device))

    sequence = torch.cat([visual_tokens.to(device), text_embeds, emo_embed], dim=0)
    attention_mask = torch.ones(sequence.shape[0], device=device)

    return sequence.unsqueeze(0), attention_mask.unsqueeze(0)