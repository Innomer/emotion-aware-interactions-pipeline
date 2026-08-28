import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.add_special_tokens({"additional_special_tokens": ["<EMO>"]})
model.resize_token_embeddings(len(tokenizer))
emo_token_id = tokenizer.convert_tokens_to_ids("<EMO>")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["embed_tokens"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


def build_sequence(text, visual_tokens, device):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    text_embeds = model.get_input_embeddings()(input_ids).squeeze(0)

    emo_embed = model.get_input_embeddings()(torch.tensor([emo_token_id], device=device))
    visual_tokens = visual_tokens.to(device).reshape(-1, visual_tokens.shape[-1]).to(text_embeds.dtype)

    return torch.cat([visual_tokens, text_embeds, emo_embed], dim=0)


def build_input_embeds(text, visual_tokens, device):
    sequence = build_sequence(text, visual_tokens, device)
    attention_mask = torch.ones(sequence.shape[0], device=device)
    return sequence.unsqueeze(0), attention_mask.unsqueeze(0)


def build_batch_input_embeds(texts, visual_tokens_list, device):
    sequences = [build_sequence(t, v, device) for t, v in zip(texts, visual_tokens_list)]
    max_len = max(s.shape[0] for s in sequences)
    hidden_dim = sequences[0].shape[-1]

    padded = torch.zeros(len(sequences), max_len, hidden_dim, dtype=sequences[0].dtype, device=device)
    attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)

    for i, seq in enumerate(sequences):
        pad_len = max_len - seq.shape[0]
        padded[i, pad_len:] = seq
        attention_mask[i, pad_len:] = 1

    return padded, attention_mask