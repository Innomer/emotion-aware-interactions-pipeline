import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HIDDEN_DIM = 1536

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


class EmoToken(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

    def forward(self):
        return self.embedding


emo_token = EmoToken()


def build_sequence(text, visual_tokens, device):
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a Engligsh speaking robot companion. Based on the conversation and the speaker's visible emotional state, reply briefly and appropriately in one short sentence in english.",
            },
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = model.get_input_embeddings()(input_ids).squeeze(0)

    emo_embed = emo_token().to(text_embeds.dtype)
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