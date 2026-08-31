import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HIDDEN_DIM = 1536
SYSTEM_PROMPT = (
    "You are a English speaking robot companion. Based on the conversation and "
    "the speaker's visible emotional state (the speaker is the one with the "
    "last dialogue), reply briefly and appropriately in one short sentence in "
    "english."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    param.requires_grad = "lora_" in name


class EmoToken(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

    def forward(self):
        return self.embedding


emo_token = EmoToken()


def _build_prompt(text):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _embed_token_ids(input_ids):
    return model.get_input_embeddings()(input_ids).squeeze(0)


def _prepare_visual_tokens(visual_tokens, device, dtype):
    return (
        visual_tokens.to(device)
        .reshape(-1, visual_tokens.shape[-1])
        .to(dtype)
    )


def build_sequence(text, visual_tokens, device):
    prompt = _build_prompt(text)

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(device)

    text_embeds = _embed_token_ids(input_ids)

    emo_embed = emo_token().to(text_embeds.dtype)

    visual_tokens = _prepare_visual_tokens(visual_tokens, device, text_embeds.dtype)

    emo_position = visual_tokens.shape[0] + text_embeds.shape[0]

    sequence = torch.cat(
        [
            visual_tokens,
            text_embeds,
            emo_embed,
        ],
        dim=0,
    )

    return sequence, emo_position


def build_input_embeds(
    text,
    visual_tokens,
    device,
):
    sequence, emo_position = build_sequence(
        text,
        visual_tokens,
        device,
    )

    attention_mask = torch.ones(
        sequence.shape[0],
        dtype=torch.long,
        device=device,
    )

    emo_positions = torch.tensor(
        [emo_position],
        dtype=torch.long,
        device=device,
    )

    return (
        sequence.unsqueeze(0),
        attention_mask.unsqueeze(0),
        emo_positions,
    )


def build_batch_input_embeds(
    texts,
    visual_tokens_list,
    device,
):
    sequences = [
        build_sequence(
            t,
            v,
            device,
        )[0]
        for t, v in zip(
            texts,
            visual_tokens_list,
        )
    ]

    max_len = max(s.shape[0] for s in sequences)

    hidden_dim = sequences[0].shape[-1]

    padded = torch.zeros(
        len(sequences),
        max_len,
        hidden_dim,
        dtype=sequences[0].dtype,
        device=device,
    )

    attention_mask = torch.zeros(
        len(sequences),
        max_len,
        dtype=torch.long,
        device=device,
    )

    for i, seq in enumerate(sequences):

        pad_len = max_len - seq.shape[0]

        padded[i, pad_len:] = seq

        attention_mask[i, pad_len:] = 1

    return padded, attention_mask


def build_generation_batch(
    texts,
    targets,
    visual_tokens_list,
    device,
):
    input_sequences = []
    target_token_ids = []
    emo_positions = []

    for text, target, visual_tokens in zip(
        texts,
        targets,
        visual_tokens_list,
    ):

        prompt = _build_prompt(text)

        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to(device)

        target_ids = tokenizer(
            target,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)

        prompt_embeds = _embed_token_ids(prompt_ids)

        target_embeds = _embed_token_ids(target_ids)

        emo_embed = emo_token().to(prompt_embeds.dtype)

        visual_tokens = _prepare_visual_tokens(
            visual_tokens,
            device,
            prompt_embeds.dtype,
        )

        emo_position = visual_tokens.shape[0] + prompt_embeds.shape[0]

        sequence = torch.cat(
            [
                visual_tokens,
                prompt_embeds,
                emo_embed,
                target_embeds,
            ],
            dim=0,
        )

        input_sequences.append(sequence)

        target_token_ids.append(target_ids.squeeze(0))
        emo_positions.append(emo_position)

    max_len = max(seq.shape[0] for seq in input_sequences)

    hidden_dim = input_sequences[0].shape[-1]

    padded = torch.zeros(
        len(input_sequences),
        max_len,
        hidden_dim,
        dtype=input_sequences[0].dtype,
        device=device,
    )

    attention_mask = torch.zeros(
        len(input_sequences),
        max_len,
        dtype=torch.long,
        device=device,
    )

    labels = torch.full(
        (len(input_sequences), max_len),
        -100,
        dtype=torch.long,
        device=device,
    )

    for i, seq in enumerate(input_sequences):

        pad_len = max_len - seq.shape[0]

        padded[i, pad_len:] = seq
        attention_mask[i, pad_len:] = 1

        target_len = target_token_ids[i].shape[0]

        labels[i, max_len - target_len :] = target_token_ids[i]
        emo_positions[i] += pad_len

    return (
        padded,
        attention_mask,
        labels,
        torch.tensor(
            emo_positions,
            dtype=torch.long,
            device=device,
        ),
    )
