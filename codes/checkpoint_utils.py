import os
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from model_heads import model, tokenizer


def save_checkpoint(path, projector, emotion_head, emo_token):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    torch.save(projector.state_dict(), os.path.join(path, "projector.pt"))
    torch.save(emotion_head.state_dict(), os.path.join(path, "emotion_head.pt"))
    torch.save(emo_token.state_dict(), os.path.join(path, "emo_token.pt"))


def load_checkpoint(path, projector, emotion_head, emo_token, device):
    safetensors_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        adapter_weights = load_file(safetensors_path, device=str(device))
    else:
        adapter_weights = torch.load(bin_path, map_location=device)

    set_peft_model_state_dict(model, adapter_weights)
    projector.load_state_dict(
        torch.load(os.path.join(path, "projector.pt"), map_location=device)
    )
    emotion_head.load_state_dict(
        torch.load(os.path.join(path, "emotion_head.pt"), map_location=device)
    )
    emo_token.load_state_dict(
        torch.load(os.path.join(path, "emo_token.pt"), map_location=device)
    )
