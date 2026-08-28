import os
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from model_heads import model, tokenizer


def save_checkpoint(path, projector, emotion_head):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(projector.state_dict(), f"{path}\\projector.pt")
    torch.save(emotion_head.state_dict(), f"{path}\\emotion_head.pt")


def load_checkpoint(path, projector, emotion_head, device):
    adapter_weights = load_file(f"{path}\\adapter_model.safetensors")
    set_peft_model_state_dict(model, adapter_weights)
    projector.load_state_dict(torch.load(f"{path}\\projector.pt", map_location=device))
    emotion_head.load_state_dict(torch.load(f"{path}\\emotion_head.pt", map_location=device))