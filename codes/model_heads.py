import torch
import torch.nn as nn
from sequence_builder import model, tokenizer, emo_token, build_input_embeds


class EmotionHead(nn.Module):
    def __init__(self, hidden_dim=1536, num_classes=7):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, emo_hidden_state):
        return self.classifier(emo_hidden_state.to(self.classifier.weight.dtype))


emotion_head = EmotionHead()


def _get_final_norm_module():
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    transformer = getattr(base_model, "model", None)
    return getattr(transformer, "norm", None)


def forward_with_last_hidden(**model_kwargs):
    final_norm = _get_final_norm_module()

    if final_norm is None:
        outputs = model(
            **model_kwargs,
            output_hidden_states=True,
        )
        return outputs, outputs.hidden_states[-1]

    captured = {}

    def capture_last_hidden_state(_module, _inputs, output):
        captured["last_hidden_state"] = output

    handle = final_norm.register_forward_hook(capture_last_hidden_state)

    try:
        outputs = model(
            **model_kwargs,
            output_hidden_states=False,
        )
    finally:
        handle.remove()

    return outputs, captured["last_hidden_state"]


def forward_pass(text, visual_tokens, device):
    inputs_embeds, attention_mask, emo_positions = build_input_embeds(
        text,
        visual_tokens,
        device,
    )

    outputs, last_hidden_state = forward_with_last_hidden(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )

    batch_indices = torch.arange(
        inputs_embeds.shape[0],
        device=device,
    )
    emo_hidden_state = last_hidden_state[batch_indices, emo_positions]
    emotion_logits = emotion_head(emo_hidden_state)

    return emotion_logits, outputs
