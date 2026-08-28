import torch.nn as nn
from sequence_builder import model, tokenizer, emo_token_id, build_input_embeds


class EmotionHead(nn.Module):
    def __init__(self, hidden_dim=1536, num_classes=7):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, emo_hidden_state):
        return self.classifier(emo_hidden_state.to(self.classifier.weight.dtype))


emotion_head = EmotionHead()


def forward_pass(text, visual_tokens, device):
    inputs_embeds, attention_mask = build_input_embeds(text, visual_tokens, device)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    emo_hidden_state = outputs.hidden_states[-1][:, -1, :]
    emotion_logits = emotion_head(emo_hidden_state)

    return emotion_logits, outputs