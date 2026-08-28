import torch
from single_video_processing import process_video
from clip_to_LLM_embedding import VisualProjector
from model_heads import model, tokenizer, emotion_head, forward_pass
from sequence_builder import build_input_embeds
from checkpoint_utils import load_checkpoint

EMOTION_LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
projector = VisualProjector().to(device).to(model.dtype)

CHECKPOINT_PATH = r"D:\emotion-aware-interactions-pipeline\checkpoints\epoch_2"
load_checkpoint(CHECKPOINT_PATH, projector, emotion_head, device)


def run_inference(video_path, text, max_new_tokens=40):
    model.eval()

    visual_features = process_video(video_path).to(device)
    visual_tokens = projector(visual_features)

    emotion_logits, _ = forward_pass(text, visual_tokens, device)
    predicted_label = EMOTION_LABELS[emotion_logits.argmax(dim=-1).item()]

    inputs_embeds, attention_mask = build_input_embeds(text, visual_tokens, device)
    generated_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return predicted_label, response


if __name__ == "__main__":
    label, response = run_inference(
        video_path=r"D:\emotion-aware-interactions-pipeline\data\MELD.Raw\dev_splits_complete\dia0_utt0.mp4",
        text="Ross: Sure, whatever you say.",
    )
    print("predicted emotion:", label)
    print("generated response:", response)
