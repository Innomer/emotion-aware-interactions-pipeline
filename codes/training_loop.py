import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset_builder import MELDDataset, collate_fn
from clip_to_LLM_embedding import VisualProjector
from model_heads import model, emotion_head, forward_pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = r"D:\emotion-aware-interactions-pipeline\data\MELD.Raw"
train_csv = pd.read_csv(base_path + r"\train_sent_emo.csv", header=0)

dataset = MELDDataset(csv_df=train_csv, video_dir=base_path + r"\train_splits")
loader = DataLoader(
    dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0
)

projector = VisualProjector().to(device)
model.to(device)
emotion_head.to(device)

trainable_params = (
    list(projector.parameters())
    + list(emotion_head.parameters())
    + [p for p in model.parameters() if p.requires_grad]
)
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
ce_loss = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    for batch in loader:
        for sample in batch:
            visual_tokens = projector(sample["visual_features"].to(device))
            label = torch.tensor([sample["label"]], device=device)

            emotion_logits, outputs = forward_pass(
                sample["text"], visual_tokens, device
            )

            loss = ce_loss(emotion_logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"epoch {epoch} loss {loss.item()}")
