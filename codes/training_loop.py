import os
import logging
import time

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_builder import MELDDataset, collate_fn
from clip_to_LLM_embedding import VisualProjector
from model_heads import model, tokenizer, emotion_head
from sequence_builder import build_batch_input_embeds, emo_token
from checkpoint_utils import save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ],
)

logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
print("-"*50)

logger.info(f"Using device: {device}")

base_path = "data/MELD.Raw"
train_csv = pd.read_csv(base_path + "/train_sent_emo.csv", header=0)

logger.info(f"Loaded training CSV with {len(train_csv)} samples")

dataset = MELDDataset(csv_df=train_csv, cache_dir=base_path + "/train_cache")
loader = DataLoader(
    dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0
)

logger.info(f"Dataset size: {len(dataset)}")
logger.info(f"Number of batches per epoch: {len(loader)}")

projector = VisualProjector().to(device).to(model.dtype)
model.to(device)
emotion_head.to(device).to(model.dtype)
emo_token.to(device).to(model.dtype)

trainable_params = (
    list(projector.parameters())
    + list(emotion_head.parameters())
    + list(emo_token.parameters())
    + [p for p in model.parameters() if p.requires_grad]
)
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
ce_loss = torch.nn.CrossEntropyLoss()

logger.info("Models initialized")
logger.info(
    f"Trainable parameters: "
    f"{sum(p.numel() for p in trainable_params if p.requires_grad):,}"
)

CHECKPOINT_DIR = "checkpoints"

training_start = time.time()

for epoch in range(3):
    epoch_loss = 0.0
    num_batches = 0

    epoch_start = time.time()

    logger.info(f"Starting epoch {epoch + 1}/3")

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/3",
        total=len(loader),
        unit="batch",
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(progress_bar):
        if not batch:
            logger.warning(
                f"Epoch {epoch + 1} | Batch {batch_idx + 1}: empty batch skipped"
            )
            continue

        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], device=device)
        visual_features = (
            torch.stack([b["visual_features"] for b in batch])
            .to(device)
            .to(model.dtype)
        )

        visual_tokens = projector(visual_features)
        visual_tokens_list = [visual_tokens[i] for i in range(visual_tokens.shape[0])]

        inputs_embeds, attention_mask = build_batch_input_embeds(
            texts, visual_tokens_list, device
        )

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        emo_hidden_state = outputs.hidden_states[-1][:, -1, :]
        emotion_logits = emotion_head(emo_hidden_state)

        loss = ce_loss(emotion_logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1

        avg_loss = epoch_loss / num_batches

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{avg_loss:.4f}",
        )

    avg_loss = epoch_loss / max(num_batches, 1)

    epoch_time = time.time() - epoch_start
    total_time = time.time() - training_start
    remaining_epochs = 3 - (epoch + 1)
    eta = epoch_time * remaining_epochs

    logger.info(
        f"Epoch {epoch + 1}/3 completed | "
        f"Average Loss: {avg_loss:.4f} | "
        f"Time: {epoch_time / 60:.2f} min | "
        f"Total: {total_time / 60:.2f} min | "
        f"ETA: {eta / 60:.2f} min"
    )

    save_checkpoint(f"{CHECKPOINT_DIR}/epoch_{epoch}", projector, emotion_head, emo_token)

    logger.info(f"Checkpoint saved: {CHECKPOINT_DIR}/epoch_{epoch}")

total_time = time.time() - training_start

logger.info(f"Training completed | Total time: {total_time / 60:.2f} min")