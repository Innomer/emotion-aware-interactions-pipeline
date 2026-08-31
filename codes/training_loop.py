import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
import time

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_builder import MELDDataset, collate_fn
from clip_to_LLM_embedding import VisualProjector
from model_heads import model, emotion_head, forward_with_last_hidden
from sequence_builder import build_generation_batch, emo_token
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
# train_csv = train_csv.sample(n=1000, random_state=42).reset_index(drop=True)

logger.info(f"Loaded training CSV with {len(train_csv)} samples")

MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "0"))
max_train_samples = MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES > 0 else None

dataset = MELDDataset(
    csv_df=train_csv,
    cache_dir=base_path + "/train_cache",
    max_samples=max_train_samples,
)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)

logger.info(f"Dataset size: {len(dataset)}")
if max_train_samples is not None:
    logger.info(f"Training dataset limited to {len(dataset)} valid samples")
logger.info(f"Number of batches per epoch: {len(loader)}")
logger.info(
    f"Batch size: {BATCH_SIZE} | "
    f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS} | "
    f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)

projector = VisualProjector().to(device).to(model.dtype)
model.to(device)
emotion_head.to(device).to(model.dtype)
emo_token.to(device).to(model.dtype)

model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

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
EMOTION_LOSS_WEIGHT = 1.0
GENERATION_LOSS_WEIGHT = 1.0

training_start = time.time()
optimizer.zero_grad(set_to_none=True)

for epoch in range(3):
    epoch_loss = 0.0
    epoch_emotion_loss = 0.0
    epoch_generation_loss = 0.0
    num_batches = 0

    epoch_start = time.time()

    logger.info(f"Starting epoch {epoch + 1}/3")
    model.train()
    projector.train()
    emotion_head.train()
    emo_token.train()

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
        target_texts = [b["target_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], device=device)
        visual_features = (
            torch.stack([b["visual_features"] for b in batch])
            .to(device)
            .to(model.dtype)
        )

        visual_tokens = projector(visual_features)
        visual_tokens_list = [visual_tokens[i] for i in range(visual_tokens.shape[0])]

        inputs_embeds, attention_mask, generation_labels, emo_positions = (
            build_generation_batch(
                texts,
                target_texts,
                visual_tokens_list,
                device,
            )
        )

        outputs, last_hidden_state = forward_with_last_hidden(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=generation_labels,
        )
        batch_indices = torch.arange(inputs_embeds.shape[0], device=device)
        emo_hidden_state = last_hidden_state[batch_indices, emo_positions]
        emotion_logits = emotion_head(emo_hidden_state)

        emotion_loss = ce_loss(emotion_logits.float(), labels)
        generation_loss = outputs.loss
        total_loss = (
            EMOTION_LOSS_WEIGHT * emotion_loss
            + GENERATION_LOSS_WEIGHT * generation_loss
        )

        (total_loss / GRADIENT_ACCUMULATION_STEPS).backward()

        should_step = (
            (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (batch_idx + 1) == len(loader)
        )

        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        epoch_loss += total_loss.item()
        epoch_emotion_loss += emotion_loss.item()
        epoch_generation_loss += generation_loss.item()
        num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_emotion_loss = epoch_emotion_loss / num_batches
        avg_generation_loss = epoch_generation_loss / num_batches
        learning_rate = optimizer.param_groups[0]["lr"]

        progress_bar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            emo=f"{emotion_loss.item():.4f}",
            gen=f"{generation_loss.item():.4f}",
            avg=f"{avg_loss:.4f}",
            avg_emo=f"{avg_emotion_loss:.4f}",
            avg_gen=f"{avg_generation_loss:.4f}",
            lr=f"{learning_rate:.1e}",
            micro_bs=BATCH_SIZE,
            accum=GRADIENT_ACCUMULATION_STEPS,
        )

    avg_loss = epoch_loss / max(num_batches, 1)

    epoch_time = time.time() - epoch_start
    total_time = time.time() - training_start
    remaining_epochs = 3 - (epoch + 1)
    eta = epoch_time * remaining_epochs

    logger.info(
        f"Epoch {epoch + 1}/3 completed | "
        f"Average Loss: {avg_loss:.4f} | "
        f"Emotion Loss: {epoch_emotion_loss / max(num_batches, 1):.4f} | "
        f"Generation Loss: {epoch_generation_loss / max(num_batches, 1):.4f} | "
        f"Time: {epoch_time / 60:.2f} min | "
        f"Total: {total_time / 60:.2f} min | "
        f"ETA: {eta / 60:.2f} min"
    )

    save_checkpoint(f"{CHECKPOINT_DIR}/epoch_{epoch}", projector, emotion_head, emo_token)

    logger.info(f"Checkpoint saved: {CHECKPOINT_DIR}/epoch_{epoch}")

total_time = time.time() - training_start

logger.info(f"Training completed | Total time: {total_time / 60:.2f} min")
