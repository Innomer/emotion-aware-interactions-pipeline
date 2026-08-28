import pandas as pd
import torch
from torch.utils.data import Dataset
from single_video_processing import process_video

EMOTION_LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}


class MELDDataset(Dataset):
    def __init__(self, csv_df, video_dir, context_window=2):
        self.df = csv_df
        self.video_dir = video_dir
        self.context_window = context_window
        self.dialogues = {did: grp.sort_values("Utterance_ID") for did, grp in csv_df.groupby("Dialogue_ID")}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = f'{self.video_dir}\\dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}.mp4'

        dialogue = self.dialogues[row["Dialogue_ID"]]
        context = dialogue[dialogue["Utterance_ID"] < row["Utterance_ID"]].tail(self.context_window)
        lines = [f'{r.Speaker}: {r.Utterance}' for r in context.itertuples()]
        lines.append(f'{row["Speaker"]}: {row["Utterance"]}')
        text_input = "\n".join(lines)

        try:
            visual_features = process_video(video_path)
        except Exception:
            return None

        label = EMOTION_TO_IDX[row["Emotion"]]

        return {"visual_features": visual_features, "text": text_input, "label": label}


def collate_fn(batch):
    return [b for b in batch if b is not None]