import os
import torch
import pandas as pd
from tqdm import tqdm
from single_video_processing import process_video


def precompute(csv_path, video_dir, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(csv_path, header=0)

    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="precomputing features"):
        cache_path = f"{cache_dir}/dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.pt"
        if os.path.exists(cache_path):
            skipped += 1
            continue

        video_path = (
            f"{video_dir}/dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        )
        try:
            features = process_video(video_path)
        except Exception as e:
            failed += 1
            print("Exception: ", e)
            print(video_path)
            continue

        torch.save(features, cache_path)

    print(
        f"done. total={len(df)} skipped={skipped} failed={failed} cached={len(df) - skipped - failed}"
    )


if __name__ == "__main__":
    base_path = "data/MELD.Raw"
    split = os.getenv("MELD_SPLIT", "train")
    split_paths = {
        "train": (
            "train_sent_emo.csv",
            "train_splits",
            "train_cache",
        ),
        "dev": (
            "dev_sent_emo.csv",
            "dev_splits_complete",
            "dev_cache",
        ),
        "test": (
            "test_sent_emo.csv",
            "output_repeated_splits_test",
            "test_cache",
        ),
    }

    if split not in split_paths:
        raise ValueError(
            f"Unknown MELD_SPLIT={split!r}. "
            f"Choose one of: {', '.join(split_paths)}"
        )

    csv_name, video_dir_name, cache_dir_name = split_paths[split]
    precompute(
        csv_path=base_path + "/" + csv_name,
        video_dir=base_path + "/" + video_dir_name,
        cache_dir=base_path + "/" + cache_dir_name,
    )
