"""
Script for downloading, preparing, and indexing the Animals-10 dataset.

Source dataset:
https://huggingface.co/datasets/Rapidata/Animals-10

The script:
1. Downloads the dataset from HuggingFace.
2. Splits it into train and test subsets (90/10).
3. Saves images to disk using the torchvision ImageFolder structure:

data/
    train/
        class_name/
            image.jpg
    test/
        class_name/
            image.jpg

4. Generates a metadata.csv file with the following columns:
    - filepath: path to the saved image
    - label: class name
    - label_id: numeric class id
    - split: dataset split (train/test)
"""


from datasets import load_dataset
import os
from tqdm import tqdm
import csv

from src import IMG_DATA_DIR, METADATA_PATH


dataset = load_dataset("Rapidata/Animals-10")
dataset = dataset["train"].train_test_split(test_size=0.1)

train_ds = dataset["train"]
test_ds = dataset["test"]

labels = train_ds.features["label"].names


def save_split(ds, split_name, metadata_rows):
    for i, sample in enumerate(tqdm(ds, desc=f"Saving {split_name}")):

        img = sample["image"].convert("RGB")
        label_id = sample["label"]
        label = labels[label_id]

        save_dir = os.path.join(IMG_DATA_DIR, split_name, label)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{split_name}_{i}.jpg"
        path = os.path.join(save_dir, filename)

        img.save(path)

        metadata_rows.append({
            "filepath": path,
            "label": label,
            "label_id": label_id,
            "split": split_name
        })


def save_metadata(rows):
    os.makedirs(IMG_DATA_DIR, exist_ok=True)

    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filepath", "label", "label_id", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_dataset():
    metadata_rows = []

    save_split(train_ds, "train", metadata_rows)
    save_split(test_ds, "test", metadata_rows)

    save_metadata(metadata_rows)

    print(f"\nMetadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    save_dataset()
