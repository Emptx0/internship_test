"""
Script for generating, preparing, and saving synthetic NER data
for animal name extraction.

! Run this script here: internship_test/task2

The script:
1. Generates synthetic text data using predefined templates.
2. Adds variability using synonyms (e.g., "dog", "puppy", "canine").
3. Labels tokens in BIO format for Named Entity Recognition (NER):
    - B-ANIMAL: beginning of animal entity
    - O: outside entity
4. Saves the dataset in CSV format.

src/data/text_data/
    ner_dataset.csv

CSV columns:
    - tokens: list of tokens (space-separated)
    - ner_tags: list of labels (space-separated)
"""


import os
import random
import pandas as pd

DATA_DIR = "./src/data/text_data"
N_SAMPLES = 5000

animal_classes = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel"
]

synonyms = {
    "dog": ["dog", "puppy", "canine"],
    "cat": ["cat", "kitten", "feline"],
    "horse": ["horse"],
    "cow": ["cow"],
    "sheep": ["sheep", "lamb"],
    "chicken": ["chicken", "hen"],
    "butterfly": ["butterfly"],
    "spider": ["spider"],
    "elephant": ["elephant"],
    "squirrel": ["squirrel"]
}

templates = [
    "There is a {} in the picture",
    "I think it's a {}",
    "Looks like a {}",
    "Could this be a {}?",
    "That might be a {}",
    "Pretty sure it's a {}",
    "This seems like a {}",
    "Is that a {}?",
    "Definitely a {}",
    "Maybe a {}"
]


def generate_sample():
    base_animal = random.choice(list(synonyms.keys()))
    animal = random.choice(synonyms[base_animal])

    text = random.choice(templates).format(animal)

    tokens = text.split()
    labels = []

    for token in tokens:
        if animal in token.lower():
            labels.append("B-ANIMAL")
        else:
            labels.append("O")

    return tokens, labels


def generate_dataset(n_samples):
    data = []

    for _ in range(n_samples):
        tokens, labels = generate_sample()

        data.append({
            "tokens": " ".join(tokens),
            "ner_tags": " ".join(labels)
        })

    return pd.DataFrame(data)


def save_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = generate_dataset(N_SAMPLES)

    save_path = os.path.join(DATA_DIR, "ner_dataset.csv")
    df.to_csv(save_path, index=False)

    print(f"Dataset saved to: {save_path}")
    print(f"Total samples: {len(df)}")


if __name__ == "__main__":
    save_dataset()
