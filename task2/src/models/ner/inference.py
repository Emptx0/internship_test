import argparse
import os
import json
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification

from src import NER_MODEL_DIR, SYNONYMS_PATH


def extract_animal(text, model, tokenizer, id2label):

    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")

    tokens = text.split()

    if len(tokens) == 0:
        return []

    try:
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True
        )
    except Exception as e:
        raise RuntimeError(f"Tokenizer failed: {e}")

    # device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    try:
        word_ids = tokenizer(
            tokens,
            is_split_into_words=True
        ).word_ids()
    except Exception as e:
        raise RuntimeError(f"Failed to align tokens: {e}")

    predicted_labels = []
    prev_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        if word_idx != prev_word_idx:
            label_id = predictions[0][idx].item()
            label = id2label.get(label_id, "O")
            predicted_labels.append((tokens[word_idx], label))

        prev_word_idx = word_idx

    # extract animal
    animals = [
        word for word, label in predicted_labels
        if "ANIMAL" in label
    ]

    return animals


def load_mapping(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Synonyms file not found: {path}")

    with open(path, "r") as f:
        synonyms = json.load(f)

    mapping = {}
    for canonical, words in synonyms.items():
        for w in words:
            mapping[w.lower()] = canonical

    return mapping


def normalize_animals(animals, mapping):
    normalized = []

    for a in animals:
        key = a.lower()
        if key in mapping:
            normalized.append(mapping[key])
        else:
            normalized.append(a)

    return list(set(normalized))


def get_ner_result(args):

    # Load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
        model = AutoModelForTokenClassification.from_pretrained(str(args.model_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to prepare model: {e}")

    if not hasattr(model.config, "id2label"):
        raise RuntimeError("Model config missing id2label")

    id2label = model.config.id2label

    # Inference
    try:
        animals = extract_animal(args.text, model, tokenizer, id2label)
        mapping = load_mapping(SYNONYMS_PATH)
        animals = normalize_animals(animals, mapping)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    print("TEXT:", args.text)
    print("ANIMALS:", animals)

    return animals


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=NER_MODEL_DIR
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True
    )

    args = parser.parse_args()

    get_ner_result(args)
