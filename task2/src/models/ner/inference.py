import argparse
import os
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


def extract_animal(text, model, tokenizer, id2label):

    tokens = text.split()

    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    )

    # device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    word_ids = inputs["input_ids"].new_zeros(inputs["input_ids"].shape).tolist()[0]
    word_ids = tokenizer(
        tokens,
        is_split_into_words=True
    ).word_ids()

    predicted_labels = []
    prev_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        if word_idx != prev_word_idx:
            label_id = predictions[0][idx].item()
            predicted_labels.append((tokens[word_idx], id2label[label_id]))

        prev_word_idx = word_idx

    # extract animal
    animals = [
        word for word, label in predicted_labels
        if "ANIMAL" in label
    ]

    return animals


def main(args):

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    id2label = model.config.id2label

    # Inference
    animals = extract_animal(args.text, model, tokenizer, id2label)

    print("TEXT:", args.text)
    print("ANIMALS:", animals)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(BASE_DIR, "artifacts/ner_model")
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True
    )

    args = parser.parse_args()

    main(args)
    