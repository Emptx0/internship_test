import argparse
import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

MODEL_NAME = "distilbert-base-uncased"


def main(args):

    # Load dataset (CSV)
    dataset = load_dataset(
        "csv",
        data_files=args.data_path
    )["train"]

    # Split
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    # Parse tokens + labels
    def parse(example):
        example["tokens"] = example["tokens"].split()
        example["ner_tags"] = example["ner_tags"].split()
        return example

    train_dataset = train_dataset.map(parse)
    val_dataset = val_dataset.map(parse)

    # Labels
    label_list = sorted(
        list(set(l for ex in train_dataset["ner_tags"] for l in ex))
    )

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize + align
    def tokenize_and_align_labels(example):
        tokenized = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True
        )

        word_ids = tokenized.word_ids()
        labels = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != prev_word_idx:
                labels.append(label2id[example["ner_tags"][word_idx]])
            else:
                labels.append(-100)

            prev_word_idx = word_idx

        tokenized["labels"] = labels
        return tokenized

    train_dataset = train_dataset.map(tokenize_and_align_labels)
    val_dataset = val_dataset.map(tokenize_and_align_labels)

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        report_to=[],
        disable_tqdm=not args.verbose
    )

    # Trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(BASE_DIR, "data/text_data/ner_dataset.csv")
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(BASE_DIR, "artifacts/ner_model")
    )
    
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    main(args)
    