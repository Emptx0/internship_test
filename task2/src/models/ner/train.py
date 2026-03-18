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

from src import TEXT_DATASET_PATH, NER_MODEL_DIR


MODEL_NAME = "distilbert-base-uncased"


def main(args):

    # Check path
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Dataset
    try:
        dataset = load_dataset(
            "csv",
            data_files=str(args.data_path)
        )["train"]
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    # Split
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    # Parse
    def parse(example):
        try:
            example["tokens"] = example["tokens"].split()
            example["ner_tags"] = example["ner_tags"].split()

            if len(example["tokens"]) != len(example["ner_tags"]):
                raise ValueError("Tokens and labels length mismatch")

            return example

        except Exception as e:
            raise ValueError(f"Parsing error: {e}")

    try:
        train_dataset = train_dataset.map(parse)
        val_dataset = val_dataset.map(parse)
    except Exception as e:
        raise RuntimeError(f"Dataset parsing failed: {e}")

    # Labels
    try:
        label_list = sorted(
            list(set(l for ex in train_dataset["ner_tags"] for l in ex))
        )

        label2id = {l: i for i, l in enumerate(label_list)}
        id2label = {i: l for l, i in label2id.items()}
    except Exception as e:
        raise RuntimeError(f"Label processing failed: {e}")

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Tokenizer loading failed: {e}")

    # Tokenization
    def tokenize_and_align_labels(example):
        try:
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

        except Exception as e:
            raise ValueError(f"Tokenization error: {e}")

    try:
        train_dataset = train_dataset.map(tokenize_and_align_labels)
        val_dataset = val_dataset.map(tokenize_and_align_labels)
    except Exception as e:
        raise RuntimeError(f"Tokenization failed: {e}")

    # Model
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id
        )
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

    # Training
    try:
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

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

        trainer.train()

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    # Save
    try:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        raise RuntimeError(f"Saving failed: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default=TEXT_DATASET_PATH
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=NER_MODEL_DIR
    )
    
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    main(args)
    