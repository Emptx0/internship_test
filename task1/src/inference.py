import argparse
import logging
import os

from torchvision import datasets
import numpy as np

from models.mnist_classifier import MnistClassifier


def main(args):

    logging_level = logging.INFO if args.verbose else logging.ERROR
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Load dataset
    try:
        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True
        )

        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()

        logging.info("MNIST test dataset loaded successfully.")

    except Exception as e:
        logging.error(f"Failed to load MNIST dataset: {e}")
        raise


    # Initialize classifier
    try:
        clf = MnistClassifier(
            algorithm=args.algorithm
        )

        logging.info(f"Initialized classifier: {args.algorithm}")

    except Exception as e:
        logging.error(f"Failed to initialize classifier: {e}")
        raise


    # Load model
    try:
        ext = "joblib" if args.algorithm == "rf" else "pth"
        filepath = os.path.join(args.model_path, f"{args.algorithm}_model.{ext}")

        clf.load(filepath)

        logging.info(f"Model loaded from: {filepath}")

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


    # Run inference
    try:
        logging.info("Running inference...")

        predictions = clf.predict(test_images)

        accuracy = (predictions == test_labels).mean()

        logging.info(f"Inference completed.")
        logging.info(f"Accuracy: {accuracy:.4f}")

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", choices=["rf", "nn", "cnn"], required=True)

    parser.add_argument("--model_path", type=str, default="./artifacts")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    main(args)
    