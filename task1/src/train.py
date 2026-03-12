import argparse
import logging
import os

from torchvision import datasets

from models.mnist_classifier import MnistClassifier


def main(args):

    logging_level = logging.INFO if args.verbose else logging.ERROR
    logging.basicConfig(level=logging_level,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Load dataset
    try:
        train_dataset = datasets.MNIST(
            root="./src/data",
            train=True,
            download=True
        )

        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()

        logging.info("MNIST dataset loaded successfully.")

    except Exception as e:
        logging.error(f"Failed to load MNIST dataset: {e}")
        raise


    # Initialize classifier
    try:
        clf = MnistClassifier(
            algorithm=args.algorithm,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        logging.info(f"Initialized classifier: {args.algorithm}")

    except Exception as e:
        logging.error(f"Failed to initialize classifier: {e}")
        raise


    # Train model
    try:
        logging.info("Starting training...")
        clf.train(train_images, train_labels, verbose=args.verbose)
        logging.info("Training completed.")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


    # Save model
    try:
        os.makedirs(args.save_path, exist_ok=True)

        ext = "joblib" if args.algorithm == "rf" else "pth"
        filepath = os.path.join(args.save_path, f"{args.algorithm}_model.{ext}")

        clf.save(filepath)

        logging.info(f"Model saved to: {filepath}")

    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", choices=["rf", "nn", "cnn"], required=True)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default="./artifacts")

    args = parser.parse_args()

    main(args)
