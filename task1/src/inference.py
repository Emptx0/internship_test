import argparse
import logging
import os

import numpy as np
from PIL import Image

from src.models.mnist_classifier import MnistClassifier


def preprocess_image(image_path, algorithm):
    try:
        image = Image.open(image_path).convert("L")  # grayscale
        image = image.resize((28, 28))

        image = np.array(image) / 255.0  # normalize

        if algorithm in ["rf", "nn"]:
            image = image.reshape(1, -1)  # flatten
        else:  # cnn
            image = image.reshape(1, 28, 28)

        return image

    except Exception as e:
        raise RuntimeError(f"Failed to preprocess image: {e}")


def main(args):
    if not os.path.exists(args.image_path):
        raise RuntimeError("Image not found")

    try:
        # preprocess
        image = preprocess_image(args.image_path, args.algorithm)

        # init model
        clf = MnistClassifier(algorithm=args.algorithm)

        # load model
        ext = "joblib" if args.algorithm == "rf" else "pth"
        filepath = os.path.join(args.model_path, f"{args.algorithm}_model.{ext}")

        clf.load(filepath)

        # predict
        pred = clf.predict(image)

        # print only label
        print(int(pred[0]))

    except Exception as e:
        raise RuntimeError(f"Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", choices=["rf", "nn", "cnn"], required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./src/artifacts")

    args = parser.parse_args()

    main(args)
