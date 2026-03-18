import argparse
import os

from src.models import get_ner_result
from src.models import get_img_result

from src import NER_MODEL_DIR, RESNET_MODEL_DIR, RESNET_CLASSES_DIR


def main(args):

    try:
        # --- NER ---
        ner_args = argparse.Namespace(
            text=args.text,
            model_path=NER_MODEL_DIR
        )

        animals = get_ner_result(ner_args)

        if not animals:
            print(False)
            return

        # --- CV ---
        cv_args = argparse.Namespace(
            image_path=args.image_path,
            model_path=RESNET_MODEL_DIR,
            classes_path=RESNET_CLASSES_DIR
        )

        image_pred = get_img_result(cv_args)

        # --- Compare ---
        result = any(a.lower() in image_pred.lower() for a in animals)

        print(result)

    except FileNotFoundError as e:
        print(f"File error: {e}")
        print(False)

    except RuntimeError as e:
        print(f"Inference error: {e}")
        print(False)

    except Exception as e:
        print(f"Unexpected error: {e}")
        print(False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        required=True
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True
    )

    args = parser.parse_args()

    # simple validation
    if not os.path.exists(args.image_path):
        print("Image not found")
        print(False)
    else:
        main(args)
