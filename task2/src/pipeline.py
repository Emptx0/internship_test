import argparse

from models import extract_animal, get_img_result


def main(args):

    # text -> animal
    ner_animals = extract_animal(args.text, ...)

    if len(ner_animals) == 0:
        print(False)
        return

    ner_animal = ner_animals[0]

    # image -> animal
    image_animal = get_img_result(args.image_path)

    result = ner_animal.lower() in image_animal.lower()

    print(result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)

    args = parser.parse_args()

    main(args)
