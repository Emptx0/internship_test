import argparse
import os
import torch

from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

from src import RESNET_MODEL_DIR, RESNET_CLASSES_DIR


def load_classes(path):
    with open(path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def predict_image(image_path, model, transform, device, classes):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

    pred = outputs.argmax(dim=1).item()

    return classes[pred]


def get_img_result(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = load_classes(args.classes_path)
    num_classes = len(classes)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    pred = predict_image(
        args.image_path,
        model,
        transform,
        device,
        classes
    )

    print("IMAGE:", args.image_path)
    print("PREDICTION:", pred)

    return str(pred)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=RESNET_MODEL_DIR
    )

    parser.add_argument(
        "--classes_path",
        type=str,
        default=RESNET_CLASSES_DIR
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True
    )

    args = parser.parse_args()

    get_img_result(args)
    