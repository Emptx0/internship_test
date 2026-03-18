from pathlib import Path


# Base directories

# src
SRC_DIR = Path(__file__).resolve().parent

# Procect root
PROJECT_ROOT = SRC_DIR.parent


# src/data
DATA_DIR = SRC_DIR / "data"

# text dataset files
text_ds_name = "ner_dataset.csv"
synonyms = "synonyms.json"

TEXT_DATA_DIR = DATA_DIR / "text_data"

TEXT_DATASET_PATH = TEXT_DATA_DIR / text_ds_name
SYNONYMS_PATH = TEXT_DATA_DIR / synonyms

# img dataset files
ds_metadata = "metadata.csv"

IMG_DATA_DIR = DATA_DIR / "img_data"

METADATA_PATH = IMG_DATA_DIR / ds_metadata

# CV train / test data dir
TRAIN_IMG_DIR = IMG_DATA_DIR / "train"
TEST_IMG_DIR = IMG_DATA_DIR / "test"


# src/artifacts
ARTIFACTS_DIR = SRC_DIR / "artifacts"

# src/artifacts/ner_model
NER_MODEL_DIR = ARTIFACTS_DIR / "ner_model"

# src/artifacts/resnet_weights + classes.txt
RESNET_MODEL_DIR = ARTIFACTS_DIR / "resnet18_animals.pth"

RESNET_CLASSES_FILE = "classes.txt"

RESNET_CLASSES_DIR = ARTIFACTS_DIR / RESNET_CLASSES_FILE
