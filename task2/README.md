# Task 2 - Named Entity Recognition + Image Classification Pipeline
## Overview
This code implements a machine learning pipeline that combines **Natural Language Processing (NLP)** and **Computer Vision (CV)** to verify whether a textual statement about an animal in an image is true.

The pipeline:
1. Extracts animal entities from user-provided text using a **NER model**
2. Classifies the animal in the image using a **ResNet18 model**
3. Compares both results and returns a **boolean value**

## Pipeline Flow
```
Text → NER → Animal Entity
Image → CNN → Predicted Animal
↓
Comparison → True / False
```

Example:
```
Input:
Text: "There is a cow in the picture"
Image: (image of a cow)

Output:
True
```
## Task 2 Structure
```text
.
├── src/
│   ├── models/
│   │   ├── ner/         # NER model (train + inference)
│   │   └── cv/          # Image classifier (train + inference)
│   ├── data/
│   │   ├── img_data/    # Image dataset (Animals-10, train/test split + metadata.csv)
│   │   └── text_data/   # Synthetic NER dataset (tokens + BIO labels, synonyms)
│   ├── artifacts/       # Trained models and classes
│   ├── load_img_data.py # Download + split + save images + metadata.csv
│   ├── load_ner_data.py # Generate synthetic NER dataset
│   └── pipeline.py      # Main pipeline script
│
├── task2_ner_nb.ipynb      # NER EDA & experiments
├── task2_img_clf_nb.ipynb  # CV EDA & experiments
├── requirements.txt
└── README.md
```

## Dataset

### Image Classification
- 10 animal classes:
    - Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider, Squirrel
- Dataset is automatically downloaded, split, creating `metadata.csv` and saved using `load_img_data.py`
- Train/test split and metadata stored in: `src/data/img_data
### Named Entity Recognition
- Custom dataset for animal entity extraction
- Includes synonym normalization (e.g., _puppy → dog_)
- Dataset is synthetically generated and saved using `load_ner_data.py`
- Files: `src/data/text_data/ner_dataset.csv`, `src/data/text_data/synonyms.json`
## Models
### NER Model
- Transformer-based model (HuggingFace)
- Task: extract animal entities from text
- Output: list of detected animals
### Image Classification Model
- Architecture: ResNet18
- Task: classify animal in the image
- Output: predicted class label
## Notebooks
- `task2_ner_nb.ipynb` - NER data analysis and training experiments
- `task2_img_clf_nb.ipynb` - image dataset exploration and model training
## Installation

```
# create virtual environment
python -m venv venv

# activate it
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```
## Training
### Data Preparation
```
python -m src.load_img_data
python -m src.load_ner_data
```
### NER
```
python -m src.models.ner.train
```
### Image Classification
```
python -m src.models.cv.train
```
## Inference
### NER
```
python -m src.models.ner.inference --text "There is a dog in the image"
```
### CV
```
python -m src.models.cv.inference --image_path path/to/image.jpg
```
## Full Pipeline
Run the complete pipeline:
```
python -m src.pipeline \
    --text "There is a cat in the picture" \
    --image_path "./src/data/img_data/test/Cat/test_1.jpg"
```
### Output
```
True / False
```
