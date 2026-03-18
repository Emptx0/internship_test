# Task 1 — Image Classification + OOP
## Overview
This project implements three different machine learning models for **MNIST digit classification** using a unified object-oriented interface.

The goal is to:
- Train multiple models with different approaches
- Hide their implementations behind a common interface
- Provide a single entry point for inference

## Architecture
All models implement a shared interface:
```
MnistClassifierInterface
    ├── train()
    └── predict()
```

Three models are implemented:
- Random Forest (`rf`)
- Feed-Forward Neural Network (`nn`)
- Convolutional Neural Network (`cnn`)

A wrapper class: \
`MnistClassifier(algorithm)` \
selects the model and provides a unified API.
## Project Structure
```
.
├── src/
│   ├── models/
│   │   ├── interface.py        # Base interface (train, predict)
│   │   ├── random_forest.py    # Random Forest implementation
│   │   ├── feed_forward.py     # Feed-forward NN
│   │   ├── cnn.py              # CNN model
│   │   └── mnist_classifier.py # Wrapper over all models
│   ├── data/
│   │   └── MNIST/              # Raw MNIST dataset (auto-downloaded)
│   ├── artifacts/              # Trained models
│   │   ├── rf_model.joblib
│   │   ├── nn_model.pth
│   │   └── cnn_model.pth
│   ├── train.py                # Training script
│   └── inference.py            # Unified inference script
│
├── test_img/
│   └── 1.png                   # Test image for inference
│
├── task1_notebook.ipynb        # EDA & experiments
├── requirements.txt
└── README.md
```
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
## Notebook
- `task1_notebook.ipynb` - dataset exploration and model experiments
## Training
Train a specific model:
```
python -m src.train \
    --algorithm cnn \
    --epochs 15 \
    --batch_size 256 \
    --verbose
```
Available options:
```
cnn | nn | rf
```
By default models are saved to:
```
./src/artifacts
```
## Inference
Run prediction using selected algorithm:
```
python -m src.inference --algorithm cnn --image_path "./test_img/1.png"
```
### Output
```
3
```
