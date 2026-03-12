import torch
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import logging

from .mnist_classifier import MnistClassifierInterface


class _CNNClassifier(MnistClassifierInterface):
 
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # block 1
                nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),                              # 14x14
                nn.Dropout2d(0.25),
                # block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),                               # 7x7
                nn.Dropout2d(0.25),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )
 
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))  
        
    def __init__(
            self, 
            epochs: int = 15, 
            batch_size: int = 256, 
            lr: float =1e-3, 
            **kwargs
        ):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._Net().to(self.device)

    def _to_loader(self, X: np.ndarray, y: np.ndarray | None, shuffle: bool) -> DataLoader:
        X_t = torch.tensor(X / 255.0, dtype=torch.float32).unsqueeze(1)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.long)
            return DataLoader(
                TensorDataset(X_t, y_t),
                batch_size=self.batch_size, shuffle=shuffle
                )
        
        return DataLoader(
            TensorDataset(X_t),
            batch_size=self.batch_size, shuffle=False
            )
    
    def train(self, X: np.ndarray, y: np.ndarray, verbose=False) -> None:

        if verbose:
            logging.info(f"[CNN] Device: {self.device}\n")

        loader = self._to_loader(X, y, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            if verbose:
                logging.info(
                    f"[CNN] Epoch {epoch + 1}/{self.epochs}"
                    f"\tloss={total_loss / len(loader):.4f}"
                )
    
    @torch.no_grad()
    def predict(self, X: np.ndarray):
        loader = self._to_loader(X, None, shuffle=False)
        results = []

        self.model.eval()
        for (xb,) in loader:
            xb = xb.to(self.device)
            logits = self.model(xb)
            results.append(torch.softmax(logits, dim=1).cpu())
        
        result_tensor = torch.cat(results).numpy().astype(np.float32)

        return result_tensor.argmax(axis=1).astype(int)
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def __repr__(self):
        return self.__class__.__name__
    