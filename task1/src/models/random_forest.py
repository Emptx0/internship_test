import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

from .mnist_classifier import MnistClassifierInterface


class _RandomForestClassifier(MnistClassifierInterface):
    
    def __init__(
            self, 
            n_estimators: int = 200, 
            random_state: int = 42,
            **kwargs
        ):

        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
        )

    def train(self, X: np.ndarray, y: np.ndarray, verbose=False) -> None:
        
        if verbose:
            logging.info("[RF] Training...")
        
        X = X.reshape(len(X), -1)
        self._model.fit(X / 255.0, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(len(X), -1)
        return self._model.predict(X / 255.0)
    
    def save(self, path: str):
        joblib.dump(self._model, path)

    def load(self, path: str):
        self._model = joblib.load(path)
    
    def __repr__(self):
        return self.__class__.__name__
    