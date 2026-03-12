from .interface import MnistClassifierInterface
from .random_forest import _RandomForestClassifier
from .feed_forward import _FeedForwardClassifier
from .cnn import _CNNClassifier

import numpy as np
import logging

from typing import Literal


_REGISTRY: dict[str, type] = {
    "rf": _RandomForestClassifier,
    "nn": _FeedForwardClassifier,
    "cnn": _CNNClassifier,
}

Algorithm = Literal["cnn", "rf", "nn"]


class MnistClassifier:
    """
    Unified facade for three MNIST classifiers.
 
    Parameters
    ----------
    algorithm : "rf" | "nn" | "cnn"
    **kwargs  : forwarded to the underlying classifier constructor
 
    Examples
    --------
    >>> clf = MnistClassifier(algorithm="cnn")
    >>> clf.train(X_train, y_train)
    >>> labels = clf.predict(X_test)          # np.ndarray  shape (N,)
    >>> probs  = clf.predict_proba(X_test)    # np.ndarray  shape (N, 10)
    """

    def __init__(self, algorithm: Algorithm, **kwargs):
        if algorithm not in _REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Choose from: {list(_REGISTRY)}"
            )
        
        self.algorithm = algorithm
        self._clf: MnistClassifierInterface = _REGISTRY[algorithm](**kwargs)

    def train(self, X: np.ndarray, y: np.ndarray, verbose=False) -> None:
        """
        Parameters
        ----------
        X : np.ndarray shape (N, 28, 28) pixel values in [0, 255]
        y : np.ndarray shape (N,) integer labels  0-9
        verbose : bool
        """
        
        if verbose:
            logging.info(f"[MnistClassifier] Training '{self._clf}' ...\n")

        self._clf.train(X, y, verbose)
        
        if verbose:
            logging.info("[MnistClassifier] Done.\n")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray shape (N, 28, 28)
 
        Returns
        -------
        np.ndarray shape (N,)  dtype int
        """

        return self._clf.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        path : str
            File path where the model will be stored.
            The exact format depends on the underlying implementation
            (e.g. `.pth` for neural networks or `.joblib` for tree models).
        """

        self._clf.save(path)


    def load(self, path: str) -> None:
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        path : str
            File path to the stored model.
            The file must contain a model saved using the corresponding
            `save` method.
        """

        self._clf.load(path)
