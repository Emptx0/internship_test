from abc import ABC, abstractmethod
import numpy as np


class MnistClassifierInterface(ABC):

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier.

        Parameters
        ----------
        X : np.ndarray
            Input images of shape (N, 28, 28) with pixel values in [0, 255].

        y : np.ndarray
            Target labels of shape (N,) with integer classes 0-9.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input images.

        Parameters
        ----------
        X : np.ndarray
            Input images of shape (N, 28, 28).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (N,) with dtype int.
        """
    
    @abstractmethod
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

    @abstractmethod
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
    