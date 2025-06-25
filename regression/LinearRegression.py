import numpy as np
from numpy.linalg import pinv
from typing import List, Union

class LinearRegression:
    def __init__(self, use_gd: bool = False, learning_rate: float = 0.01, epochs: int = 1000):
        """
        Linear Regression using Normal Equation or Gradient Descent
        """
        self.use_gd = use_gd
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._coef = None
        self._inte = None
        self.loss_history = []

    def fit(self, x: Union[List[List], np.ndarray], y: Union[List, np.ndarray]) -> None:
        x, y = np.array(x), np.array(y)
        assert x.ndim == 2, "x must be 2D"
        assert y.ndim == 1, "y must be 1D"
        assert x.shape[0] == len(y), "x and y must have same number of samples"

        ones = np.ones([x.shape[0], 1])
        x = np.concatenate([ones, x], axis=1)

        if not self.use_gd:
            # Normal Equation
            theta = pinv(x.T @ x) @ x.T @ y
            self._inte, self._coef = theta[0], theta[1:]
        else:
            # Gradient Descent
            n_samples, n_features = x.shape
            theta = np.random.randn(n_features)
            self.loss_history = []

            for _ in range(self.epochs):
                y_pred = x @ theta
                error = y_pred - y
                loss = np.mean(error ** 2)
                self.loss_history.append(loss)
                grad = 2 * x.T @ error / n_samples
                theta -= self.learning_rate * grad

            self._inte, self._coef = theta[0], theta[1:]

    def predict(self, x_pred: Union[List[List], np.ndarray]) -> np.ndarray:
        x_pred = np.array(x_pred)
        assert x_pred.ndim == 2, "x_pred must be 2D"
        return np.dot(x_pred, self._coef) + self._inte

    def score(self, x: Union[List[List], np.ndarray], y_true: Union[List, np.ndarray]) -> float:
        y_pred = self.predict(x)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot