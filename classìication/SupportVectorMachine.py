import numpy as np
from typing import List, Union

class SVM:
    def __init__(self, learning_rate=0.001, epochs=1000, lambda_:float=0.01) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_

    def fit(self, X: Union[List[List], np.ndarray], y: Union[List, np.ndarray]) -> None:
        X, y = np.array(X), np.array(y)
        assert np.ndim(X)==2, Exception('ndim of X must be 2')
        assert np.ndim(y)==1, Exception('ndim of y must be 1')

        n_features = X.shape[1]

        y_ = np.where(y<=0, -1, 1)
        self.theta = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            for idx, x_ in enumerate(X):
                condition = y_[idx]*(x_@self.theta - self.b) >= 1
                if condition:
                    self.theta -= self.lr*2*self.lambda_*self.theta
                else:
                    self.theta -= self.lr*(2*self.lambda_*self.theta - y_[idx]*x_)
                    self.b -= self.lr*y_[idx]

    def predict(self, X_test: Union[List[List], np.ndarray]):
        X_test = np.array(X_test)
        assert np.ndim(X_test)==2, Exception('ndim of X_test must be 2')
        approx = X_test@self.theta - self.b
        return np.sign(approx)
    
    def score(self, X_test: Union[List[List], np.ndarray], y_test: Union[List, np.ndarray]) -> float:
        y_test = np.array(y_test)
        assert np.ndim(y_test)==1, Exception('ndim of y_test must be 1')
        y_pred = self.predict(X_test)
        return np.mean(y_test == y_pred)
# Example usage
if __name__ == "__main__":
    X = [[1, 2], [2, 3], [3, 4], [5, 6]]
    y = [1, 1, -1, -1]
    svm = SVM(learning_rate=0.01, epochs=1000)
    svm.fit(X, y)
    print("Coefficients:", svm.theta)
    print("Intercept:", svm.b)
    print("Predictions:", svm.predict([[1, 2], [4, 5]]))
    print("Score:", svm.score(X, y))

        
