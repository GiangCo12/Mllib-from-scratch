import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, random_state: int = None, save_best_option: bool = True) ->  None:
        np.random.seed(random_state)
        self.save_best_choice = save_best_option
        self.intercept = None
        self.coef_ = None

    def __calculate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logits = X@self.theta
        y_pred = 1 / (1+np.exp(-logits))
        dW = X
        dy_pred = y_pred*(1-y_pred)
        return logits, y_pred, dW, dy_pred
    
    def __binary_cross_entropy(self, y: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        len_arr = len(y)
        epsilon = 1e-9
        pos_y_pred = y_pred+epsilon
        neg_y_pred = 1-y_pred+epsilon
        loss = -(y*(np.log(pos_y_pred) + (1-y)*np.log(neg_y_pred)))
        mean_loss = sum(loss) / len_arr

        dy_loss = -(y/pos_y_pred - (1-y)/(neg_y_pred)) / len_arr
        return mean_loss, dy_loss
    
    def __shuffle(self, X: np.ndarray, y: np.ndarray, shape: int) -> tuple[np.ndarray, np.ndarray]:
        shuffle =  np.random.permutation(shape)
        return X[shuffle], y[shuffle]
    
    def fit(self, X: Union[np.ndarray, List[List]], y: Union[np.ndarray, List], learning_rate: float=0.001, epoch: int=1000, batch: int=64) -> None:
        X, y = np.array(X), np.array(y)
        assert np.ndim(X)==2, Exception('the ndim of X must be 2')
        assert np.ndim(y)==2, Exception('ndim of y must be 1')
        assert X.shape[0]==y.shape[0], Exception('x and y must have the same size')

        self.batch = batch
        height, weight = X.shape
        self.theta = np.random.rand(weight, 1)
        loss_history = []
        min_loss = float('inf')
        for _ in range(epoch):
            X, y = self.__shuffle(X, y, height)
            loss_in_1batch = []
            for ith_batch in range(0, height, batch):
                _, y_pred, dW, dy_pred = self.__calculate(X[ith_batch:ith_batch+batch, :])
                loss, dy_loss = self.__binary_cross_entropy(y[ith_batch:ith_batch+batch, :], y_pred)

                loss_in_1batch.append(loss)
                grad = dy_loss*dy_pred*dW
                grad_mean = np.mean(grad, axis=0, keepdims=True).T
                self.theta -= learning_rate*grad_mean
            loss = sum(loss_in_1batch) / len(loss_in_1batch)
            loss_history.append(loss)
            if loss_history[-1] < min_loss:
                best_theta = self.theta
                min_loss = loss_history[-1]

        if self.save_best_choice: self.theta = best_theta

        self.intercept_, self.coef_ = self.theta[0], self.theta[1:]

    def predict(self, X: Union[np.ndarray, List[List]], threshold: float=0.5) -> np.ndarray:
        X = np.array(X)
        assert np.ndim(X)==2, Exception('the ndim of X must be 2')
        _, y_pred, _, _ = self.__calculate(X)
        return np.where(y_pred <= threshold, 0, 1)
    
    def predict_proba(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        X = np.array(X)
        assert np.dim(X)==2, Exception('ndim of X must be 2')

        _, y_pred, _, _ = self.__calculate(X)
        return y_pred
    
    def score(self, X: Union[np.ndarray, List[List]], y: Union[np.ndarray, List], threshold: float=0.5) -> float:
        y_pred = self.predict(X, threshold=threshold)
        return np.mean(y==y_pred)

# Tạo dữ liệu nhị phân với 2 đặc trưng để dễ trực quan
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, random_state=42)

# Thêm bias vào X
X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_with_bias, y.reshape(-1, 1), test_size=0.3, random_state=42)

# Huấn luyện mô hình
model = LogisticRegression()
model.fit(X_train, y_train, learning_rate=0.1, epoch=1000)

# Đánh giá
acc = model.score(X_test, y_test)
print(f"Accuracy: {acc:.2f}")