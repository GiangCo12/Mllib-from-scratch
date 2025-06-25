import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Union, List, Tuple
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None) -> None:
        self.feature = feature  
        self.threshold = threshold  
        self.left = left  
        self.right = right
        self.value = value 

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, min_samples_split: Union[int, float]=2, max_depth: int = 100, n_features: Union[int, float]=None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame, List[List]], y: Union[np.ndarray, List]) -> None:
        X, y = np.array(X), np.array(y)
        assert np.ndim(X)==2, Exception("ndim of X must be 2")
        assert np.ndim(y)==1, Exception("ndim of y must be 1")
        if self.n_features:
            self.n_features = min(X.shape[1], self.n_features)
        else:
            self.n_features = X.shape[1]
        self.root = self.__grow_tree(X, y)

    def __grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_node = self.__most_common_label(y)
            return Node(value=leaf_node)

        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self.__best_split(X, y, feature_idxs)

        left_idxs, right_idxs = self.__split(X[:, best_feature], best_threshold)
        left = self.__grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self.__grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)
    
    def __best_split(self, X: np.ndarray, y: np.ndarray, feature_idxs: np.ndarray):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.__information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        return split_idx, split_threshold
    
    def __information_gain(self, y: np.ndarray, X_column: np.ndarray, threshold: Union[int, float]):
        parent_entropy = self.__entropy(y)
        left_idxs, right_idxs = self.__split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.__entropy(y[left_idxs]), self.__entropy(y[right_idxs])
        children_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        return parent_entropy - children_entropy
    
    def __split(self, X_column: np.ndarray, threshold: Union[int, float]):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
    
    def __entropy(self, y: np.ndarray):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])
    
    def __most_common_label(self, y: np.ndarray):
        counter = Counter(y)
        most_common = counter.most_common(1)
        return counter.most_common(1)[0][0]
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, List[List]]) -> np.ndarray:
        X = np.array(X)
        assert np.ndim(X)==2, Exception("ndim of X must be 2")
        return np.array([self.__predict(sample) for sample in X])
    
    def __predict(self, x: np.ndarray):
        return self.__traversal(x, self.root)
    
    def __traversal(self, x: np.ndarray, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.__traversal(x, node.left)
        else:
            return self.__traversal(x, node.right)
        
#Load dữ liệu
iris = load_iris()
X = iris.data[:, :2]  # chỉ lấy 2 feature đầu để vẽ 2D
y = iris.target

# Train model
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Tạo lưới để dự đoán toàn không gian
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid)
Z = Z.reshape(xx.shape)

# Tạo color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['red', 'green', 'blue']

# Vẽ vùng dự đoán
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Vẽ dữ liệu thực tế
for idx, color in enumerate(cmap_bold):
    plt.scatter(X[y == idx, 0], X[y == idx, 1],
                c=color, label=iris.target_names[idx], edgecolor='k')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Decision Boundary của Decision Tree Custom")
plt.legend()
plt.grid(True)
plt.show()