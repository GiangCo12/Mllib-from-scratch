from numbers import Number
from typing import List, Callable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from utils.distance import minkowski  
class KNN:
    def __init__(self, k_neighbors: int,
                 p: int = 2,
                 custom_metric: Callable[[List[Number], List[Number]], Number] = None):
        
        assert k_neighbors > 0, "k_neighbors must be greater than 0"
        assert p > 0, "p must be greater than 0"
        self._k_neighbors = k_neighbors
        self._p = p
        self._custom_metric = custom_metric
    
    def fit(self, X: List[List[Number]], y: List[Number]) -> None:
        assert len(X) == len(y), "Datapoints and labels must be equal"
        self._X = np.array(X)
        self._y = np.array(y)

    def _get_max_count_label(self, x: List):
        unique, counts = np.unique(x, return_counts=True)
        return unique[np.argmax(counts)]
    
    def predict(self, X: List[List[Number]]):
        distances = []
        for point in X:
            if self._custom_metric:
                dists = np.array([self._custom_metric(point, train_point) for train_point in self._X])
            else:
                dists = np.apply_along_axis(minkowski, 1, self._X, point, self._p)
            distances.append(dists)

        distances = np.array(distances)
        indices = np.argsort(distances, axis=-1)[:, :self._k_neighbors]
        labels = [[self._y[i] for i in row] for row in indices]
        return [self._get_max_count_label(label) for label in labels]

# Load dữ liệu Iris 
iris = load_iris()
X = iris.data[:, :2]  
y = iris.target
#Khởi tạo
knn = KNN(k_neighbors=5, p=2)
knn.fit(X.tolist(), y.tolist())
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(grid_points.tolist())
Z = np.array(Z).reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel2, alpha=0.8)

#Vẽ dữ liệu 
for class_value in np.unique(y):
    plt.scatter(X[y == class_value, 0], X[y == class_value, 1],
                label=iris.target_names[class_value],
                edgecolor='black', s=40)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Decision Boundary với KNN tự cài")
plt.legend()
plt.grid(True)
plt.show()