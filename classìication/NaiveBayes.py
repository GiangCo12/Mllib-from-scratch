import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class naive_bayes:
    class GaussianNB:
        def fit(self, X, y):
            __num_shape, __num_features = X.shape
            self.__classes = np.unique(y)
            __num_class = len(self.__classes)

            self.__mean = np.zeros((__num_class, __num_features))
            self.__var = np.zeros((__num_class, __num_features))
            self.__prior = np.zeros(__num_class)

            for idx, value in enumerate(self.__classes):
                X_value = X[y==value]
                self.__mean[idx, :] = np.mean(X_value, axis=0)
                self.__var[idx, :] = np.var(X_value, axis=0)
                self.__prior[idx] = X_value.shape[0] / X.shape[0]

        def predict(self, X):
            temp_y = []
            for x in X:
                temp_y_th = []
                for idx in range(len(self.__classes)):
                    mean = self.__mean[idx]
                    var = self.__var[idx]
                    gausiance = np.exp(-((x-mean)**2) / (2*var)) / (np.sqrt(2*np.pi*var))
                    gausiance_result = np.sum(np.log(gausiance))
                    prior = np.log(self.__prior[idx])
                    posterior = gausiance_result + prior
                    temp_y_th.append(posterior)
                temp_y.append(np.argmax(temp_y_th))
            return np.array(temp_y)
        
#Dữ liệu ví dụ
data = load_iris()
X = data.data
y = data.target

# Chỉ dùng 2 lớp đầu (cho đơn giản)
X = X[y != 2]
y = y[y != 2]

# Tách tập
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Huấn luyện và dự đoán
model = naive_bayes.GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Kết quả
print("Accuracy:", accuracy_score(y_test, y_pred))