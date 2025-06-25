import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from regression.LinearRegression import LinearRegression

st.title("📊 Linear Regression Dashboard")
st.write("## Nhập dữ liệu đầu vào")

# Cách nhập
option = st.radio("Chọn cách nhập dữ liệu", ("Tự tạo dữ liệu", "Tải file CSV"))

if option == "Tự tạo dữ liệu":
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2.5 * X.flatten() + 3 + np.random.randn(50) * 2
else:
    uploaded_file = st.file_uploader("Tải file CSV với 2 cột: X, y", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        if "X" in df.columns and "y" in df.columns:
            X = df["X"].values.reshape(-1, 1)
            y = df["y"].values
        else:
            st.error("File CSV phải có cột 'X' và 'y'")
            st.stop()
    else:
        st.stop()

# Tùy chỉnh
use_gd = st.checkbox("Dùng Gradient Descent (GD)?", value=True)
lr = st.slider("Learning rate", 0.001, 1.0, 0.01)
epochs = st.slider("Số epoch", 10, 5000, 1000)

# Huấn luyện
model = LinearRegression(use_gd=use_gd, learning_rate=lr, epochs=epochs)
model.fit(X, y)
y_pred = model.predict(X)

# Kết quả
st.write("## Kết quả hồi quy")
st.write(f"**Hệ số (a):** {model._coef}")
st.write(f"**Intercept (b):** {model._inte}")
st.write(f"**Công thức mô hình:**  \n $\\hat{{y}} = {model._coef[0]:.3f} \\cdot X + {model._inte:.3f}$")
st.write(f"**R² score:** {model.score(X, y):.4f}")

# Vẽ đường hồi quy
fig, ax = plt.subplots()
ax.scatter(X, y, color="blue", label="Dữ liệu gốc")
ax.plot(X, y_pred, color="red", label="Dự đoán")
ax.set_title("Biểu đồ hồi quy tuyến tính")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()
st.pyplot(fig)

# Vẽ loss history nếu dùng GD
if use_gd:
    st.write("## Biểu đồ Loss trong quá trình học")
    fig2, ax2 = plt.subplots()
    ax2.plot(model.loss_history)
    ax2.set_title("Loss history")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    st.pyplot(fig2)