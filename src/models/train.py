import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.build_features import build_features
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Set feature and target variables
X, y = build_features()
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # test_size = 0.2 tương đương với train_size =0.8
print(len(x_train), len(x_test), len(y_train), len(y_test))
print(X, y)
# # Chuẩn hóa standardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train, x_test)
joblib.dump(scaler, r"D:\AI_Projects\Project_1\models\scaler.pkl")

cls = SVC() # Khởi tạo model
cls.fit(x_train, y_train) # Huấn luyện

# Dự đoán
y_pred = cls.predict(x_test)
for i, j in zip(y_test, y_pred):
    print(f"Actual: {i}, Predicted: {j}")

# Đánh giá model
print(classification_report(y_test, y_pred))