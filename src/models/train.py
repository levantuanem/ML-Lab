import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.build_features import build_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LosgisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

X_scaled, y = build_features()
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)