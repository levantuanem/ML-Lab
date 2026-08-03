import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def build_features():
    data = pd.read_csv(r'D:\AI_Projects\Project_1\data\processed\diabetes_clean.csv', sep= ';')
    x = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    return x, y