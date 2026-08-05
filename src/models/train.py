import pandas as pd
import joblib
from pyparsing import results
from src.features.build_features import build_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

def train_models():
    # =====================================================
    # 1. Load dataset
    # =====================================================
    X, y = build_features()
    # =====================================================
    # 2. Split dataset
    # =====================================================
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # test_size = 0.2 tương đương với train_size =0.8
    # =====================================================
    # 3. Feature Scaling
    # =====================================================
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    joblib.dump(scaler, r"D:\AI_Projects\Project_1\models\scaler.pkl")
    # =====================================================
    # 4. Models
    # =====================================================
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC()
    }
    results = []
    for name, model in models.items():
        print("=" * 60)
        print(name)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(accuracy_score(y_test, y_pred))
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred)})
    results = pd.DataFrame(results)
    results = results.sort_values(by="F1-score",ascending=False)
    return results

if __name__ == "__main__":
    train_models()