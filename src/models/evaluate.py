import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.build_features import build_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)

def evaluate_model():
    # Chuẩn bị dữ liệu
    X, y = build_features()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = joblib.load(r"D:\AI_Projects\Project_1\models\scaler.pkl")
    x_test = scaler.transform(x_test)
    # Load model
    model = joblib.load(r"D:\AI_Projects\Project_1\models\random_forest.pkl")
    # Predict
    y_pred = model.predict(x_test)
    # Evaluate
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(r"D:\AI_Projects\Project_1\reports\figures\confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model()