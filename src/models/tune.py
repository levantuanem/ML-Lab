from src.features.build_features import build_features
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def tune_hyperparameters():
    X, y = build_features()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    joblib.dump(scaler, r"D:\AI_Projects\Project_1\models\scaler.pkl")
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]}
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=2,
        verbose=2)
    grid_search.fit(x_train, y_train)
    print("=" * 60)
    print("Best Parameters")
    print(grid_search.best_params_)
    print("Best F1-score")
    print(grid_search.best_score_)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, r"D:\AI_Projects\Project_1\models\random_forest.pkl")
    print("Best model saved successfully!")
    return best_model

if __name__ == "__main__":
    tune_hyperparameters()