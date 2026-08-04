from src.models.train import train_models
from src.models.tune import tune_hyperparameters
from src.models.evaluate import evaluate_model
from src.models.predict import predict

def main():
    # Train models
    train_results = train_models()
    print("Training Results:")
    print(train_results)
    # Tune hyperparameters
    best_model = tune_hyperparameters()
    print("Best Model:")
    print(best_model)
    # Evaluate model
    evaluate_model()
    # Predict new data
    predict()

if __name__ == "__main__":
    main()