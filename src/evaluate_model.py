import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import add_time_features, add_lag_features

def evaluate_model():
    # Load trained model
    model = joblib.load("models/energy_forecast_model.pkl")
    print("Model loaded ✅")

    # Load and preprocess data
    df = load_data("data/raw/smart_meter.csv")
    df = clean_data(df)
    df = add_time_features(df)
    df = add_lag_features(df, lags=[1, 2])

    # Split features and target
    X = df.drop(columns=["timestamp", "consumption_kwh"])
    y = df["consumption_kwh"]

    # Predict
    y_pred = model.predict(X)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y.values, label="Actual Consumption")
    plt.plot(y_pred, label="Predicted Consumption")
    plt.title("Actual vs Predicted Energy Consumption")
    plt.xlabel("Time Index")
    plt.ylabel("Consumption (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Evaluation completed 📊")

if __name__ == "__main__":
    evaluate_model()
