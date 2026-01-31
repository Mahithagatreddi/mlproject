import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import add_time_features, add_lag_features


def train_model():
    data_path = "data/raw/smart_meter.csv"

    # Load & clean
    df = load_data(data_path)
    print("After load:", df.shape)

    df = clean_data(df)
    print("After clean:", df.shape)

    # Feature engineering
    df = add_time_features(df)
    print("After time features:", df.shape)

    df = add_lag_features(df)
    print("After lag features:", df.shape)

    # Target and features
    y = df["consumption_kwh"]
    X = df.drop(columns=["timestamp", "consumption_kwh"])

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) < 5:
        raise ValueError("Not enough data to train the model.")

    # Train-test split (time-series safe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"Model trained successfully ✅")
    print(f"MAE: {mae:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/energy_forecast_model.pkl")
    print("Model saved to models/energy_forecast_model.pkl")


if __name__ == "__main__":
    train_model()
