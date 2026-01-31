import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import add_time_features, add_lag_features


def validate_model():
    # 1️⃣ Load model
    model = joblib.load("models/energy_forecast_model.pkl")
    print("Model loaded ✅")

    # 2️⃣ Load & preprocess data
    df = load_data("data/raw/smart_meter.csv")
    df = clean_data(df)

    # 3️⃣ Feature engineering
    df = add_time_features(df)
    df = add_lag_features(df)

    # 4️⃣ Prepare X and y
    X = df.drop(columns=["timestamp", "consumption_kwh"])
    y = df["consumption_kwh"]

    # 5️⃣ Predict
    y_pred = model.predict(X)

    # 6️⃣ Evaluation metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # 7️⃣ Print results
    print("\n📊 MODEL VALIDATION RESULTS")
    print("--------------------------")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


if __name__ == "__main__":
    validate_model()
