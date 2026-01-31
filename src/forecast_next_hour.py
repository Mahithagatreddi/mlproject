import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import add_time_features, add_lag_features

MODEL_PATH = "models/energy_forecast_model.pkl"
DATA_PATH = "data/raw/smart_meter.csv"
OUTPUT_DIR = "data/predictions"
OUTPUT_FILE = "next_24_hours_forecast.csv"

def forecast_and_save():
    # 1️⃣ Load model
    model = joblib.load(MODEL_PATH)
    print("Model loaded ✅")

    # 2️⃣ Load & preprocess data
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_time_features(df)
    df = add_lag_features(df, lags=[1, 2])

    last_row = df.iloc[-1].copy()
    current_time = last_row["timestamp"]

    predictions = []

    # 3️⃣ 24-hour forecast loop
    for step in range(1, 25):
        input_df = pd.DataFrame([{
            "hour": last_row["hour"],
            "dayofweek": last_row["dayofweek"],
            "month": last_row["month"],
            "hour_sin": last_row["hour_sin"],
            "hour_cos": last_row["hour_cos"],
            "lag_1": last_row["lag_1"],
            "lag_2": last_row["lag_2"]
        }])

        y_pred = model.predict(input_df)[0]
        current_time += timedelta(hours=1)

        predictions.append({
            "timestamp": current_time,
            "predicted_consumption_kwh": round(y_pred, 2)
        })

        # Update lags
        last_row["lag_2"] = last_row["lag_1"]
        last_row["lag_1"] = y_pred

        # Update time features
        last_row["timestamp"] = current_time
        last_row["hour"] = current_time.hour
        last_row["dayofweek"] = current_time.dayofweek
        last_row["month"] = current_time.month
        last_row["hour_sin"] = np.sin(2 * np.pi * last_row["hour"] / 24)
        last_row["hour_cos"] = np.cos(2 * np.pi * last_row["hour"] / 24)

    forecast_df = pd.DataFrame(predictions)

    # 4️⃣ Create directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # 5️⃣ Save forecast
    forecast_df.to_csv(output_path, index=False)

    print("\n📁 Forecast saved successfully!")
    print(f"📍 Location: {output_path}")
    print(forecast_df.head())

if __name__ == "__main__":
    forecast_and_save()
