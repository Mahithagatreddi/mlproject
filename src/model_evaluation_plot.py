import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import add_time_features, add_lag_features

df = load_data("data/raw/smart_meter.csv")
df = clean_data(df)
df = add_time_features(df)
df = add_lag_features(df)

X = df.drop(columns=["timestamp", "consumption_kwh"])
y = df["consumption_kwh"]

model = joblib.load("models/energy_forecast_model.pkl")
y_pred = model.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred)
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.title("Actual vs Predicted Energy Consumption")
plt.show()
