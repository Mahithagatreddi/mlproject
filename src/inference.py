import joblib
import pandas as pd
import numpy as np

def predict_next_hour(consumption_now, lag_1, lag_2, lag_24, timestamp):
    model = joblib.load("models/forecast_model.pkl")

    timestamp = pd.to_datetime(timestamp)
    hour = timestamp.hour
    dayofweek = timestamp.dayofweek
    month = timestamp.month

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    X = [[hour_sin, hour_cos, dayofweek, month, lag_1, lag_2, lag_24]]
    pred = model.predict(X)[0]
    return pred
