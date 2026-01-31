import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load("models/energy_forecast_model.pkl")
print("Model loaded successfully ✅")

# Example input (must match training features)
sample_input = pd.DataFrame([{
    "hour": 10,
    "dayofweek": 2,
    "month": 1,
    "hour_sin": np.sin(2 * np.pi * 10 / 24),
    "hour_cos": np.cos(2 * np.pi * 10 / 24),
    "lag_1": 45.0,
    "lag_2": 44.5,
    "lag_3": 46.2
}])

# Predict
prediction = model.predict(sample_input)
print("Predicted consumption:", prediction[0])
print(model)