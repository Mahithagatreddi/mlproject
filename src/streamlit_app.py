import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="⚡ Energy Consumption Forecast",
    layout="centered"
)

st.title("⚡ Energy Consumption Forecast")
st.write("Predict next-hour and next-24-hours energy usage using a trained ML model")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = joblib.load("models/energy_forecast_model.pkl")

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.subheader("📥 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input("Select date")
with col2:
    selected_time = st.time_input("Select time")

lag_1 = st.number_input(
    "Energy at previous hour (lag_1)",
    value=100.0
)

lag_2 = st.number_input(
    "Energy at two hours ago (lag_2)",
    value=95.0
)

# Combine date and time
current_dt = datetime.combine(selected_date, selected_time)

# --------------------------------------------------
# Feature Engineering Function
# --------------------------------------------------
def create_features(dt, lag1, lag2):
    hour = dt.hour
    dayofweek = dt.weekday()
    month = dt.month

    return pd.DataFrame([{
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "lag_1": lag1,
        "lag_2": lag2
    }])

# --------------------------------------------------
# Next Hour Prediction
# --------------------------------------------------
st.subheader("⏱ Next Hour Prediction")

if st.button("🔮 Predict Next Hour"):
    input_df = create_features(current_dt, lag_1, lag_2)
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Energy Consumption (Next Hour): **{prediction:.2f} kWh**")

    st.subheader("🛠 Debug Info")
    st.write("Model expects features:")
    st.json(list(input_df.columns))
    st.write("Input DataFrame:")
    st.dataframe(input_df)

# --------------------------------------------------
# Next 24 Hours Forecast
# --------------------------------------------------
st.subheader("📈 Next 24 Hours Forecast")

if st.button("📊 Predict Next 24 Hours"):
    future_predictions = []

    current_lag_1 = lag_1
    current_lag_2 = lag_2

    for i in range(1, 25):
        future_dt = current_dt + timedelta(hours=i)

        feature_df = create_features(
            future_dt,
            current_lag_1,
            current_lag_2
        )

        pred = model.predict(feature_df)[0]

        future_predictions.append({
            "timestamp": future_dt,
            "predicted_consumption": round(pred, 2)
        })

        # Update lags for next step
        current_lag_2 = current_lag_1
        current_lag_1 = pred

    forecast_df = pd.DataFrame(future_predictions)

    st.success("✅ 24-hour forecast generated")
    st.dataframe(forecast_df)

    # Line chart
    st.line_chart(
        forecast_df.set_index("timestamp")["predicted_consumption"]
    )
