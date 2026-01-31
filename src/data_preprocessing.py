import pandas as pd
import os

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at path: {path}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clean_data(df):
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # Hourly resampling (correct frequency)
    df = df.resample("h").mean()

    # Fill missing consumption values
    df["consumption_kwh"] = df["consumption_kwh"].interpolate(method="time")

    df = df.reset_index()
    return df
