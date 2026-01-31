import numpy as np

def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def add_lag_features(df, lags=[1, 2]):
    for lag in lags:
        df[f"lag_{lag}"] = df["consumption_kwh"].shift(lag)

    df = df.dropna()
    return df


