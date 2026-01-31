import pandas as pd
import numpy as np

timestamps = pd.date_range(
    start="2024-01-01",
    periods=24 * 7,  # 1 week hourly
    freq="h"
)

consumption = (
    20
    + 10 * np.sin(2 * np.pi * timestamps.hour / 24)
    + np.random.normal(0, 2, len(timestamps))
)

df = pd.DataFrame({
    "timestamp": timestamps,
    "consumption_kwh": consumption.round(2)
})

df.to_csv("data/raw/smart_meter.csv", index=False)
print("✅ Generated smart_meter.csv with", len(df), "rows")
