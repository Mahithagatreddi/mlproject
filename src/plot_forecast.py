import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/predictions/next_24_hours_forecast.csv")

print("Columns found:", df.columns)

# Automatically detect prediction column
pred_col = [c for c in df.columns if c != "timestamp"][0]

plt.figure()
plt.plot(df["timestamp"], df[pred_col])
plt.xticks(rotation=45)
plt.title("Next 24 Hours Energy Consumption Forecast")
plt.xlabel("Time")
plt.ylabel("Consumption (kWh)")
plt.tight_layout()
plt.show()
