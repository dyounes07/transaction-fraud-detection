import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log-transform Amount — raw dollar amounts are heavily skewed
    df["log_amount"] = np.log1p(df["Amount"])

    # Convert Time (seconds since first transaction) into hour-of-day
    df["hour_of_day"] = (df["Time"] // 3600) % 24

    # Simulate a "time since previous transaction" feature
    df = df.sort_values("Time").reset_index(drop=True)
    df["time_since_last"] = df["Time"].diff().fillna(0)

    return df