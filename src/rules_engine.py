import pandas as pd

def apply_rules(df: pd.DataFrame) -> pd.Series:
    """
    Simple rule-based flags, mimicking a first-layer fraud filter
    used in real payment systems before ML scoring.
    """
    flags = pd.Series(0, index=df.index)

    # Rule 1: unusually large transaction amount
    flags |= (df["Amount"] > df["Amount"].quantile(0.999)).astype(int)

    # Rule 2: transaction happens extremely soon after the previous one
    flags |= (df["time_since_last"] < 1).astype(int)

    return flags