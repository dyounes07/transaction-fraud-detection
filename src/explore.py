import pandas as pd
from pathlib import Path

# build a path relative to this script's location
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

# basic shape and info
print("Shape:", df.shape)
print(df.info())
print(df.head())

# check the class imbalance
fraud_counts = df["Class"].value_counts()
fraud_pct = df["Class"].value_counts(normalize=True) * 100
print("\nClass counts:\n", fraud_counts)
print("\nClass percentages:\n", fraud_pct)

# look at amount and time distributions split by fraud vs not
print("\nAmount stats by class:")
print(df.groupby("Class")["Amount"].describe())
