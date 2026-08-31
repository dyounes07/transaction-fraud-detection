from pathlib import Path

import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

# Use one row as an example transaction
transaction = df.drop(columns=["Class"]).iloc[0].to_dict()

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=transaction
)

print("Status code:", response.status_code)
print("Response:", response.json())
