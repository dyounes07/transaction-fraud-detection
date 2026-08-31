import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"
df = pd.read_csv(DATA_PATH)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class imbalance bar chart
sns.countplot(x="Class", data=df, ax=axes[0])
axes[0].set_title("Class Distribution (0 = Legit, 1 = Fraud)")
axes[0].set_yscale("log")  # log scale because the imbalance is so extreme

# Amount distribution for fraud vs legit
sns.boxplot(x="Class", y="Amount", data=df, ax=axes[1])
axes[1].set_title("Transaction Amount by Class")
axes[1].set_ylim(0, 500)  # zoom in, there are huge outliers

plt.tight_layout()

# Save using an absolute path so it works no matter where the script is run from
output_path = Path(__file__).resolve().parent.parent / "notebooks" / "class_distribution.png"
plt.savefig(output_path)
plt.show()