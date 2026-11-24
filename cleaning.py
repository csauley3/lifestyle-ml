import pandas as pd


# -- Cleaning --

#Load dataset
df = pd.read_csv("heart_2022_with_nans.csv", na_values=["?", " ", "", "NA", "null", "NaN" "N/A"])
print(f"Original df.shape: {df.shape}")

#remove na vlaues and duplicates
df = df.dropna().drop_duplicates()
print(f"Cleaned df.shape: {df.shape}")
df.to_csv("heart_2022_no_nans.csv", index=False)

