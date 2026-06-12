import pandas as pd

print("Loading 2018-2023...")
df_old = pd.read_csv("data/raw/fire_archive_SV-C2_761772.csv", on_bad_lines="skip")
df_old.columns = [c.lower() for c in df_old.columns]
df_old["acq_date"] = pd.to_datetime(df_old["acq_date"])
print(f"  2018-2023: {len(df_old):,} rows")

print("Loading 2024-2026...")
df_new = pd.read_csv("data/raw/filtered_2024_2026.csv", on_bad_lines="skip")
df_new.columns = [c.lower() for c in df_new.columns]
df_new["acq_date"] = pd.to_datetime(df_new["acq_date"])
print(f"  2024-2026: {len(df_new):,} rows")

print("Merging...")
combined = pd.concat([df_old, df_new], ignore_index=True)
combined = combined.sort_values("acq_date").reset_index(drop=True)

print(f"Combined: {len(combined):,} rows")
print(f"Date range: {combined['acq_date'].min()} to {combined['acq_date'].max()}")
print("Year counts:")
print(combined["acq_date"].dt.year.value_counts().sort_index())

combined.to_csv("data/raw/filtered_2018_2025.csv", index=False)
print("Done. Saved to data/raw/filtered_2018_2025.csv")