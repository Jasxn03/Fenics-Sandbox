import os
import pandas as pd
import numpy as np

DATA_DIR = "simulation_results/data"
INDEX_FILE = "simulation_results/index.csv"

index = pd.read_csv(INDEX_FILE)
dfs = []
print(index.columns)
index.columns = index.columns.str.strip()
for _, row in index.iterrows():
    df = pd.read_csv(os.path.join(DATA_DIR, row.file))
    df["height"] = row.height
    df["length"] = row.length
    df["velocity"] = row.velocity
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

full_df.to_csv("simulation_results/stress_full.csv", index=False)
print("end")