import pandas as pd 
import re
import os
import yaml
from pathlib import Path

with open("dataconf.yml", "r") as f:
    config = yaml.safe_load(f)
file_path = os.path.join(config["data_path"], "dum_data.xlsx")

#print(f"File path: {file_path}")

file_path = Path("basedata/dumm_data.xlsx").resolve()
df = pd.read_excel(file_path, skiprows = 0)

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace("<- Robot pos", "")     # Remove the "<- Robot pos"

df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col or 'empty' in col.lower()])

def extract_numvalue(value):
    if isinstance(value, str):
        if "SetVal" in value:
            # Extract value between single quotes after 'v'
            match = re.search(r"v': '(-?\d+\.?\d*)", value)
            if match:
                return float(match.group(1))
        # Try to convert string numbers directly
        try:
            return float(value)
        except ValueError:
            return value
    return value

numeric_columns = df.columns.difference(["Datetime", "Timestamp"])
for col in numeric_columns:
    df[col] = df[col].apply(extract_numvalue)


df['Datetime'] = pd.to_datetime(df["Datetime"])

df = df.dropna(how = "all")

#Reset the index.
df = df.reset_index(drop = True)

print(f"Missing values: \n {df.isnull().sum()}")

print("Basic statistics:\n")

df.describe()