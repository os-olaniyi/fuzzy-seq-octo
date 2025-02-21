import pandas as pd
import numpy as np
from pathlib import Path


def load_clean(file_path, sheet_name="Sheet1"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # Don't wrap wide tables
    pd.set_option('display.max_rows', 100)       #Max can be overriden (e.g., by df.head(n))

        # Clean the data
    df.columns = df.columns.str.strip()
    #df.columns = df.columns.str.lower()

    expected_columns = [
        "Time", "X", "Y", "Z", "Z_Adjust", "q1", "q2", "q3", "q4", 
        "Layer", "Bead", "Actual_Conductance", "Actual_Power_HotWire",
        "Robot_Wire_Speed", "Pyro_1_Low", "Pyro_3_High", "Wire_Speed",
        "Laser_Power", "AI_ProcadaVoltage", "Control_Z_Send",
        "AI_ProcadaCurrent", "Z_Adj", "TCP_Speed", "MIG_Voltage",
        "MIG_Current", "frame_address", "Anamoly"
    ]

    # Check if all expected columns are present (ignoring extra columns)
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    
    # Keep only the columns we need
    df = df[expected_columns]
    
    # Categorize columns by Data type.
    time_cols = ["Time"]
    categorical_cols = ["Layer", "Bead", "frame_address", "Anamoly"]
    coordinate_cols = ["X", "Y", "Z", "Z_Adjust", "Z_Adj"]
    numeric_cols = [
        "q1", "q2", "q3", "q4", "Actual_Conductance", 
        "Actual_Power_HotWire", "Robot_Wire_Speed", "Pyro_1_Low",
        "Pyro_3_High", "Wire_Speed", "Laser_Power", "AI_ProcadaVoltage",
        "Control_Z_Send", "AI_ProcadaCurrent", "TCP_Speed",
        "MIG_Voltage", "MIG_Current"
    ]

    df["Time"] = pd.to_datetime(df["Time"])
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    for col in numeric_cols + coordinate_cols:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # For categorical columns, fill with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Fill missing numeric values
    for col in numeric_cols + coordinate_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates()
    
    # Step 7: Reset index
    df = df.reset_index(drop=True)

    return df
    

def validate_data(df):
    # Check value ranges for numeric columns
    print("\nValue range checks:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"{col}:")
        print(f"Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")


def find_outliers(df, column, iqr_multiplier=1.5, show_bounds=True):
    """
    Find outliers using the IQR method with additional context
    
    Parameters:
    - df: DataFrame
    - column: str, column name to check
    - iqr_multiplier: float, multiplier for IQR (default 1.5)
    - show_bounds: bool, whether to print the bounds
    
    Returns:
    - int: number of outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    if show_bounds:
        print(f"\n{column} bounds:")
        print(f"Lower bound: {lower_bound:.2f}")
        print(f"Upper bound: {upper_bound:.2f}")
        print(f"IQR: {IQR:.2f}")
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return len(outliers)

def verify_relationships(df):
    # Check correlations between key welding parameters
    key_params = [
        'Actual_Power_HotWire', 'Robot_Wire_Speed', 'Wire_Speed',
        'Laser_Power', 'MIG_Current', 'MIG_Voltage', 'AI_ProcadaCurrent',
        'AI_ProcadaVoltage'
    ]
    correlations = df[key_params].corr()
    
    print("\nCorrelations between key welding parameters:")
    print(correlations)