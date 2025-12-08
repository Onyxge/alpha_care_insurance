import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """
    Loads insurance data using the pipe symbol (|) as a separator.
    """
    # 1. Load Data with Pipe Separator
    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")

        # --- 2. LOAD DATA ---
    try:
        # We explicitly set sep='|' because the data shows "Field1|Field2|Field3"
        df = pd.read_csv(filepath, sep='|', low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # 2. STANDARDIZE COLUMN NAMES
    # Strip spaces (e.g., "TotalPremium " -> "TotalPremium")
    df.columns = df.columns.str.strip()

    # Normalize common inconsistent casing from the raw data
    rename_map = {
        'make': 'Make',
        'model': 'Model',
        'cubiccapacity': 'CubicCapacity',
        'kilowatts': 'Kilowatts',
        'bodytype': 'BodyType',
        'mmcode': 'Mmcode',
        # Ensure target columns are standard
        'Total_Premium': 'TotalPremium',
        'Total_Claims': 'TotalClaims',
        'Transaction_Month': 'TransactionMonth'
    }
    df.rename(columns=rename_map, inplace=True)

    # 3. VERIFY CRITICAL COLUMNS
    required = ['TotalClaims', 'TotalPremium', 'TransactionMonth', 'Province', 'VehicleType']
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"CRITICAL WARNING: Missing columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        return df  # Return whatever we have so we can inspect it

    # 4. PARSE DATES
    # We explicitly handle the format '2015-07-01 00:00:00' to avoid warnings
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(
            df['TransactionMonth'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )

    if 'VehicleIntroDate' in df.columns:
        # Handle format '6/2002' (Month/Year) explicitly to avoid warnings
        df['VehicleIntroDate'] = pd.to_datetime(
            df['VehicleIntroDate'],
            format='%m/%Y',
            errors='coerce'
        )

    # 5. CLEAN NUMERIC COLUMNS
    numeric_cols = [
        'TotalPremium', 'TotalClaims',
        'CalculatedPremiumPerTerm',
        'SumInsured',
        'Cylinders', 'CubicCapacity',
        'Kilowatts', 'CustomValueEstimate',
        'CapitalOutstanding'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 6. CLEAN CATEGORICAL COLUMNS
    cat_cols = ['PostalCode', 'Mmcode', 'PolicyID', 'UnderwrittenCoverID']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df