# data/data_cleaning.py

import pandas as pd
import os

def clean_data_files(raw_path="data/raw/", clean_path="data/clean/"):
    """
    Cleans all CSV files in the specified raw data directory and saves the cleaned
    versions to the specified clean data directory.
    
    Steps performed:
    - Remove rows with missing dates
    - Convert columns to proper data types
    - Remove rows with invalid or missing prices/volumes
    - Compute daily returns
    - Save cleaned files to target directory
    """
    os.makedirs(clean_path, exist_ok=True)

    raw_files = [f for f in os.listdir(raw_path) if f.endswith('.csv')]
    if not raw_files:
        print("⚠️ No raw CSV files found.")
        return

    for file in raw_files:
        file_path = os.path.join(raw_path, file)
        print(f"🔧 Cleaning {file}...")

        try:
            df = pd.read_csv(file_path)

            # Drop rows with missing dates
            df = df[df['Date'].notna()]

            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)

            # Convert price and volume columns to numeric
            num_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with missing values or zero volume
            df.dropna(subset=num_cols, inplace=True)
            df = df[df['Volume'] > 0]

            # General cleanup
            df.drop_duplicates(inplace=True)
            df.sort_values(by='Date', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Compute daily returns
            df['Return'] = df['Close'].pct_change().fillna(0)

            # Save cleaned file
            clean_file_path = os.path.join(clean_path, file)
            df.to_csv(clean_file_path, index=False)
            print(f"✅ Cleaned file saved: {clean_file_path}")

        except Exception as e:
            print(f"❌ Error while processing {file}: {e}")
