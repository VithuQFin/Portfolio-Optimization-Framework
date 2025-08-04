import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_multiple_tickers(tickers, start_date, end_date, raw_path="data/raw/"):
    """
    Downloads historical market data for a list of tickers using Yahoo Finance.

    Parameters:
    - tickers (list of str): List of stock symbols to download
    - start_date (str or datetime): Start date (format: 'YYYY-MM-DD')
    - end_date (str or datetime): End date (format: 'YYYY-MM-DD')
    - raw_path (str): Path to save the raw CSV files (default: "data/raw/")

    Returns:
    - list of DataFrames: The downloaded and formatted data for each ticker
    """
    # Validate date inputs
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start >= end:
            raise ValueError("Start date must be before end date.")
        if end > datetime.now():
            raise ValueError("End date cannot be in the future.")
    except ValueError as e:
        raise ValueError(f"Invalid date format or error: {e}")

    os.makedirs(raw_path, exist_ok=True)
    all_data = []

    for ticker in tickers:
        try:
            print(f"🔄 Downloading {ticker} from {start_date} to {end_date}...")
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

            if not data.empty:
                data.reset_index(inplace=True)
                data['Ticker'] = ticker

                # Ensure expected columns exist
                if 'Adj Close' not in data.columns:
                    data['Adj Close'] = data['Close']  # fallback if missing

                data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']]

                # Format filename using date range
                start_str = start.strftime("%Y%m%d")
                end_str = end.strftime("%Y%m%d")
                file_name = f"{ticker}_{start_str}_to_{end_str}.csv"
                file_path = os.path.join(raw_path, file_name)
                data.to_csv(file_path, index=False)

                print(f"✅ Data saved: {file_path}")
                all_data.append(data)
            else:
                print(f"⚠️ No data found for {ticker}")

        except Exception as e:
            print(f"❌ Error while downloading {ticker}: {e}")

    return all_data
