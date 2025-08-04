from data.data_loader import fetch_multiple_tickers
from data.data_cleaning import clean_data_files
import os
import pandas as pd

def load_assets_and_benchmark(tickers, benchmark, start, end,
                              raw_path, clean_path, bench_raw_path, bench_clean_path):
    """
    Loads and processes both asset and benchmark data: 
    downloads, cleans, and prepares them for optimization and backtesting.

    Parameters:
    - tickers: list of str — Asset tickers
    - benchmark: str — Benchmark index (e.g., "^GSPC")
    - start, end: str — Date range (e.g., "2010-01-01")
    - raw_path, clean_path: str — Directories for asset data
    - bench_raw_path, bench_clean_path: str — Directories for benchmark data

    Returns:
    - pivoted: pd.DataFrame — Daily returns of all assets, indexed by date
    - benchmark_returns: pd.Series — Benchmark daily returns, indexed by date
    """
    start_str = pd.to_datetime(start).strftime("%Y%m%d")
    end_str = pd.to_datetime(end).strftime("%Y%m%d")

    # 1. Download and clean asset data
    fetch_multiple_tickers(tickers, start, end, raw_path)
    clean_data_files(raw_path, clean_path)

    # 2. Download and clean benchmark data
    fetch_multiple_tickers([benchmark], start, end, bench_raw_path)
    clean_data_files(bench_raw_path, bench_clean_path)

    # 3. Load benchmark returns
    bench_file = f"{benchmark}_{start_str}_to_{end_str}.csv"
    df_benchmark = pd.read_csv(os.path.join(bench_clean_path, bench_file))
    df_benchmark['Date'] = pd.to_datetime(df_benchmark['Date'])
    df_benchmark.sort_values('Date', inplace=True)
    df_benchmark['Return'] = df_benchmark['Close'].pct_change().fillna(0)
    benchmark_returns = df_benchmark.set_index('Date')['Return']

    # 4. Load and merge cleaned asset returns
    clean_files = [
        f for f in os.listdir(clean_path)
        if f.endswith('.csv') and f"_{start_str}_to_{end_str}.csv" in f
    ]
    if not clean_files:
        raise FileNotFoundError("❌ No cleaned files found for the selected tickers.")

    dfs = [pd.read_csv(os.path.join(clean_path, f)) for f in clean_files]
    merged = pd.concat(dfs)
    pivoted = merged.pivot(index='Date', columns='Ticker', values='Return')
    pivoted.dropna(inplace=True)
    pivoted.index = pd.to_datetime(pivoted.index)

    return pivoted, benchmark_returns
