import matplotlib.pyplot as plt
import numpy as np

TRADING_DAYS = 252  # Used for annualization


def plot_rolling_sharpe(returns_dict, risk_free_rate=0.02, window=252):
    """
    Plots the rolling Sharpe ratio for each portfolio.

    Parameters:
    - returns_dict: dict[str, pd.Series]
        Dictionary of daily returns per portfolio
    - risk_free_rate: float
        Annualized risk-free rate (default = 2%)
    - window: int
        Rolling window in trading days (default = 1 year)
    """
    plt.figure(figsize=(10, 6))
    for label, ret in returns_dict.items():
        excess = ret - (risk_free_rate / TRADING_DAYS)
        rolling_sharpe = excess.rolling(window=window).mean() / ret.rolling(window=window).std()
        plt.plot(rolling_sharpe, label=label)
    plt.title(f'Rolling Sharpe Ratio ({window} days)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_drawdowns(cum_returns_dict):
    """
    Plots the rolling drawdowns (in %) for each portfolio.

    Parameters:
    - cum_returns_dict: dict[str, pd.Series]
        Dictionary of cumulative returns per portfolio
    """
    plt.figure(figsize=(10, 6))
    for label, cum_ret in cum_returns_dict.items():
        rolling_max = cum_ret.cummax()
        drawdown = cum_ret / rolling_max - 1
        plt.plot(drawdown, label=label)
    plt.title('Rolling Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_correlation(portfolio_dict, benchmark_returns, window=252):
    """
    Plots rolling correlations between each portfolio and the benchmark.

    Parameters:
    - portfolio_dict: dict[str, pd.Series]
        Dictionary of daily returns per portfolio
    - benchmark_returns: pd.Series
        Daily returns of the benchmark
    - window: int
        Rolling window in trading days (default = 1 year)
    """
    plt.figure(figsize=(10, 6))
    for label, port_ret in portfolio_dict.items():
        rolling_corr = port_ret.rolling(window).corr(benchmark_returns)
        plt.plot(rolling_corr, label=f"{label} vs Benchmark")
    plt.title(f'Rolling Correlation ({window} days) with Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_cumulative_performance(cum_returns_dict):
    """
    Prints the cumulative performance over the full period for each portfolio.

    Parameters:
    - cum_returns_dict: dict[str, pd.Series]
        Dictionary of cumulative returns per portfolio
    """
    print("\n📈 Cumulative Performance over the Period:")
    for label, cum_ret in cum_returns_dict.items():
        start_val = cum_ret.iloc[0]
        end_val = cum_ret.iloc[-1]
        perf = (end_val / start_val - 1) * 100
        print(f"  {label}: {perf:.2f}%")
