import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_portfolio_returns(weights, returns_df):
    """
    Computes the daily returns of the portfolio given the weights and asset returns.
    
    Parameters:
    - weights: np.ndarray or pd.Series
        Portfolio weights.
    - returns_df: pd.DataFrame
        Daily returns of the assets.
    
    Returns:
    - pd.Series
        Daily portfolio returns.
    """
    return returns_df @ weights


def compute_performance_stats(portfolio_returns, risk_free_rate=0.0, trading_days=252):
    """
    Computes key performance indicators of a portfolio.
    
    Parameters:
    - portfolio_returns: pd.Series
        Daily portfolio returns.
    - risk_free_rate: float
        Annual risk-free rate used for Sharpe ratio.
    - trading_days: int
        Number of trading days per year (default is 252).
    
    Returns:
    - dict
        Dictionary of performance metrics.
    """
    ann_return = np.mean(portfolio_returns) * trading_days
    ann_vol = np.std(portfolio_returns) * np.sqrt(trading_days)
    sharpe_ratio = (ann_return - risk_free_rate) / ann_vol

    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio
    }


def plot_cumulative_returns(cum_returns_dict, title="Backtest of Optimized Portfolios"):
    """
    Plots the cumulative return curves of multiple portfolios.
    
    Parameters:
    - cum_returns_dict: dict[str, pd.Series]
        Dictionary mapping strategy names to cumulative return series.
    - title: str
        Plot title.
    """
    plt.figure(figsize=(12, 6))
    for label, cum_returns in cum_returns_dict.items():
        plt.plot(cum_returns, label=label)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
