import plotly.graph_objects as go
import pandas as pd
import numpy as np

TRADING_DAYS = 252

def plot_rolling_sharpe_plotly(returns_dict, risk_free_rate=0.02, window=252):
    fig = go.Figure()
    for label, ret in returns_dict.items():
        excess = ret - (risk_free_rate / TRADING_DAYS)
        rolling_sharpe = excess.rolling(window=window).mean() / ret.rolling(window=window).std()
        fig.add_trace(go.Scatter(x=ret.index, y=rolling_sharpe, mode='lines', name=label))

    fig.update_layout(
        title=f"Rolling Sharpe Ratio ({window} days)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_white"
    )
    return fig

def plot_drawdowns_plotly(cum_returns_dict):
    fig = go.Figure()
    for label, cum_ret in cum_returns_dict.items():
        rolling_max = cum_ret.cummax()
        drawdown = cum_ret / rolling_max - 1
        fig.add_trace(go.Scatter(x=cum_ret.index, y=drawdown, mode='lines', name=label))

    fig.update_layout(
        title="Rolling Drawdowns",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white"
    )
    return fig

def plot_rolling_correlation_plotly(portfolio_dict, benchmark_returns, window=252):
    fig = go.Figure()
    for label, port_ret in portfolio_dict.items():
        rolling_corr = port_ret.rolling(window).corr(benchmark_returns)
        fig.add_trace(go.Scatter(x=port_ret.index, y=rolling_corr, mode='lines', name=f"{label} vs Benchmark"))

    fig.update_layout(
        title=f"Rolling Correlation ({window} days) with Benchmark",
        xaxis_title="Date",
        yaxis_title="Correlation",
        template="plotly_white"
    )
    return fig

def print_cumulative_performance(cum_returns_dict):
    print("\n\U0001F4C8 Cumulative Performance over the Period:")
    for label, cum_ret in cum_returns_dict.items():
        start_val = cum_ret.iloc[0]
        end_val = cum_ret.iloc[-1]
        perf = (end_val / start_val - 1) * 100
        print(f"  {label}: {perf:.2f}%")
