# core/backtest_runner.py

from backtest import compute_portfolio_returns, compute_performance_stats

def run_backtest(mvp_weights, tan_weights, rp_weights, eq_weights, max_div_weights, pivoted, benchmark_returns):
    port_ret_mvp = compute_portfolio_returns(mvp_weights, pivoted)
    port_ret_tan = compute_portfolio_returns(tan_weights, pivoted)
    port_ret_rp = compute_portfolio_returns(rp_weights, pivoted)
    port_ret_eq = compute_portfolio_returns(eq_weights, pivoted)
    port_ret_maxdiv = compute_portfolio_returns(max_div_weights, pivoted)

    cum_mvp = (1 + port_ret_mvp).cumprod()
    cum_tan = (1 + port_ret_tan).cumprod()
    cum_rp = (1 + port_ret_rp).cumprod()
    cum_eq = (1 + port_ret_eq).cumprod()
    cum_maxdiv = (1 + port_ret_maxdiv).cumprod()
    cum_bench = (1 + benchmark_returns).cumprod()

    stats = {
        "MVP": compute_performance_stats(port_ret_mvp),
        "Tangency": compute_performance_stats(port_ret_tan),
        "Risk Parity": compute_performance_stats(port_ret_rp),
        "Equal Weights": compute_performance_stats(port_ret_eq),
        "Max Diversification": compute_performance_stats(port_ret_maxdiv),
        "Benchmark": compute_performance_stats(benchmark_returns)
    }

    returns_dict = {
        "MVP": port_ret_mvp,
        "Tangency": port_ret_tan,
        "Risk Parity": port_ret_rp,
        "Equal Weights": port_ret_eq,
        "Max Diversification": port_ret_maxdiv,
    }

    cum_returns_dict = {
        "MVP": cum_mvp,
        "Tangency": cum_tan,
        "Risk Parity": cum_rp,
        "Equal Weights": cum_eq,
        "Max Diversification": cum_maxdiv,
        "Benchmark": cum_bench
    }

    return returns_dict, cum_returns_dict, stats
from backtest import compute_portfolio_returns, compute_performance_stats

def run_backtest(mvp_weights, tan_weights, rp_weights, eq_weights, max_div_weights, pivoted, benchmark_returns):
    """
    Executes backtesting for all portfolios using their respective weights and historical asset returns.

    Parameters:
    - mvp_weights: np.ndarray
    - tan_weights: np.ndarray
    - rp_weights: pd.Series
    - eq_weights: np.ndarray
    - max_div_weights: pd.Series
    - pivoted: pd.DataFrame of asset prices
    - benchmark_returns: pd.Series

    Returns:
    - returns_dict: dict of daily returns for each portfolio
    - cum_returns_dict: dict of cumulative returns for each portfolio
    - stats: dict of performance statistics (Sharpe, volatility, etc.)
    """
    # Compute daily returns for each portfolio
    port_ret_mvp = compute_portfolio_returns(mvp_weights, pivoted)
    port_ret_tan = compute_portfolio_returns(tan_weights, pivoted)
    port_ret_rp = compute_portfolio_returns(rp_weights, pivoted)
    port_ret_eq = compute_portfolio_returns(eq_weights, pivoted)
    port_ret_maxdiv = compute_portfolio_returns(max_div_weights, pivoted)

    # Compute cumulative returns
    cum_mvp = (1 + port_ret_mvp).cumprod()
    cum_tan = (1 + port_ret_tan).cumprod()
    cum_rp = (1 + port_ret_rp).cumprod()
    cum_eq = (1 + port_ret_eq).cumprod()
    cum_maxdiv = (1 + port_ret_maxdiv).cumprod()
    cum_bench = (1 + benchmark_returns).cumprod()

    # Compute performance statistics for each strategy
    stats = {
        "MVP": compute_performance_stats(port_ret_mvp),
        "Tangency": compute_performance_stats(port_ret_tan),
        "Risk Parity": compute_performance_stats(port_ret_rp),
        "Equal Weights": compute_performance_stats(port_ret_eq),
        "Max Diversification": compute_performance_stats(port_ret_maxdiv),
        "Benchmark": compute_performance_stats(benchmark_returns)
    }

    # Group daily returns
    returns_dict = {
        "MVP": port_ret_mvp,
        "Tangency": port_ret_tan,
        "Risk Parity": port_ret_rp,
        "Equal Weights": port_ret_eq,
        "Max Diversification": port_ret_maxdiv,
    }

    # Group cumulative returns
    cum_returns_dict = {
        "MVP": cum_mvp,
        "Tangency": cum_tan,
        "Risk Parity": cum_rp,
        "Equal Weights": cum_eq,
        "Max Diversification": cum_maxdiv,
        "Benchmark": cum_bench
    }

    return returns_dict, cum_returns_dict, stats
