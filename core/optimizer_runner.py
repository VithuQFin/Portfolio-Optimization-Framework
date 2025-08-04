import numpy as np
from optimization.markowitz import MarkowitzOptimizer
from optimization.risk_parity import RiskParityOptimizer
from optimization.max_diversification import MaxDiversificationOptimizer


def run_optimizations(pivoted, trading_days, risk_free_rate):
    """
    Runs Markowitz optimization to generate the efficient frontier and optimal portfolios.

    Parameters:
    - pivoted: pd.DataFrame of daily asset returns
    - trading_days: int — number of trading days per year
    - risk_free_rate: float — annual risk-free rate

    Returns:
    - mvp: np.ndarray — minimum variance portfolio weights
    - tan: np.ndarray — tangency (maximum Sharpe ratio) portfolio weights
    - expected_returns: pd.Series — annualized expected returns
    - cov_matrix: pd.DataFrame — annualized covariance matrix
    - ret_range: np.ndarray — efficient frontier expected returns
    - vol_range: np.ndarray — efficient frontier volatilities
    """
    expected_returns = pivoted.mean() * trading_days
    cov_matrix = pivoted.cov() * trading_days

    optimizer = MarkowitzOptimizer(expected_returns, cov_matrix, risk_free_rate=risk_free_rate)
    mvp = optimizer.min_variance_portfolio()
    tan = optimizer.tangency_portfolio()
    ret_range, vol_range = optimizer.efficient_frontier()

    return mvp, tan, expected_returns, cov_matrix, ret_range, vol_range


def run_risk_parity_optimization(cov_matrix):
    """
    Runs Risk Parity optimization (Equal Risk Contribution).

    Parameters:
    - cov_matrix: pd.DataFrame — annualized covariance matrix

    Returns:
    - weights: pd.Series — portfolio weights from Risk Parity
    """
    rp_optimizer = RiskParityOptimizer(cov_matrix)
    weights = rp_optimizer.optimize()
    return weights


def run_max_div_optimization(cov_matrix):
    """
    Runs Maximum Diversification portfolio optimization.

    Parameters:
    - cov_matrix: pd.DataFrame — annualized covariance matrix

    Returns:
    - weights: pd.Series — portfolio weights maximizing diversification ratio
    """
    optimizer = MaxDiversificationOptimizer(cov_matrix)
    weights = optimizer.optimize()
    return weights
