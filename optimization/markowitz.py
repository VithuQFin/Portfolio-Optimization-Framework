import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MarkowitzOptimizer:
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.0):
        """
        Initializes the Markowitz Optimizer.

        Parameters:
        - expected_returns: pd.Series
            Expected annualized returns for each asset
        - cov_matrix: pd.DataFrame
            Annualized covariance matrix of asset returns
        - risk_free_rate: float
            Annual risk-free rate used for Sharpe ratio calculation
        """
        self.mu = expected_returns
        self.Sigma = cov_matrix
        self.rf = risk_free_rate
        self.num_assets = len(expected_returns)

    def _portfolio_performance(self, weights):
        """
        Computes expected return and volatility of a given portfolio.

        Returns:
        - expected_return: float
        - volatility: float
        """
        ret = np.dot(weights, self.mu)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.Sigma, weights)))
        return ret, vol

    def _check_weights(self, weights):
        """Checks if portfolio weights sum to 1."""
        return np.isclose(np.sum(weights), 1.0)

    def min_variance_portfolio(self, short_allowed=False):
        """
        Computes the Minimum Variance Portfolio (MVP).

        Parameters:
        - short_allowed: bool — Whether short selling is allowed

        Returns:
        - weights: np.ndarray
        """
        init_guess = np.ones(self.num_assets) / self.num_assets
        bounds = None if short_allowed else [(0.0, 1.0)] * self.num_assets
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        def portfolio_volatility(w):
            return np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))

        result = minimize(portfolio_volatility, init_guess,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)

        return result.x if result.success else None

    def tangency_portfolio(self, short_allowed=False):
        """
        Computes the Tangency Portfolio (max Sharpe ratio).

        Parameters:
        - short_allowed: bool — Whether short selling is allowed

        Returns:
        - weights: np.ndarray
        """
        init_guess = np.ones(self.num_assets) / self.num_assets
        bounds = None if short_allowed else [(0.0, 1.0)] * self.num_assets
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        def neg_sharpe_ratio(w):
            ret, vol = self._portfolio_performance(w)
            return - (ret - self.rf) / vol

        result = minimize(neg_sharpe_ratio, init_guess,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)

        return result.x if result.success else None

    def efficient_frontier(self, return_range=None, short_allowed=False, points=50):
        """
        Computes the Efficient Frontier.

        Parameters:
        - return_range: np.ndarray or None — Range of target returns
        - short_allowed: bool — Whether short selling is allowed
        - points: int — Number of points along the frontier

        Returns:
        - return_range: np.ndarray
        - volatility_range: np.ndarray
        """
        if return_range is None:
            min_ret = self._portfolio_performance(self.min_variance_portfolio())[0]
            max_ret = self._portfolio_performance(self.tangency_portfolio())[0]
            return_range = np.linspace(min_ret, max_ret, points)

        init_guess = np.ones(self.num_assets) / self.num_assets
        bounds = None if short_allowed else [(0.0, 1.0)] * self.num_assets

        frontier_vols = []

        for target_ret in return_range:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, self.mu) - target_ret}
            ]

            result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(self.Sigma, w))),
                              init_guess,
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)

            if result.success:
                vol = np.sqrt(np.dot(result.x.T, np.dot(self.Sigma, result.x)))
                frontier_vols.append(vol)
            else:
                frontier_vols.append(np.nan)

        return return_range, np.array(frontier_vols)
