import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MaxDiversificationOptimizer:
    def __init__(self, cov_matrix):
        """
        Initializes the Max Diversification optimizer.

        Parameters:
        - cov_matrix: pd.DataFrame
            Annualized covariance matrix of asset returns
        """
        self.Sigma = cov_matrix
        self.assets = cov_matrix.columns
        self.n = len(self.assets)
        self.sigma_individual = np.sqrt(np.diag(cov_matrix))

    def _portfolio_volatility(self, weights):
        """
        Computes the volatility of the portfolio for given weights.

        Parameters:
        - weights: np.ndarray

        Returns:
        - float: portfolio volatility
        """
        return np.sqrt(weights.T @ self.Sigma @ weights)

    def _diversification_ratio(self, weights):
        """
        Computes the Diversification Ratio:
        sum(w_i * sigma_i) / portfolio_volatility

        Parameters:
        - weights: np.ndarray

        Returns:
        - float: diversification ratio
        """
        weighted_vols = np.dot(weights, self.sigma_individual)
        portfolio_vol = self._portfolio_volatility(weights)
        return weighted_vols / portfolio_vol

    def _neg_objective(self, weights):
        """
        Objective function to be minimized (negative diversification ratio).

        Parameters:
        - weights: np.ndarray

        Returns:
        - float: negative diversification ratio
        """
        return -self._diversification_ratio(weights)

    def optimize(self, short_allowed=False):
        """
        Runs the optimization to compute the Maximum Diversification Portfolio.

        Parameters:
        - short_allowed: bool — Whether short selling is allowed

        Returns:
        - pd.Series: optimal weights indexed by asset names
        """
        init_weights = np.ones(self.n) / self.n
        bounds = None if short_allowed else [(0.0, 1.0)] * self.n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(self._neg_objective,
                          init_weights,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)

        if result.success:
            return pd.Series(result.x, index=self.assets)
        else:
            raise ValueError("Optimization failed: " + result.message)
