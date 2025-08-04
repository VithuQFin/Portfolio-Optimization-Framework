import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParityOptimizer:
    def __init__(self, cov_matrix, risk_budget=None):
        """
        Initializes the Risk Parity optimizer.

        Parameters:
        - cov_matrix: pd.DataFrame
            Covariance matrix of asset returns
        - risk_budget: np.ndarray or None
            Target risk contributions per asset (default: equal budget)
        """
        self.Sigma = cov_matrix
        self.assets = cov_matrix.columns
        self.n = len(self.assets)
        self.risk_budget = (
            np.ones(self.n) / self.n if risk_budget is None else np.array(risk_budget)
        )

    def _portfolio_volatility(self, weights):
        """
        Computes the portfolio volatility.

        Parameters:
        - weights: np.ndarray

        Returns:
        - float: portfolio volatility
        """
        return np.sqrt(weights.T @ self.Sigma @ weights)

    def _risk_contributions(self, weights):
        """
        Computes the risk contribution of each asset:
        RC_i = w_i * (Σw)_i / portfolio_volatility

        Parameters:
        - weights: np.ndarray

        Returns:
        - np.ndarray: risk contributions per asset
        """
        portfolio_vol = self._portfolio_volatility(weights)
        marginal_contrib = self.Sigma @ weights
        risk_contrib = weights * marginal_contrib
        return risk_contrib / portfolio_vol

    def _objective(self, weights):
        """
        Objective function:
        Sum of squared deviations between actual and target risk contributions.

        Parameters:
        - weights: np.ndarray

        Returns:
        - float: value of the objective function
        """
        risk_contrib = self._risk_contributions(weights)
        return np.sum((risk_contrib - self.risk_budget) ** 2)

    def optimize(self, short_allowed=False):
        """
        Solves the risk parity optimization problem.

        Parameters:
        - short_allowed: bool — Whether short selling is allowed

        Returns:
        - pd.Series: optimal portfolio weights indexed by asset names
        """
        init_weights = np.ones(self.n) / self.n
        bounds = None if short_allowed else [(0.0, 1.0)] * self.n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(self._objective,
                          init_weights,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)

        if result.success:
            return pd.Series(result.x, index=self.assets)
        else:
            raise ValueError("Optimization failed: " + result.message)
