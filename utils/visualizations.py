import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_annotated_frontier(return_range, volatility_range, 
                             mvp_weights=None, tangency_weights=None,
                             expected_returns=None, cov_matrix=None, rf=0.0):
    """
    Plots the efficient frontier with annotations for MVP and Tangency portfolios.

    Parameters:
    - return_range: np.ndarray
        Expected returns for the frontier
    - volatility_range: np.ndarray
        Corresponding volatilities
    - mvp_weights: np.ndarray or None
        Minimum variance portfolio weights
    - tangency_weights: np.ndarray or None
        Tangency portfolio weights
    - expected_returns: pd.Series
    - cov_matrix: pd.DataFrame
    - rf: float
        Risk-free rate
    """
    plt.figure(figsize=(10, 6))
    plt.plot(volatility_range, return_range, label='Efficient Frontier', color='blue')

    if mvp_weights is not None:
        mvp_ret = np.dot(mvp_weights, expected_returns)
        mvp_vol = np.sqrt(mvp_weights.T @ cov_matrix @ mvp_weights)
        plt.scatter(mvp_vol, mvp_ret, color='red', marker='o', label='MVP')
        plt.annotate(f"MVP\n({mvp_vol:.2%}, {mvp_ret:.2%})", (mvp_vol, mvp_ret), xytext=(10, 10), textcoords='offset points')

    if tangency_weights is not None:
        tan_ret = np.dot(tangency_weights, expected_returns)
        tan_vol = np.sqrt(tangency_weights.T @ cov_matrix @ tangency_weights)
        plt.scatter(tan_vol, tan_ret, color='green', marker='*', label='Tangency')
        plt.annotate(f"Tangency\n({tan_vol:.2%}, {tan_ret:.2%})", (tan_vol, tan_ret), xytext=(10, 10), textcoords='offset points')

        # Capital Market Line (CML)
        slope = (tan_ret - rf) / tan_vol
        x_vals = np.linspace(0, max(volatility_range) * 1.1, 100)
        y_vals = rf + slope * x_vals
        plt.plot(x_vals, y_vals, linestyle='--', color='gray', label='Capital Market Line')

    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Annualized Risk)')
    plt.ylabel('Expected Return (Annualized)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def display_weights(weights, tickers, name):
    """
    Displays asset weights for a given portfolio.

    Parameters:
    - weights: array-like
    - tickers: list[str]
    - name: str
        Name of the portfolio
    """
    print(f"\n🔹 {name} Portfolio Weights:")
    for ticker, w in zip(tickers, weights):
        print(f"  - {ticker}: {w:.2%}")


def plot_cumulative_returns(cum_returns_dict):
    """
    Plots the cumulative performance for each portfolio.

    Parameters:
    - cum_returns_dict: dict[str, pd.Series]
        Dictionary of cumulative returns
    """
    plt.figure(figsize=(10, 6))
    for label, cum_returns in cum_returns_dict.items():
        plt.plot(cum_returns, label=label)
    plt.title("Cumulative Performance")
    plt.xlabel("Date")
    plt.ylabel("Capital Growth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_risk_contributions(weights, cov_matrix, title="Risk Contributions"):
    """
    Plots a bar chart of the risk contributions from each asset.

    Parameters:
    - weights: pd.Series
    - cov_matrix: pd.DataFrame
    - title: str
    """
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_contrib = cov_matrix @ weights
    contrib = weights * marginal_contrib / portfolio_vol

    plt.figure(figsize=(10, 5))
    plt.bar(weights.index, contrib)
    plt.title(title)
    plt.ylabel("Risk Contribution (volatility points)")
    plt.xlabel("Assets")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(matrix, title="Correlation Matrix"):
    """
    Plots a correlation heatmap of asset returns.

    Parameters:
    - matrix: pd.DataFrame
    - title: str
    """
    corr = matrix.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_weights_comparison(weights_dict):
    """
    Visually compares the weights of multiple portfolios.

    Parameters:
    - weights_dict: dict[str, pd.Series]
        Dictionary of portfolio weights
    """
    weights_df = pd.DataFrame(weights_dict)
    weights_df.plot(kind='bar', figsize=(12, 6))
    plt.title("Portfolio Weights Comparison")
    plt.ylabel("Weight (%)")
    plt.xlabel("Assets")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title="Portfolio")
    plt.show()


def display_comparative_table(stats_dict, sort_by="Sharpe Ratio"):
    """
    Display a summary comparison table of performance metrics for all portfolios.

    Parameters:
    - stats_dict: dict[str, dict[str, float]]
        Dictionary with portfolio names as keys and performance stats as sub-dictionaries.
    - sort_by: str
        Metric to sort the table by (default is Sharpe Ratio)
    """
    df = pd.DataFrame(stats_dict).T
    df = df[[
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Calmar Ratio",
        "Max Drawdown"
    ]]

    # Format for readability
    df["Annualized Return"] = df["Annualized Return"].apply(lambda x: f"{x:.2%}")
    df["Annualized Volatility"] = df["Annualized Volatility"].apply(lambda x: f"{x:.2%}")
    df["Max Drawdown"] = df["Max Drawdown"].apply(lambda x: f"{x:.2%}")
    df["Sharpe Ratio"] = df["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
    df["Calmar Ratio"] = df["Calmar Ratio"].apply(lambda x: f"{x:.2f}")

    print("\n📋 Comparative Performance Table:")
    print(df.sort_values(by=sort_by, ascending=False).to_string())