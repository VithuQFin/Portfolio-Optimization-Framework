import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_annotated_frontier_plotly(return_range, volatility_range, 
                                    mvp_weights=None, tangency_weights=None,
                                    expected_returns=None, cov_matrix=None, rf=0.0):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=volatility_range, y=return_range,
        mode='lines', name='Efficient Frontier',
        line=dict(color='blue')
    ))

    if mvp_weights is not None and expected_returns is not None and cov_matrix is not None:
        mvp_ret = np.dot(mvp_weights, expected_returns)
        mvp_vol = np.sqrt(mvp_weights.T @ cov_matrix @ mvp_weights)
        fig.add_trace(go.Scatter(
            x=[mvp_vol], y=[mvp_ret],
            mode='markers+text', name='MVP',
            marker=dict(color='red', size=10),
            text=[f"MVP<br>({mvp_vol:.2%}, {mvp_ret:.2%})"],
            textposition='top center'
        ))

    if tangency_weights is not None and expected_returns is not None and cov_matrix is not None:
        tan_ret = np.dot(tangency_weights, expected_returns)
        tan_vol = np.sqrt(tangency_weights.T @ cov_matrix @ tangency_weights)
        fig.add_trace(go.Scatter(
            x=[tan_vol], y=[tan_ret],
            mode='markers+text', name='Tangency',
            marker=dict(color='green', size=10, symbol='star'),
            text=[f"Tangency<br>({tan_vol:.2%}, {tan_ret:.2%})"],
            textposition='top center'
        ))

        slope = (tan_ret - rf) / tan_vol
        x_vals = np.linspace(0, max(volatility_range) * 1.1, 100)
        y_vals = rf + slope * x_vals
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines', name='Capital Market Line',
            line=dict(dash='dash', color='gray')
        ))

    fig.update_layout(
        title="Efficient Frontier (Plotly)",
        xaxis_title="Volatility (Annualized Risk)",
        yaxis_title="Expected Return (Annualized)",
        template="plotly_white"
    )
    return fig


def plot_cumulative_returns_plotly(cum_returns_dict):
    fig = go.Figure()
    for name, series in cum_returns_dict.items():
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode='lines', name=name
        ))

    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white"
    )
    return fig


def plot_risk_contributions_plotly(weights, cov_matrix, title="Risk Contributions"):
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_contrib = cov_matrix @ weights
    contrib = weights * marginal_contrib / portfolio_vol

    fig = go.Figure(data=[
        go.Bar(x=weights.index, y=contrib.values, marker_color='indianred')
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Assets",
        yaxis_title="Risk Contribution",
        template="plotly_white"
    )
    return fig


def plot_correlation_heatmap_plotly(matrix, title="Correlation Matrix"):
    corr = matrix.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        labels=dict(color="Correlation")
    )
    fig.update_layout(title=title)
    return fig


def plot_weights_comparison_plotly(weights_dict):
    df = pd.DataFrame(weights_dict)
    df = df.fillna(0)

    fig = go.Figure()
    for portfolio in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[portfolio],
            name=portfolio
        ))

    fig.update_layout(
        barmode='group',
        title="Portfolio Weights Comparison",
        xaxis_title="Assets",
        yaxis_title="Weight",
        template="plotly_white"
    )
    return fig
