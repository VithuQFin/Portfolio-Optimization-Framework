import numpy as np
import pandas as pd

# Core modules
from core.data_pipeline import load_assets_and_benchmark
from core.optimizer_runner import run_optimizations, run_risk_parity_optimization, run_max_div_optimization
from core.backtest_runner import run_backtest
from utils.visualizations import display_comparative_table

# Visualization & Reporting
from utils.visualizations import (
    plot_annotated_frontier,
    display_weights,
    plot_cumulative_returns,
    plot_risk_contributions,
    plot_weights_comparison,
    plot_correlation_heatmap,
)
from utils.reporting import (
    plot_rolling_sharpe,
    plot_drawdowns,
    plot_rolling_correlation,
    print_cumulative_performance,
)

# === Global Parameters ===
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'META']
START_DATE, END_DATE = '2022-01-01', '2024-01-01'
RAW_PATH, CLEAN_PATH = "data/raw/", "data/clean/"
BENCHMARK = '^GSPC'
BENCH_RAW_PATH, BENCH_CLEAN_PATH = "data/raw_benchmark/", "data/clean_benchmark/"
RISK_FREE_RATE, TRADING_DAYS = 0.02, 252

# === 1. Load data ===
pivoted, benchmark_returns = load_assets_and_benchmark(
    TICKERS, BENCHMARK, START_DATE, END_DATE,
    RAW_PATH, CLEAN_PATH, BENCH_RAW_PATH, BENCH_CLEAN_PATH
)

# === 2. Markowitz Optimizations ===
mvp_weights, tan_weights, mu, cov, rets, vols = run_optimizations(
    pivoted, TRADING_DAYS, RISK_FREE_RATE
)

display_weights(mvp_weights, mu.index, "Minimum Variance")
display_weights(tan_weights, mu.index, "Tangency")
plot_annotated_frontier(rets, vols, mvp_weights, tan_weights, mu, cov, rf=RISK_FREE_RATE)

# === 2.b Risk Parity ===
rp_weights = run_risk_parity_optimization(cov)
display_weights(rp_weights.values, rp_weights.index, "Risk Parity")
plot_risk_contributions(rp_weights, cov)

# === 2.c Equal Weights ===
equal_weights = np.ones(len(mu)) / len(mu)
display_weights(equal_weights, mu.index, "Equal Weights")

# === 2.d Maximum Diversification ===
max_div_weights = run_max_div_optimization(cov)
display_weights(max_div_weights.values, max_div_weights.index, "Max Diversification")

# === 3. Backtesting ===
returns_dict, cum_returns_dict, stats = run_backtest(
    mvp_weights, tan_weights, rp_weights, equal_weights, max_div_weights,
    pivoted, benchmark_returns
)

# === 4. Visualization & Reporting ===
plot_correlation_heatmap(cov, title="Empirical Correlation Matrix")

print("\n📊 Performance Statistics:")
for name, s in stats.items():
    print(f"\n🔹 {name} Portfolio:")
    for k, v in s.items():
        print(f"  {k}: {v:.2%}" if 'Ratio' not in k else f"  {k}: {v:.2f}")

plot_cumulative_returns(cum_returns_dict)
plot_rolling_sharpe(returns_dict, RISK_FREE_RATE)
plot_drawdowns(cum_returns_dict)
plot_rolling_correlation(returns_dict, benchmark_returns)
print_cumulative_performance(cum_returns_dict)

plot_weights_comparison({
    "Minimum Variance": pd.Series(mvp_weights, index=mu.index),
    "Tangency": pd.Series(tan_weights, index=mu.index),
    "Risk Parity": rp_weights,
    "Equal Weights": pd.Series(equal_weights, index=mu.index),
    "Max Diversification": max_div_weights
})

display_comparative_table(stats, sort_by="Sharpe Ratio")