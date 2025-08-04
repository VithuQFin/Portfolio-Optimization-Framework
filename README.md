# 🧠 Portfolio Optimization Strategies

This project implements and compares several portfolio optimization methods, using historical data from multiple assets. It includes a full pipeline: from data loading and cleaning, through optimization and backtesting, all the way to performance analysis and visualizations.

---

## 🚀 Features

- ✅ **Mean-Variance Optimization** (MVP & Tangency Portfolio)
- ✅ **Risk Parity Optimization**
- ✅ **Maximum Diversification Portfolio**
- ✅ **Equal-Weight Portfolio** (as benchmark)
- 📊 Performance metrics: Sharpe Ratio, Calmar Ratio, Max Drawdown, Annualized Return, etc.
- 📈 Visualizations: Efficient Frontier, Rolling Sharpe, Correlation Heatmap, Risk Contributions, etc.
- 🧪 Full backtest pipeline with benchmark comparison

---

## 📁 Project Structure

```plaintext
.
├── core/
│   ├── data_pipeline.py         # Load & clean asset and benchmark data
│   ├── optimizer_runner.py      # Runs all optimization strategies
│   ├── backtest_runner.py       # Runs backtests and computes performance
│
├── strategies/
│   ├── risk_parity.py           # Risk Parity optimizer
│   ├── max_div.py               # Max Diversification optimizer
│
├── utils/
│   ├── visualizations.py        # Charts & comparison tables
│   ├── reporting.py             # Rolling Sharpe, drawdowns, correlations
│
├── data/
│   ├── raw/                     # Raw downloaded prices
│   ├── clean/                   # Cleaned asset return data
│   ├── raw_benchmark/          # Raw benchmark data
│   └── clean_benchmark/        # Cleaned benchmark returns
│
├── backtest.py                 # Backtest metrics and plots
├── main.py                     # Entry point to run the full pipeline
├── requirements.txt
└── .gitignore
```

## Sample Outputs

    Efficient Frontier with MVP & Tangency portfolios

    Rolling Sharpe Ratios for each strategy

    Correlation heatmap of asset returns

    Barplot of risk contributions (Risk Parity)

    Comparative table of portfolio statistics

## How to Run

Install dependencies:
```
pip install -r requirements.txt
```
Run the main pipeline:
```
python main.py
```

## Strategies Compared

Strategy	            Description
```
MVP	                    Minimum Variance Portfolio (Markowitz model)
Tangency	            Maximum Sharpe Ratio portfolio (with risk-free asset)
Risk Parity	            Allocates risk equally across assets
Max Diversification	    Maximizes diversification ratio (Choueifaty & Coignard, 2008)
Equal Weights	            Baseline with 1/N asset allocation
```

## Author

Developed by Vithusan Kailasapillai — M2 Quantitative Finance @ ESILV
For educational and research purposes only.
