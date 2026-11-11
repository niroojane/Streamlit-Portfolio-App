# 📊 Portfolio Optimization Tools

Interactive **portfolio construction and risk analysis apps** built with **Streamlit** (and a **Gradio** version), with the same tabs as the [Risk Management](https://github.com/niroojane/Risk-Management) notebooks: optimization, rolling backtests with overlays, performance and risk attribution, Value at Risk with stress tests, and market risk.

## Apps

| App | Data | Live |
|---|---|---|
| **Streamlit_App.py** | Your Excel workbook (prices or returns, optional exposure table) | https://portfolioconstruction.streamlit.app |
| **Minimum Variance App.py** | Top crypto by market cap, prices from Binance | https://cryptoptimization.streamlit.app |
| **Gradio.py** | Your Excel workbook | run locally |

---

## 🧭 Tabs

Same structure as the Risk Management Crypto App notebook (without the account P&L):

| Tab | Sub-tabs |
|---|---|
| **Investment Universe** | Asset returns / risk, price and cumulative performance, exposure table |
| **Strategy** | **Strategy**: constraints, optimization grid, core strategy + overlays, rolling backtest, efficient frontier · **Strategy Return**: calendar returns |
| **Portfolio Analysis** | Return Analysis (return, active and drawdown contribution) · Beta Analysis |
| **Risk Analysis** | Volatility Analysis (Risk Contribution, Risk Trajectory, Tracking Error) · Value at Risk (stress grids, VaR History) |
| **Market Risk** | PCA · Correlation · Market Drivers |

Each tab has its own **Date** slider; rebalancing frequency and rolling window are set in the Strategy tab.

---

## 🚀 Features

### ⚖️ Portfolio Optimization
Long-only, fully invested, constrained and unconstrained side by side:
* Maximum Sharpe Ratio, Minimum Variance, Risk Parity, Maximum Diversification, Eigen Portfolio, Equal Weight
* **Minimum Tracking Error** and **Maximum Information Ratio** relative to a benchmark allocation

### 🔒 Constraints
* Per asset (≥, ≤, =) and on every weight (`All`)
* **Exposure buckets** (sector, asset class, country...) when the workbook has an exposure table: `weights @ exposure` is bounded, and a look-through table shows each allocation's exposures
* Risk limits: ex-ante **volatility**, **beta**, **tracking error**, **information ratio**
* Incomplete constraint rows are reported instead of silently ignored

### 🔁 Backtesting
* Buy & hold and rebalanced (monthly / quarterly / yearly) versions of every allocation
* **Rolling optimization** of a core strategy blended with tactical **overlays** (Fund / Core / Overlay), with the latest *Model* allocation added to the grid

### 📈 Performance & Risk
* Returns, annualized returns, YTD, volatility, tracking error, drawdowns, parametric CVaR
* Calendar returns, volatility, tracking error and Sharpe ratio by year / month
* Return and drawdown contribution, rolling betas
* Ex-ante volatility / tracking error and their decomposition (contribution, correlation, idiosyncratic) over time

### 🧩 Value at Risk & Stress Testing
* Historical, parametric, multivariate normal, Gaussian / t-Student / Gumbel copulas, Monte Carlo
* Editable stress grids: volatility multipliers, correlation shocks, mean-return shocks
* Per-asset VaR / CVaR contribution, rolling VaR history, Kupiec backtest

### 🌐 Market Risk
* PCA: variance explained, eigen portfolios and the assets closest to them
* Rolling pair correlation and first principal component over time
* Eigen-portfolio market index, its P&L drivers and risk decomposition

---

## 📂 Input Data

### Excel workbook (Streamlit_App, Gradio)

* **Sheet 1** — dates in the first column, one column per asset: **prices**, or **daily returns** (detected from negative values below 100%, compounded to prices).
* **Sheet 2 (optional)** — **exposure table**: buckets in rows, assets in columns, in % or decimals. Its buckets become constraint targets and appear in the look-through table.

| Date       | Asset A | Asset B | Asset C |
| ---------- | ------- | ------- | ------- |
| 2020-01-01 | 100     | 200     | 150     |
| 2020-01-02 | 101     | 198     | 152     |

`Crypto Prices1.xlsx` is a sample workbook. Pick the benchmark asset after uploading.

### Crypto universe (Minimum Variance App)

Choose the number of crypto (top market caps on Binance, tokenized stocks excluded), untick the ones you don't want, set the starting date and click **Get Prices**. Daily closes come from Binance's public market data (no API key); assets with 30 or more missing days in the period are left out, as in the notebook.

---

## 🛠️ Installation

```bash
git clone https://github.com/niroojane/Streamlit-Portfolio-App
cd Streamlit-Portfolio-App
pip install -r requirements.txt
```

## ▶️ Run

```bash
streamlit run Streamlit_App.py
streamlit run "Minimum Variance App.py"
python Gradio.py          # needs: pip install gradio
```

Stop an app with **Ctrl+C** before starting another one, or give it its own port (`--server.port 8502`): several servers left on the same port answer in turn.

---

## ⚙️ Structure

```
.
├── Streamlit_App.py            # Workbook app
├── Minimum Variance App.py     # Crypto app (Binance)
├── Gradio.py                   # Gradio version of the workbook app
├── tabs/                       # One module per tab
│  ├── engine.py                # UI-agnostic glue to src/ (also used by Gradio and the BQuant apps)
│  ├── charts.py                # Plotly figures and the notebook chart layouts
│  ├── ui.py                    # Streamlit helpers: context, caching, editors, tables
│  ├── data.py                  # Loaders: Excel workbook, Binance universe
│  ├── universe.py              # Investment Universe
│  ├── strategy.py              # Strategy | Strategy Return
│  ├── portfolio.py             # Portfolio Analysis
│  ├── risk.py                  # Risk Analysis
│  └── market.py                # Market Risk
├── src/                        # Analytics library (shared with Risk Management)
│  ├── RiskMetrics.py           # Optimization, constraints, VaR models, risk decomposition
│  ├── Rebalancing.py           # Buy & hold, periodic and dynamic rebalancing, book cost
│  ├── Metrics.py               # Performance / risk tables, calendar metrics, efficient frontier
│  ├── Backtest.py              # Rolling optimization: core strategy + overlays
│  ├── PortfolioAnalytics.py    # Constraints, optimization grid, Value at Risk
│  ├── Workspace.py             # Named portfolio series and the tab views
│  ├── Exposure.py              # Exposure-bucket constraints, look-through, bucket bounds
│  ├── Trade.py                 # Rebalancing trade optimizer (used by the BQuant Trade app)
│  ├── Price_Endpoint.py        # Binance market caps and daily closes
│  ├── Yahoo_API.py             # Yahoo Finance prices
│  └── Stock_Data.py            # Yahoo Finance loader (legacy)
├── Crypto Prices1.xlsx         # Sample workbook
├── requirements.txt
└── README.md
```

---

## 📄 License

This project is licensed under the **MIT License**.

## 👤 Author

**Niroojane Selvam** — https://www.linkedin.com/in/niroojane-selvam-498157196/

For questions or feedback, feel free to reach out or open an issue.

⭐ If you find this project useful, consider giving it a star!
