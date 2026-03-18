# Apps:

* https://cryptoptimization.streamlit.app
* https://optimisationapp.streamlit.app

# 📊 Portfolio Optimization Tools

An interactive **Streamlit-based portfolio optimization and risk analysis tool** designed for financial analysis, asset allocation, and portfolio construction.
---

## 🚀 Features

### 📈 Portfolio Analysis

* Upload your own **Excel time series data**
* Compute:

  * Returns (since inception & YTD)
  * Annualized returns
  * Volatility (daily & monthly)
  * Maximum drawdown
  * Conditional Value at Risk (CVaR)
    
* Visualize:

  * Portfolio value evolution
  * Drawdowns
  * Rolling volatility
  * Calendar Metrics
  * PCA
  * Contribution to Risk

---

### ⚖️ Portfolio Optimization

Supports multiple portfolio construction methods and  dynamic stragies:

* ✅ Maximum Sharpe Ratio
* ✅ Minimum Variance
* ✅ Risk Parity
* ✅ Maximum Diversification
* ✅ Equal Weight Portfolio

---

### 🔒 Constraint System

Flexible constraint builder:

* Per-asset constraints (≥, ≤, =)
* Global diversification constraints
* Interactive UI for adding/removing constraints dynamically

---

### 🔁 Rebalancing Engine

Compare:

* Buy & Hold strategy
* Periodic rebalancing:

  * Monthly
  * Quarterly
  * Yearly
    
* Dynamic Funds
---

### 📊 Efficient Frontier

* Interactive efficient frontier visualization
* Portfolio overlays
* Sharpe ratio heatmap
* Correlation matrix

---

### 🧩 Risk Decomposition

* Risk contribution by asset
* Profit & Loss breakdown
* Value At Risk Analysis
* Historical Contribution to TE/Vol
* Market Risk Analysis

---

## 📂 Input Data Format for the first app
https://optimisationapp.streamlit.app


Upload an Excel file (`.xlsx`) with:

* **Rows** → Dates
* **Columns** → Asset prices
* **Index** → Date column

Example:

| Date       | Asset A | Asset B | Asset C |
| ---------- | ------- | ------- | ------- |
| 2020-01-01 | 100     | 200     | 150     |
| 2020-01-02 | 101     | 198     | 152     |

## 📂 Use Binance API to retrieve prices for the second app

https://cryptoptimization.streamlit.app

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/niroojane/Streamlit-Portfolio-App
cd Streamlit-Portfolio-App
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run Portfolio_App.py
streamlit run Minimum
```

---

## 📦 Dependencies

* beautifulsoup4
* ipython
* matplotlib
* numpy
* pandas
* plotly
* Requests
* scipy
* seaborn
* statsmodels
* streamlit
* yfinance
* openpyxl


Custom modules:

* `RiskMetrics`
* `Rebalancing`
* `Metrics`
---

## ⚙️ App Structure

```
.
├── Portfolio_App.py        #  Streamlit application with manual input
├── Minimum_Variance_App.py #  Streamlit application for crypto market
├── RiskMetrics.py          # Portfolio & risk calculations
├── Rebalancing.py          # Rebalancing logic
├── Metrics.py              # Returns metrics
├── Price_Endpoint.py       # Binance API
├── Stock_Data.py           # Yahoo Finance API
├── Gradio.py               # App built using Gradio (not updated)
├── requirements.txt
└── README.md
```
---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Niroojane Selvam**

---

## 📬 Contact

https://www.linkedin.com/in/niroojane-selvam-498157196/

For questions or feedback, feel free to reach out or open an issue.

---

⭐ If you find this project useful, consider giving it a star!
