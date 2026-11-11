# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Portfolio construction and risk app on an uploaded Excel workbook (full version).

Tabs as the Risk Management notebooks: Investment Universe, Strategy (optimization,
rolling backtest with overlays), Portfolio Analysis, Risk Analysis, Market Risk.
The tabs live in tabs/, their calculations in src/.

    streamlit run Streamlit_App.py
"""
from tabs import data, run_app

run_app("Portfolio Construction", data.excel_loader)
