# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Crypto portfolio construction and risk app (prices from Binance), as the Crypto App notebook.

Tabs: Investment Universe (top crypto by market cap, Get Prices), Strategy,
Portfolio Analysis, Risk Analysis, Market Risk. The tabs live in tabs/, their
calculations in src/.

    streamlit run "Minimum Variance App.py"
"""
from tabs import data, run_app

run_app("Crypto Portfolio Construction", data.binance_loader)
