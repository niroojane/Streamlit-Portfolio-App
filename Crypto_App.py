# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import requests
from scipy.stats import norm, chi2,gumbel_l
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

import ipywidgets as widgets
from ipydatagrid import DataGrid, TextRenderer
from IPython.display import display,Markdown
from IPython.display import HTML
import plotly.express as px
import plotly.graph_objects as go

from src import GitHub
from src import BinanceAPI
from src.RiskMetrics import *
from src import PnL
from src import get_close
from src.Rebalancing import *
from src.Metrics import *

def display_crypto_app(Binance,Pnl_calculation,git):
    # --- strategy dictionary ---
    dico_strategies = {
        'Minimum Variance': 'minimum_variance',
        'Risk Parity': 'risk_parity',
        'Sharpe Ratio': 'sharpe_ratio',
        'Maximum Diversification':'maximum_diversification',
        'Eigen Strategy':'eigenportfolio'}
    
    
    options_strat = list(dico_strategies.keys())

    # --- globals ---
    global tickers_dataframe, tickers, dataframe, returns_to_use, prices
    global rolling_optimization, performance_pct, performance_fund, dates_end,quantities,quantities_core,quantities_overlay,cumulative_results,global_returns
    global book_cost,realized_pnl,profit_and_loss,holding_tickers,current_weights,fund_names,grid,trades
    
    # tickers_dataframe = Binance.get_market_cap().set_index('Ticker')
    tickers_dataframe = (
    Binance.get_market_cap()
    .set_index('Ticker')
    .loc[lambda df: ~df['Long name'].str.contains(r'\(bStocks\)', na=False)]
    )
    
    tickers = []
    holding_tickers=[]
    dataframe = pd.DataFrame()
    cumulative_results=pd.DataFrame()
    global_returns=pd.DataFrame()

    fund_names=[]
    current_weights=pd.DataFrame()
    book_cost=pd.DataFrame()
    realized_pnl=pd.DataFrame()
    trades=pd.DataFrame()

    profit_and_loss=pd.DataFrame()
    returns_to_use = pd.DataFrame()
    prices = pd.DataFrame()
    
    rolling_optimization = pd.DataFrame()
    quantities=pd.DataFrame()
    quantities_core=pd.DataFrame()
    quantities_overlay=pd.DataFrame()
    
    performance_pct = pd.DataFrame()
    performance_fund = pd.DataFrame()
    
    dates_end = []
    constraint_container = {'constraints': [], 'allocation_df': pd.DataFrame()}

    # --- UI setup ---
    start_date = widgets.DatePicker(
        value=datetime.date(2020, 1, 1),
        description='Starting Date of Backtest',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='350px')
    )

    n_crypto = widgets.IntSlider(
        min=1, max=100, value=20,
        description='Number of Crypto',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='500px')
    )
    
    loading_bar = widgets.IntProgress(description='Loading prices...',min=0, max=100,style={'description_width': '150px'})
    
    data_button = widgets.Button(description='Get Prices', button_style='info')
    scope_output = widgets.Output()
    strategy_output = widgets.Output()
    main_output = widgets.Output()
    output_returns = widgets.Output()
    constraint_output = widgets.Output()
    
    dropdown_asset1 = widgets.Dropdown(description='Asset 1:',style={'description_width': '150px'})
    dropdown_asset2 = widgets.Dropdown(description='Asset 2:',style={'description_width': '150px'} )

    
    def scope_update(n):
        nonlocal scope_output
        global tickers_dataframe, tickers

        try:
            selected_tickers=tickers_dataframe.iloc[:n]
            selected_tickers_list = list(set(selected_tickers.index))
            
        except Exception as e:
            with scope_output:
                scope_output.clear_output(wait=True)
                print("Error fetching market caps:", e)
            return
  
              
        with scope_output:
            scope_output.clear_output(wait=True)
            display(display_scrollable_df(selected_tickers))
    
            checkboxes = {
                t: widgets.Checkbox(description=t, value=True)
                for t in selected_tickers_list
            }
    
            rows = [
                widgets.HBox(list(checkboxes.values())[i:i+5])
                for i in range(0, len(checkboxes), 5)
            ]
    
            ui = widgets.VBox(rows)
    
            def on_change(change=None):
                selected = [k for k, cb in checkboxes.items() if cb.value]
                # update global tickers
                global tickers
                tickers = selected
    
                df = pd.DataFrame(selected, columns=["Selected Tickers"]).set_index("Selected Tickers")
    
            for cb in checkboxes.values():
                cb.observe(on_change, names="value")
    
            on_change()

            display(Markdown("### Selected Tickers"))

            display(ui)
            
    scope_update(n_crypto.value)
    
    n_crypto.observe(lambda ch: scope_update(ch['new']) if ch['name'] == 'value' else None, names='value')
    
    price_output=widgets.Output()

    def get_price_threading(tickers,start_date):
            
        today = datetime.date.today()
        days_total = (today - start_date).days
        if days_total <= 0:
            print("Start date must be in the past.")
            return

        remaining = days_total % 500
        numbers_of_table = days_total // 500

        loading_bar.value = 0
        display(loading_bar)
        loading_bar.max = numbers_of_table + 1
        
        start_dt= datetime.datetime.combine(start_date, datetime.time())
        end_dates = [
                start_dt + datetime.timedelta(days=500 * i)
                for i in range(numbers_of_table + 1)
            ]
    
        end_dates.append(
            datetime.datetime.combine(
                today - datetime.timedelta(days=remaining),
                datetime.time()
            )
        )

        def fetch_prices(end_date):
            return Binance.get_price(tickers, end_date)

        price = None

        try:
            with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                futures = [executor.submit(fetch_prices, d) for d in end_dates]

                for future in as_completed(futures):
                    data = future.result()

                    if price is None:
                        price = data
                    else:
                        price = price.combine_first(data)

                    loading_bar.value += 1

        except Exception as e:
            print("❌ Error while fetching prices:", e)
            return

        price = price.sort_index()
        price = price[~price.index.duplicated(keep="first")]
        price.index = pd.to_datetime(price.index)
        
        loading_bar.value = loading_bar.max

        return price

        
    def get_prices(_=None):
        
        global prices, dataframe, returns_to_use, dates_end, valid_cols_model
    
        get_holdings(None)
        combined_tickers = sorted(list(set(tickers + holding_tickers)))
    
        with main_output:
            main_output.clear_output(wait=True)
    
            if not tickers:
                print("No tickers available. Please fetch tickers first.")
                return
    
            start = start_date.value
            if not isinstance(start, datetime.date):
                print("Please select a valid start date.")
                return

            
            scope_prices=get_price_threading(combined_tickers,start)
    
            prices = scope_prices.loc[:, scope_prices.columns != "USDCUSDT"]
    
            returns = np.log(1 + prices.pct_change(fill_method=None))
            returns.index = pd.to_datetime(returns.index)
    
            valid_cols = returns.columns[returns.isna().sum() < 30]
            returns_to_use = returns[valid_cols].sort_index()
    
            dataframe = prices[valid_cols].sort_index().dropna()
            dataframe.index = pd.to_datetime(dataframe.index)
            returns_to_use = returns_to_use[~returns_to_use.index.duplicated(keep="first")]
        
            main_output.clear_output()
    
            dropdown_asset.options = list(dataframe.columns) + ["All"]
    
            dropdown_asset1.options = dataframe.columns
            dropdown_asset1.value = dataframe.columns[0]
    
            dropdown_asset2.options = dataframe.columns
            dropdown_asset2.value = dataframe.columns[1]
    
            print(
                f"✅ Loaded prices for {len(dataframe.columns)} assets "
                f"from {dataframe.index[0].date()} to {dataframe.index[-1].date()}"
            )
    
            asset_risk = get_asset_risk(dataframe)
            asset_returns = get_asset_returns(dataframe)
            
            display(display_scrollable_df(asset_returns))
            display(display_scrollable_df(asset_risk))
                        
        with price_output:
            
            price_output.clear_output(wait=True)

            price_output_graph=widgets.Output()
            return_output_graph=widgets.Output()
            
            with price_output_graph:
                fig = px.line(
                    dataframe.loc[start_date_perf.value:end_date_perf.value],
                    title="Price",
                    width=800,
                    height=400
                , render_mode = 'svg')
                fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig.update_traces(visible="legendonly", selector=lambda t: t.name != "BTCUSDT")
                fig.show()

            with return_output_graph:
                
                cumulative_returns = returns_to_use.loc[start_date_perf.value:end_date_perf.value].copy()
                cumulative_returns.iloc[0] = 0
                cumulative_returns = (1 + cumulative_returns).cumprod() * 100
                fig2 = px.line(
                    cumulative_returns,
                    title="Cumulative Performance",
                    width=800,
                    height=400,
                    render_mode = 'svg')
                fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig2.update_traces(visible="legendonly", selector=lambda t: t.name != "BTCUSDT")
                fig2.show()

            ui=widgets.HBox([price_output_graph,return_output_graph])
            display(ui)
            display(display_scrollable_df(dataframe))

    data_button.on_click(get_prices)
    start_date.observe(lambda ch: get_prices() if ch['name'] == 'value' and ch['new'] else None, names='value')
    
    # --- constraint UI ---
    dropdown_asset = widgets.Dropdown(description='Asset:', options=['All'], value=None)
    dropdown_sign = widgets.Dropdown(description='Sign:', options=["=", "≥", "≤"])
    dropdown_limit = widgets.FloatText(description='Limit')
    add_constraint_btn = widgets.Button(description='Add Constraint', button_style='success')
    clear_constraints_btn = widgets.Button(description='Clear All', button_style='danger')
    constraints = []
    
    
    selected_fund = widgets.Dropdown(description="Fund:")
    selected_bench = widgets.Dropdown(description="Bench:")
    selected_fund_var = widgets.Dropdown(description="Fund:")

    
    add_strategy_btn = widgets.Button(description='Add Strategy',style={'description_width': '150px'} )
    clear_strategy_btn = widgets.Button(description='Clear Strategy',style={'description_width': '150px'} )
    dropdown_strategy_limit = widgets.FloatText(description='Overlay Limit',style={'description_width': '150px'} )
    strat_overlay = widgets.Dropdown(description='Strategy Overlay', options=options_strat, value='Minimum Variance',style={'description_width': '150px'} )

    overlays=[]
    overlay_output=widgets.Output()
    
    def on_add_overlays(_):
        overlays.append({
            'Strategy': strat_overlay.value,
            'Limit': dropdown_strategy_limit.value
        })
        
        with overlay_output:
            overlay_output.clear_output(wait=True)
            display(display_scrollable_df(pd.DataFrame(overlays)))
            
    def on_clear_overlays(_):
        overlays.clear()
        with overlay_output:
            overlay_output.clear_output(wait=True)
            display(display_scrollable_df(pd.DataFrame(columns=['Strategy', 'Limit'])))  

    add_strategy_btn.on_click(on_add_overlays)
    clear_strategy_btn.on_click(on_clear_overlays)

    def on_add_constraint_clicked(_):
        constraints.append({
            'Asset': dropdown_asset.value,
            'Sign': dropdown_sign.value,
            'Limit': dropdown_limit.value
        })
        with constraint_output:
            constraint_output.clear_output(wait=True)
            display(display_scrollable_df(pd.DataFrame(constraints)))

    
    def on_clear_constraints(_):
        constraints.clear()
        with constraint_output:
            constraint_output.clear_output(wait=True)
            display(display_scrollable_df(pd.DataFrame(columns=['Asset', 'Sign', 'Limit'])))

    add_constraint_btn.on_click(on_add_constraint_clicked)
    clear_constraints_btn.on_click(on_clear_constraints)
    
    def on_add_click(b):
        
        global fund_names,grid

        if grid.data is None or grid.data.empty:
            return
        new_row = np.zeros(dataframe.shape[1])
        label = f"Allocation {grid.data.shape[0]}"
        new_df = pd.DataFrame([new_row], columns=grid.data.columns, index=[label])
        updated_df = pd.concat([pd.DataFrame(grid.data), new_df])
        grid.data = updated_df
        
        benchmark_tracking_error.options=grid.data.index
        selected_fund.options=grid.data.index
        selected_bench.options=grid.data.index
        
        selected_fund_var.options=grid.data.index

    def clear_allocation(b):
        
        nonlocal constraint_container
        if constraint_container.get('allocation_df') is not None:
            grid.data = constraint_container['allocation_df']
            
    button_add = widgets.Button(description="Add Allocation")    
    button_clear = widgets.Button(description="Clear Allocation")            
    button_add.on_click(on_add_click)
    button_clear.on_click(clear_allocation)
    
    # --- date pickers for performance ---
    if isinstance(start_date.value, datetime.date):
        sd = start_date.value
        start_perf_date = datetime.date(sd.year, sd.month + 2, 1)
    else:
        start_perf_date = datetime.date.today() - datetime.timedelta(days=365)

    start_date_perf = widgets.DatePicker(value=start_perf_date, layout=widgets.Layout(width='350px'))
    end_date_perf = widgets.DatePicker(value=datetime.date.today(), layout=widgets.Layout(width='350px'))
    
    frequency_graph=widgets.Dropdown(description='Frequency:', options=['Year','Month'], value='Year')
    benchmark=widgets.Dropdown(description='Benchmark:', options=['Fund','Bitcoin'], value='Bitcoin')
    fund=widgets.Dropdown(description='Fund:', options=['Fund','Bitcoin'], value='Fund')
    benchmark_tracking_error=widgets.Dropdown(description='Benchmark:')

    perf_output=widgets.Output()
    vol_output=widgets.Output()
    drawdown_output=widgets.Output()
    frontier_output=widgets.Output()
    
    # --- performance update ---
    def updated_cumulative_perf(_):
        
        global performance_pct, performance_fund,cumulative_results,global_returns
        
        try:
            start_ts = pd.to_datetime(start_date_perf.value)
            end_ts = pd.to_datetime(end_date_perf.value)
        except Exception:
            with output_returns:
                output_returns.clear_output(wait=True)
                print("⚠️ Invalid start/end dates.")
            return
            
        if dataframe.empty:
            with main_output:
                main_output.clear_output(wait=True)
                print("⚠️ Load Prices.")
                return
        else:

            with main_output:
                
                main_output.clear_output(wait=True)
                range_prices=dataframe.loc[start_ts:end_ts]
                range_returns=range_prices.pct_change(fill_method=None)
                
                asset_risk=get_asset_risk(range_prices)
                asset_returns=get_asset_returns(range_prices)
                display(display_scrollable_df(asset_returns))
                display(display_scrollable_df(asset_risk))
                
            with price_output:
                price_output.clear_output(wait=True)
                price_output_graph=widgets.Output()
                return_output_graph=widgets.Output()
                
                with price_output_graph:
                    fig = px.line(dataframe.loc[start_date_perf.value:end_date_perf.value], title='Price', width=800, height=400, render_mode = 'svg')
                    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["BTCUSDT"])
        
                    fig.show()
                with return_output_graph:
                    
                    cumulative_returns=returns_to_use.loc[start_date_perf.value:end_date_perf.value].copy()
                    cumulative_returns.iloc[0]=0
                    cumulative_returns=(1+cumulative_returns).cumprod()*100
                    
                    fig2 = px.line(cumulative_returns, title='Cumulative Performance', width=800, height=400, render_mode = 'svg')
                    fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["BTCUSDT"])
        
                    fig2.show()
    
                ui=widgets.HBox([price_output_graph,return_output_graph])
                display(ui)
                display(display_scrollable_df(dataframe))

        if performance_pct is None or performance_pct.empty:
            with output_returns:
                output_returns.clear_output(wait=True)
                print("⚠️ No performance data available yet. Please run an optimization first.")
            return
       
        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
            
            with main_output:
                main_output.clear_output()
                print("⚠️ Invalid date range.")
            with output_returns:
                output_returns.clear_output()
                print("⚠️ Invalid date range.")
            with vol_output:
                vol_output.clear_output()
            with perf_output:
                perf_output.clear_output()
            with drawdown_output:
                drawdown_output.clear_output()
            with frontier_output:
                frontier_output.clear_output()
                
            return
            
        constraint_df = pd.DataFrame(constraints)
        cons = None
        if not constraint_df.empty:
            try:
                cons = build_constraint(dataframe, constraint_df.to_numpy())
            except Exception as e:
                print("Error building constraints:", e)

        performance_pct.index = pd.to_datetime(performance_pct.index)

        cumulative_performance = performance_pct.loc[start_ts:end_ts]

        if cumulative_performance.empty:
            available_start = performance_pct.index.min().date()
            available_end = performance_pct.index.max().date()
            with output_returns:
                output_returns.clear_output(wait=True)
                print(f"⚠️ No data found for this date range. Available range: {available_start} → {available_end}")
            return

        # cumulative_performance = cumulative_performance.copy()
        
        cumulative_performance.iloc[0] = 0
        cumulative_results = (1 + cumulative_performance).cumprod() * 100
        
        portfolio_returns = rebalanced_time_series(range_prices, grid.data, frequency=rebalancing_frequency.value)
        cumulative_results=pd.concat([cumulative_results,portfolio_returns],axis=1)
        global_returns=cumulative_results.pct_change(fill_method=None)
                
        drawdown = (cumulative_results - cumulative_results.cummax()) / cumulative_results.cummax()
        rolling_vol_ptf=cumulative_results.pct_change(fill_method=None).rolling(window_vol.value).std()*np.sqrt(260)
        frontier_indicators, fig4 = get_frontier(range_returns,grid.data,cons)
        update_dropdown_options()
        


        with output_returns:
            output_returns.clear_output(wait=True)
            display(display_scrollable_df(rebalanced_metrics(cumulative_results)))
            display(display_scrollable_df(get_portfolio_risk(grid.data, range_prices, cumulative_results, benchmark_tracking_error.value)))
            display(display_scrollable_df(frontier_indicators))

        with perf_output:
            
            perf_output.clear_output(wait=True)
            fig = px.line(cumulative_results, title='Performance', width=800, height=400, render_mode = 'svg')
            fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
            fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
            fig.update_traces(textfont=dict(family="Arial Narrow", size=15))

            fig.show()
        with drawdown_output:
            
            drawdown_output.clear_output(wait=True)

            fig2 = px.line(drawdown, title='Drawdown', width=800, height=400, render_mode = 'svg')
            fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
            fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
            fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))

            fig2.show()

        with vol_output:
            vol_output.clear_output(wait=True)

            fig3 = px.line(rolling_vol_ptf, title="Portfolio Rolling Volatility", render_mode = 'svg')
            fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400) 
            fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
            fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
            fig3.show()
            
        with frontier_output:
            frontier_output.clear_output(wait=True)

            fig4.update_layout(width=800, height=400,title={'text': "Efficient Frontier"})
            fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))

            fig4.show()
            # display(display_scrollable_df(cumulative_results))
            # display(display_scrollable_df(drawdown))

        
    # start_date_perf.observe(updated_cumulative_perf)
    # end_date_perf.observe(updated_cumulative_perf)
    refresh_perf_button=widgets.Button(description='Refresh')
    refresh_perf_button.on_click(updated_cumulative_perf)
    
    # --- optimization ---
    rebalancing_frequency = widgets.Dropdown(description='Frequency', options=['Monthly', 'Quarterly', 'Yearly'], value='Monthly')
    strat = widgets.Dropdown(description='Strategy', options=options_strat, value='Minimum Variance')
    window_vol=widgets.IntText(
    value=252,
    description='Vol Window:',
    disabled=False
    )    
    calendar_output=widgets.Output()
    
    def show_graph(_):
        with calendar_output:
            
            calendar_output.clear_output(wait=True)
            
            if cumulative_results.empty or cumulative_results.shape[1]<2:
                print("⚠️ No performance data available yet. Please run an optimization first.")
                return

            if fund.value==benchmark.value:
                print("⚠️ Benchmark and Fund must be different.")
                return
            
            graphs=get_calendar_graph(cumulative_results, 
                               freq=frequency_graph.value, 
                               benchmark=benchmark.value, 
                               fund=fund.value)
            
            return_and_vol_graph=widgets.Output()
            sharpe_and_te_graph=widgets.Output()
            keys=list(graphs.keys())
            
            with return_and_vol_graph:
                graphs[keys[0]].show()
                graphs[keys[2]].show()
            with sharpe_and_te_graph:
                graphs[keys[1]].show()
                graphs[keys[3]].show()   
            
            ui=widgets.HBox([return_and_vol_graph,sharpe_and_te_graph])
            display(ui)
            
    graph_button=widgets.Button(description='Update Perf',button_style='info')
    graph_button.on_click(show_graph)

    optimize_btn = widgets.Button(description='Optimize Portfolio', button_style='primary')
    grid = DataGrid(pd.DataFrame(), editable=True, layout={"height": "250px"})
    

    def update_dropdown_options():
        """Safely refresh fund/benchmark dropdowns after cumulative_results is updated."""
        if 'cumulative_results' not in globals() or cumulative_results.empty:
            return

        global fund_names
        
        options = list(cumulative_results.columns)
    
        # Fund options exclude the currently selected benchmark
    
        fund.options = options
        benchmark.options = options
        
        fund.value = options[0]
        benchmark.value = options[1]
        fund_names=list(grid.data.index)
        benchmark_tracking_error.options=grid.data.index

        selected_fund.options=grid.data.index

        selected_bench.options=grid.data.index

        
        selected_fund_var.options=grid.data.index
        
    def on_optimize_clicked(_):
        global fund_names,grid

        with constraint_output:
            constraint_output.clear_output(wait=True)
            if dataframe.empty or returns_to_use.empty:
                print("⚠️ Load price data before optimizing.")
                return
            
            constraint_df = pd.DataFrame(constraints)
            cons = None
            if not constraint_df.empty:
                try:
                    cons = build_constraint(dataframe, constraint_df.to_numpy())
                except Exception as e:
                    print("Error building constraints:", e)
                    
            portfolio = RiskAnalysis(returns_to_use.loc[start_date_perf.value:end_date_perf.value])
            sharpe = portfolio.optimize("sharpe_ratio")
            minvar = portfolio.optimize("minimum_variance")
            rp = portfolio.optimize("risk_parity")
            max_div=portfolio.optimize("maximum_diversification")
            
            sharpe_c = minvar_c = rp_c =max_div_c=eigen_portfolio_c= None
            equal_weights = np.ones(returns_to_use.shape[1]) / returns_to_use.shape[1]
            eigen_portfolio=portfolio.optimize("eigenportfolio")
            
            if cons is not None:
                sharpe_c = portfolio.optimize("sharpe_ratio", constraints=cons)
                minvar_c = portfolio.optimize("minimum_variance", constraints=cons)
                rp_c = portfolio.optimize("risk_parity", constraints=cons)
                max_div_c=portfolio.optimize("maximum_diversification",constraints=cons)
                eigen_portfolio_c=portfolio.optimize("eigenportfolio",constraints=cons)
                
            allocation = {
                'Optimal Portfolio': sharpe.tolist(),
                'Constrained Optimal Portfolio': sharpe_c.tolist() if sharpe_c is not None else sharpe.tolist(),
                'Min Variance': minvar.tolist(),
                'Constrained Min Var': minvar_c.tolist() if minvar_c is not None else minvar.tolist(),
                'Max Diversification':max_div.tolist(),
                'Max Diversification Constrained':max_div_c.tolist() if max_div_c is not None else max_div.tolist(),
                'Risk Parity': rp.tolist(),
                'Constrained RP': rp_c.tolist() if rp_c is not None else rp.tolist(),
                'Eigen Portfolio':eigen_portfolio.tolist(),
                'Eigen Portfolio Constrained': eigen_portfolio_c.tolist() if eigen_portfolio_c is not None else eigen_portfolio.tolist(),
                'Equal Weighted':equal_weights.tolist()}

            allocation_df = pd.DataFrame(allocation, index=dataframe.columns).T.round(4)
            if set(current_weights.index).issubset(dataframe.columns):
                allocation_df = allocation_df.combine_first(current_weights.T).fillna(0)
            
            constraint_container = {'constraints_dataframe': constraints,'constraints':cons, 'allocation_df': allocation_df}
            grid.data = allocation_df
            
            benchmark_tracking_error.options=grid.data.index
            benchmark_tracking_error.value=grid.data.index[0]

            selected_fund.options=grid.data.index
            selected_fund.value=grid.data.index[0]
    
            selected_bench.options=grid.data.index
            selected_bench.value=grid.data.index[0]
    
            selected_fund_var.options=grid.data.index
            selected_fund_var.value=grid.data.index[0]
            
            with constraint_output:
                constraint_output.clear_output(wait=True)
                display(display_scrollable_df(pd.DataFrame(constraints))) 
            
            reset_stress(None)
            
    def get_result(_):
        nonlocal constraint_container
        global rolling_optimization, performance_pct, performance_fund, dates_end, quantities,quantities_core,quantities_overlay
    
        with strategy_output:
            strategy_output.clear_output(wait=True)
            if dataframe.empty or returns_to_use.empty:
                print("⚠️ Load price data before optimizing.")
                return
    
            # Build constraints
            cons = None
            if constraints:
                try:
                    cons = build_constraint(dataframe, pd.DataFrame(constraints).to_numpy())
                except Exception as e:
                    print("Error building constraints:", e)
    
            # Candidate anchors
            freq_map = {
                'Monthly': pd.offsets.BMonthEnd(),
                'Quarterly': pd.offsets.BQuarterEnd(),
                'Yearly': pd.offsets.BYearEnd()
            }
            offset = freq_map.get(rebalancing_frequency.value, pd.offsets.BMonthEnd())
            candidate_anchors = pd.DatetimeIndex(sorted(set(dataframe.index + offset)))
            if candidate_anchors.empty:
                candidate_anchors = pd.DatetimeIndex([returns_to_use.index[-1]])
    
            idx = returns_to_use.index.get_indexer(candidate_anchors, method='nearest')
            idx = np.array(idx)
            idx = idx[idx >= 0]
            selected_dates = returns_to_use.index[idx].tolist()
            dates_end = sorted(list(set(selected_dates + [returns_to_use.index[-1]])))
    
            if len(dates_end) < 2:
                print("⚠️ Not enough anchor dates to perform rolling optimization.")
                return
    
            # Prepare tasks
            strategy_key = dico_strategies[strat.value]
            tasks = [(returns_to_use.loc[dates_end[i]:dates_end[i+1]],dates_end[i], dates_end[i+1],strategy_key) for i in range(len(dates_end)-1)]
            overlays_tasks = [
                (
                    returns_to_use.loc[dates_end[i]:dates_end[i+1]],
                    dates_end[i],
                    dates_end[i+1],
                    dico_strategies[row['Strategy']]
                )
                for i in range(len(dates_end)-1)
                for row in overlays
            ]
            
            # Combine all tasks
            all_tasks = tasks + overlays_tasks
            # Run with threads
            global results
            results = {}
            def worker(subset,start, end,strategy_key):
                if subset.empty or len(subset) < 2:
                    return None
                try:
                    risk = RiskAnalysis(subset)
                    if cons:
                        opt = risk.optimize(objective=strategy_key, constraints=cons)
                    else:
                        opt = risk.optimize(objective=strategy_key)
                    return subset.index[-1], np.round(opt, 6),strategy_key
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                futures = {
                    executor.submit(worker, subset, start, end, strat): (subset, start, end, strat)
                    for subset, start, end, strat in all_tasks
                }
            
                for future in as_completed(futures):
                    out = future.result()
                    if out is not None:
                        date_key, weights, strategy_selected = out
            
                        if strategy_selected not in results:
                            results[strategy_selected] = {}
            
                        results[strategy_selected][date_key] = weights
            
            # rolling_optimization = pd.DataFrame(results[strategy_key], index=dataframe.columns).T.sort_index()
            rolling_optimization=pd.DataFrame(results[strategy_key], index=dataframe.columns).T.sort_index()
            total_overlay = pd.DataFrame(0, index=rolling_optimization.index, columns=rolling_optimization.columns)
            core_weights = 1
            overlays_df = pd.DataFrame(overlays)
            core_strat = rolling_optimization.copy()  # keep copy of core strategy
            
            # Accumulate overlays
            for _, row in overlays_df.iterrows():
                strat_key = dico_strategies[row['Strategy']]
                overlay_df = (
                    pd.DataFrame(results[strat_key], index=dataframe.columns)
                    .T
                    .sort_index()
                    * row['Limit']
                )
            
                # Add overlay to total_overlay
                total_overlay = total_overlay.add(overlay_df, fill_value=0)
            
                # Update remaining core weight
                core_weights -= row['Limit']
            
            # Safety check
            if core_weights < 0:
                raise ValueError("⚠️ Overlay weights exceed 100%")
            
            # Combine core + overlays
            rolling_optimization = rolling_optimization * core_weights + total_overlay
            
            if not rolling_optimization.empty:
                first_row = pd.Series(1 / len(dataframe.columns), index=dataframe.columns, name=dates_end[0])
                rolling_optimization = pd.concat([pd.DataFrame([first_row]), rolling_optimization])
                core_strat= pd.concat([pd.DataFrame([first_row]), core_strat])
                total_overlay= pd.concat([pd.DataFrame([first_row]), total_overlay/(1-core_weights)])
                
            core_output=widgets.Output()
            overlay_weights_output=widgets.Output()
            final_weights_output=widgets.Output()

            with core_output:
                display(Markdown("### Core Strategy"))

                display(display_scrollable_df(core_strat))
            with overlay_weights_output:

                display(Markdown("### Tactical Allocation"))

                display(display_scrollable_df(total_overlay))   
                
            with final_weights_output:

                display(Markdown("### Final Strategy"))

                display(display_scrollable_df(rolling_optimization)),

            strategies_decomposition = widgets.HBox([final_weights_output,core_output,overlay_weights_output
            ])
            
            display(strategies_decomposition)
            
            model=pd.DataFrame(rolling_optimization.iloc[-2])
            model.columns=['Model']
            if not 'Model' in grid.data.index:
                grid.data=pd.concat([grid.data,model.T],axis=0)
                
            quantities = rebalanced_dynamic_quantities(dataframe, rolling_optimization)
            quantities_core = rebalanced_dynamic_quantities(dataframe, core_strat)
            quantities_overlay = rebalanced_dynamic_quantities(dataframe, total_overlay)

            performance_fund = pd.DataFrame({'Fund': (quantities * dataframe).sum(axis=1),
                                             'Core':(quantities_core * dataframe).sum(axis=1),
                                             'Overlay':(quantities_overlay * dataframe).sum(axis=1)})
            
            if 'BTCUSDT' in dataframe.columns:
                performance_fund['Bitcoin'] = dataframe['BTCUSDT']
            performance_pct = performance_fund.pct_change(fill_method=None)
            
            cumulative=(1+performance_pct).cumprod()*100
            drawdown=pd.DataFrame((cumulative-cumulative.cummax()))/cumulative.cummax()            
            date_drawdown=drawdown.idxmin().dt.date
            max_drawdown=drawdown.min()
            

            metrics=pd.DataFrame()
            metrics['Returns']=performance_fund.iloc[-2]/performance_fund.iloc[0]
            metrics['Volatility']=performance_pct.std()*np.sqrt(252)
            metrics['Sharpe Ratio']=(1+metrics['Returns'])**(1/len(set(returns_to_use.index.year)))/metrics['Volatility']
            metrics['Drawdown']=max_drawdown
            metrics['Date Drawdown']=date_drawdown
            excess_returns_to_btc = performance_pct.loc[:, performance_pct.columns != 'Bitcoin'].sub(
                performance_pct['Bitcoin'], axis=0
            )
            metrics['Tracking Error to Bitcoin']=(excess_returns_to_btc).std()*np.sqrt(252)
            
            excess_returns_to_core = performance_pct.loc[:, performance_pct.columns != 'Core'].sub(
                performance_pct['Core'], axis=0
            )
            metrics['Tracking Error to Core']=(excess_returns_to_core).std()*np.sqrt(252)
            metrics=metrics.fillna(0).T

            display(display_scrollable_df(metrics.round(4)))
            updated_cumulative_perf(None)
            show_graph(None)
            get_holdings(None)
            
    optimize_btn.on_click(on_optimize_clicked)
    results_button=widgets.Button(description='Get Results',button_style='info')
    results_button.on_click(get_result)

    positions_output=widgets.Output()
    holding_output=widgets.Output()
    loading_bar_pnl = widgets.IntProgress(description='Loading P&L...',min=0, max=100,style={'description_width': '150px'})
    
    def get_holdings(_):

        global holding_tickers,current_weights,pnl
        
        quantities_api=Binance.binance_api.user_asset()
        current_quantities=pd.DataFrame(quantities_api).sort_values(by='free',ascending=False)
        current_quantities['asset']=current_quantities['asset']+'USDT'
        current_quantities=current_quantities.set_index('asset')
        
        current_positions=Binance.get_inventory().round(4)
        current_positions.columns=['Current Portfolio in USDT','Current Weights']
        amount=current_positions.loc['Total']['Current Portfolio in USDT']
        condition=current_positions.index!='Total'

        holding_tickers=current_positions.index[condition]
        holding_tickers=holding_tickers.to_list()
        
        inventory_weights=(current_positions['Current Weights'].apply(lambda x: np.round(x,4))).to_dict()
        inventory_weights.pop('Total')
        inventory_weights.pop('USDCUSDT')
        
        if "USDTUSDT" in holding_tickers:
            inventory_weights.pop('USDTUSDT')
        else: 
            pass
            
        current_weights=pd.DataFrame(inventory_weights.values(),index=inventory_weights.keys(),columns=['Current Weights'])
                
        with positions_output:
            positions_output.clear_output(wait=True)
        
            if dataframe.empty or returns_to_use.empty:
                print("⚠️ Load Prices.")
        
            elif quantities.empty:
                print("⚠️ Load Model.")
        
            else:
                last_prices = Binance.get_price(list(quantities.iloc[-1].keys()))
                positions = pd.DataFrame(quantities.iloc[-1] * last_prices).T
        
                amount_ex_out_of_positions = (
                    current_positions.loc[
                        ~(current_positions.index.isin(positions.index) | (current_positions.index == 'Total')),
                        'Current Portfolio in USDT'
                    ].sum()
                )
        
                positions['Weights Model'] = positions / positions.sum()
                positions['Model (without out of Model Positions)'] = (
                    positions['Weights Model'] * (amount - amount_ex_out_of_positions)
                )
                positions['Model'] = positions['Weights Model'] * amount
        
                portfolio = pd.concat(
                    [positions[['Model', 'Model (without out of Model Positions)', 'Weights Model']],
                     current_positions.loc[condition]],
                    axis=1
                ).fillna(0)
        
                portfolio['Spread'] = portfolio['Current Portfolio in USDT'] - portfolio['Model']
                portfolio.loc['Total'] = portfolio.sum(axis=0)
                portfolio = (
                    portfolio.loc[~(portfolio == 0).all(axis=1)]
                    .sort_values(by='Weights Model', ascending=False)
                    .round(4)
                )
        
                display(display_scrollable_df(portfolio))
        
        with holding_output:
            holding_output.clear_output(wait=True)
        
            if book_cost.empty and realized_pnl.empty:
                display(display_scrollable_df(current_positions))
                print("⚠️ P&L not Computed.")
            else:
                last_book_cost = book_cost.iloc[-1] if not book_cost.empty else pd.Series(dtype=float)
                realized_pnl_filled = realized_pnl if not realized_pnl.empty else pd.Series(dtype=float)
        
        
                pnl = pd.concat(
                    [last_book_cost, last_book_cost, current_positions.loc[condition], realized_pnl_filled],
                    axis=1
                )
                pnl.columns = ['Average Cost', 'Book Cost', 'Price in USDT', 'Weights', 'Realized P&L']
        
                pnl['Book Cost'] = (pnl['Book Cost'] * current_quantities['free'].astype(float)).fillna(0)
                pnl['Unrealized P&L'] = (pnl['Price in USDT'] - pnl['Book Cost']).round(2)
                pnl = pnl.fillna(0)
                pnl['Weights'] = pnl['Weights'].round(4)
        
                pnl['Total P&L'] = pnl['Unrealized P&L'] #+ pnl['Realized P&L']
                pnl.loc['Total'] = pnl.sum()
                pnl.loc['Total', 'Average Cost'] = np.nan
                pnl.loc['Total', 'Book Cost'] = pnl.loc['Total', 'Price in USDT'] - pnl.loc['Total', 'Total P&L']
        
                if pnl.loc['Total', 'Book Cost'] != 0:
                    pnl['Total P&L %'] = pnl['Total P&L'] / pnl.loc['Total', 'Book Cost'] * 100
                else:
                    pnl['Total P&L %'] = 0
        
                display(display_scrollable_df(pnl.sort_values(by='Weights', ascending=False).round(4)))                
                display(display_scrollable_df(trades))          
    
    def get_pnl_on_click(_):
        global book_cost,realized_pnl,profit_and_loss,trades
        
        url='https://github.com/niroojane/Risk-Management/raw/refs/heads/main/Trade%20History%20Reconstructed.xlsx'
        trade_history = read_excel_from_url(url)
        
        if trade_history is None:
            raise FileNotFoundError("Trade history could not be loaded. Execution stopped.")  
        loading_bar_pnl.value=0
        
        with holding_output:
            display(loading_bar_pnl)
            
        trades=Pnl_calculation.get_trade_in_usdt(trade_history)
        loading_bar_pnl.value+=100/3
        book_cost=Pnl_calculation.get_book_cost(trades)
        book_cost['MANTRAUSDT']=book_cost['OMUSDT']/4
        loading_bar_pnl.value+=100/3
        realized_pnl,profit_and_loss=Pnl_calculation.get_pnl(book_cost,trades)
        trades=trades.set_index('Date(UTC)')
        loading_bar_pnl.value+=100/3
        with holding_output:
            holding_output.clear_output()
            
        get_holdings(None)

    pnl_button=widgets.Button(description='Get P&L',button_style='info')
    pnl_button.on_click(get_pnl_on_click)
            
    position_button=widgets.Button(description='Get Positions',button_style='info')
    position_button.on_click(get_holdings)
    
    # --- layout ---
    overlay_ui=widgets.VBox([widgets.HBox([widgets.VBox([strat_overlay,dropdown_strategy_limit]),
                                           widgets.VBox([add_strategy_btn,clear_strategy_btn])]),
                                           overlay_output])

    allocation_ui=widgets.VBox([widgets.HBox([
            widgets.VBox([dropdown_asset, dropdown_sign, dropdown_limit]),
            widgets.VBox([add_constraint_btn, clear_constraints_btn, optimize_btn])]),
                                constraint_output,
        grid,overlay_ui,
        widgets.HBox([button_add,button_clear,results_button])])


    constraint_ui = widgets.VBox([widgets.HBox([start_date_perf, end_date_perf,refresh_perf_button]),
                                               main_output,
        widgets.VBox([strat, rebalancing_frequency,benchmark_tracking_error,window_vol]),
        allocation_ui,strategy_output,
        widgets.HBox([start_date_perf, end_date_perf,refresh_perf_button]),
        output_returns,widgets.HBox([perf_output, drawdown_output]),widgets.HBox([vol_output,frontier_output])
    ])

    universe_ui = widgets.VBox([
        widgets.HBox([n_crypto, start_date, data_button]),
        scope_output,
        widgets.HBox([start_date_perf, end_date_perf,refresh_perf_button]),
        main_output,price_output
    ])
    
    calendar_perf=widgets.VBox([widgets.HBox([frequency_graph,fund,benchmark,graph_button]),calendar_output])
    positions_ui=widgets.VBox([widgets.HBox([position_button,pnl_button]),positions_output,holding_output])
    rebalancing_frequency_pnl=widgets.Dropdown(description='Frequency:', options=['Yearly','Quarterly','Monthly'], value='Quarterly')

    #------------Rik Tab------------#
    global var_scenarios, cvar_scenarios, fund_results,grid_stress,grid_mean_shock
    risk_output = widgets.Output()
    
    start_date_perf_risk = widgets.DatePicker(value=start_perf_date, layout=widgets.Layout(width='350px'))
    end_date_perf_risk = widgets.DatePicker(value=datetime.date.today(), layout=widgets.Layout(width='350px'))
    # ---------- Ex-Ante / Risk Contribution ----------
    def update_fund_display(_):
        
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with ex_ante_output:
                ex_ante_output.clear_output()   
                print("⚠️ Invalid start or end date.")
            with risk_output:
                risk_output.clear_output()   
            return
                
        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts: 
            with ex_ante_output:
                ex_ante_output.clear_output()   
                print("⚠️ Error with date range.")
            with risk_output:
                risk_output.clear_output()   
                return
    
        if dataframe.empty or returns_to_use.empty or grid.data.empty:
            with ex_ante_output:
                ex_ante_output.clear_output()   
                print("⚠️ Please compute optimization results first.")
            with risk_output:
                risk_output.clear_output()   
                return    
        range_prices = dataframe.loc[start_ts:end_ts]
    
        if range_prices.empty:
            with ex_ante_output:
                ex_ante_output.clear_output()   
                print("⚠️ No data available in selected date range.")
            with risk_output:
                risk_output.clear_output()   
            return
    
        range_returns = range_prices.pct_change(fill_method=None).dropna()
 
        if grid.data.empty:
            with ex_ante_output:
                ex_ante_output.clear_output()
                print("⚠️ No Allocation.")
            with risk_output:
                risk_output.clear_output()   
            return     
            
        portfolio = RiskAnalysis(range_returns)
     
        selected_weights = grid.data.loc[selected_fund.value]
        
        decomposition = pd.DataFrame(portfolio.var_contrib(selected_weights)[0])*100

        # decomposition_vol = pd.DataFrame(portfolio.var_contrib(selected_weights)[0])*100
        # decomposition_vol.loc['Total'] = decomposition_vol.sum(axis=0)
        
        quantities_rebalanced = rebalanced_portfolio(range_prices, selected_weights,frequency=rebalancing_frequency_pnl.value) / range_prices
        quantities_buy_hold = buy_and_hold(range_prices, selected_weights) / range_prices
        
        cost_rebalanced = rebalanced_book_cost(range_prices, quantities_rebalanced)
        cost_buy_and_hold = rebalanced_book_cost(range_prices, quantities_buy_hold)
        
        mtm_rebalanced = quantities_rebalanced * range_prices
        mtm_buy_and_hold = quantities_buy_hold * range_prices
        
        pnl_buy_and_hold=pd.DataFrame((mtm_buy_and_hold-cost_buy_and_hold).iloc[-1])
        pnl_buy_and_hold.columns=['Profit and Loss (Buy and Hold)']
        
        pnl_rebalanced=pd.DataFrame((mtm_rebalanced-cost_rebalanced).iloc[-1])
        pnl_rebalanced.columns=['Profit and Loss (Rebalanced)']
        
        profit_and_loss_simulated = pd.concat([pnl_buy_and_hold, pnl_rebalanced, decomposition], axis=1)
        profit_and_loss_simulated.loc['Total'] = profit_and_loss_simulated.sum(axis=0)
        profit_and_loss_simulated=profit_and_loss_simulated.fillna(0)
        


        with risk_output:
            risk_output.clear_output(wait=True)
            display(Markdown("### Performance and Risk Contribution"))
            display(display_scrollable_df(
                profit_and_loss_simulated.sort_values(by='Vol Contribution', ascending=False)
            ))
            # display(display_scrollable_df(
            #     decomposition_vol.sort_values(by='Vol Contribution', ascending=False)
            # ))


    def on_fund_change(change):
        if change['name'] == 'value' and change['new'] in grid.data.index:
            update_fund_display(change['new'])

    selected_fund.observe(on_fund_change)
    
    def on_freq_change(change):
        if change['name'] == 'value' and change['new'] in rebalancing_frequency_pnl.options:
            update_fund_display(change['new'])
    
    rebalancing_frequency_pnl.observe(on_freq_change,names='value')
         
    # ---------- Ex-Ante Metrics ----------
    ex_ante_output = widgets.Output()

    def ex_ante_metrics(bench_name):
        
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with ex_ante_output:
                ex_ante_output.clear_output()   
                print("⚠️ Invalid start or end date.")

            with risk_output:
                risk_output.clear_output()   
            return

        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts: 
            with ex_ante_output:
                ex_ante_output.clear_output()   
                print("⚠️ Error with date range.")
            with risk_output:
                risk_output.clear_output()   
                return
    
        if dataframe.empty or returns_to_use.empty or grid.data.empty:
            with ex_ante_output:
                ex_ante_output.clear_output()   
                print("⚠️ Please compute optimization results first.")
            with risk_output:
                risk_output.clear_output()   
                return
    
        # ✅ Safe slicing
        range_prices = dataframe.loc[start_ts:end_ts]
    
        if range_prices.empty:
            with ex_ante_output:
                ex_ante_output.clear_output()   
                print("⚠️ No data available in selected date range.")
            return
    
        range_returns = range_prices.pct_change(fill_method=None).dropna()
    
        if range_returns.empty:
            with ex_ante_output:
                ex_ante_output.clear_output()   
                print("⚠️ Not enough data to compute returns.")
            return
        
        vol_ex_ante = {}
        tracking_error_ex_ante = {}

        portfolio = RiskAnalysis(range_returns)
        
        for idx in grid.data.index:
            vol_ex_ante[idx] = portfolio.variance(grid.data.loc[idx])
            tracking_error_ex_ante[idx] = portfolio.variance(grid.data.loc[idx] - grid.data.loc[bench_name])

        data = {
            'Vol Ex Ante': vol_ex_ante,
            'Tracking Error Ex Ante': tracking_error_ex_ante
        }
        ex_ante_dataframe = pd.DataFrame(data)
        
        with ex_ante_output:
            ex_ante_output.clear_output(wait=True)
            display(Markdown("### Ex Ante Metrics"))
            display(display_scrollable_df(ex_ante_dataframe))

    def on_bench_change(change):
        if change['name'] == 'value' and change['new'] in grid.data.index:
            ex_ante_metrics(change['new'])
            
    def update_contrib_and_ex_ante(_):
        update_fund_display(None)
        ex_ante_metrics(selected_bench.value)
        
        
    ex_ante_metrics(selected_bench.value)
    selected_bench.observe(on_bench_change, names='value')

    end_date_perf_risk.observe(update_contrib_and_ex_ante)
    start_date_perf_risk.observe(update_contrib_and_ex_ante)

    # ---------- VaR / CVaR Simulation ----------
    var_output = widgets.Output()
    var_scenarios, cvar_scenarios, fund_results = {}, {}, {}

    stress_factor = widgets.BoundedFloatText(value=1.0, min=1.0, max=3.0, step=0.1, description='Stress Factor:')
    mean_factor = widgets.BoundedFloatText(value=1.0, min=0.0, max=3.0, step=0.1, description='Mean Factor:')
    iterations = widgets.BoundedIntText(value=10000, min=1000, max=100000, step=1, description='Iterations:')
    num_scenarios = widgets.BoundedIntText(value=100, min=1, max=1000, step=1, description='Scenarios:')
    var_centile = widgets.BoundedFloatText(value=0.05, min=0, max=1, step=0.01, description='VaR Centile:')
    loading_bar_var = widgets.IntProgress(description='Loading scenarios...',min=0, max=100,style={'description_width': '150px'})
    
    grid_stress = DataGrid(pd.DataFrame(),
        editable=True,
        layout={"height": "250px"}
    )
    
    grid_mean_shock = DataGrid(pd.DataFrame(),
        editable=True,
        layout={"height": "250px"}
                              )   

    def update_stress_grid(change):
    
        global grid_stress
    
        grid_stress.unobserve(update_stress_grid)
    
        new_matrix = set_symmetric(grid_stress.data.to_numpy(), limit=2)
    
        new_dataframe = pd.DataFrame(
            new_matrix,
            columns=dataframe.columns,
            index=dataframe.columns
        )
    
        grid_stress.data = new_dataframe
        grid_stress.observe(update_stress_grid)

        grid_mean_shock.observe(update_stress_grid)

        update_stress_data(None)
        
    def update_stress_data(_):
         
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with var_output:
                var_output.clear_output()   
                print("⚠️ Invalid start or end date.")
            return
                   
        start_ts = pd.to_datetime(start_date_perf_risk.value)
        end_ts = pd.to_datetime(end_date_perf_risk.value)

        range_prices=dataframe.loc[start_ts:end_ts]
        range_returns=range_prices.pct_change(fill_method=None)
        stress_vol_output = widgets.Output()
        stress_mean_output = widgets.Output()

        cov=range_returns.cov()
        
        stress_matrix=np.diag(np.diag(grid_stress.data))
        stressed_cov = stress_matrix @ cov @ stress_matrix
        stressed_std=np.sqrt(np.diag(stressed_cov))

        vol = stressed_std*np.sqrt(250)
        shocked_means=(range_returns.mean()*grid_mean_shock.data['Mean Shock'])*250
        
        corr_matrix = stressed_cov / np.outer(stressed_std, stressed_std)
        corr_matrix=corr_matrix+np.tril(grid_stress.data)+np.tril(grid_stress.data).T
        corr_matrix=np.clip(corr_matrix,-1,1)
        corr_matrix=cov_nearest(corr_matrix)
        
        corr_dataframe=pd.DataFrame(corr_matrix,index=range_returns.columns,columns=range_returns.columns)
        mean_shocked_dataframe=pd.concat([range_returns.mean()*250,shocked_means],axis=1)
        mean_shocked_dataframe.columns=['Means','Shocked Means']

        original_vol=range_returns.std()*np.sqrt(250)
        vol_dataframe=pd.DataFrame(index=range_returns.columns)
        
        vol_dataframe['Vol']=original_vol
        vol_dataframe['Shocked Vol']=vol

        original_corr=range_returns.corr()
        expected_data=pd.concat([mean_shocked_dataframe,vol_dataframe],axis=1)

        correlation_output_old= widgets.Output()
        correlation_output_new= widgets.Output()
        mean_output= widgets.Output()
        
        with stress_grid_output:
            stress_grid_output.clear_output(wait=True)

            with stress_vol_output:
                
                display(Markdown('## Correlation Shock'))
                display(grid_stress)
            with stress_mean_output:
                display(Markdown('## Returns Shock'))
                display(grid_mean_shock)

            with mean_output:
                display(Markdown('## Mean and Vol Changes'))
                display(display_scrollable_df(expected_data))
            
            with correlation_output_old:
                display(Markdown('### Original Correlation'))
                display(display_scrollable_df(original_corr))
            
            with correlation_output_new:
                display(Markdown('### New Correlation'))
                
                display(display_scrollable_df(corr_dataframe))

            ui_corr=widgets.HBox([correlation_output_old,correlation_output_new])
            ui_mean=widgets.VBox([stress_vol_output,stress_mean_output,mean_output])

            display(ui_mean)

            display(Markdown('## Correlation Changes'))
            display(ui_corr)
            
    def reset_stress(_):
    
        global grid_stress,grid_mean_shock

        if returns_to_use.empty:
            with stress_grid_output:
                print("⚠️ Load Prices")
            return

        stress_vec=np.linspace(stress_factor.value,stress_factor.value,returns_to_use.shape[1])
        stress_matrix = np.diag(stress_vec)
        
        stress_mean=np.linspace(mean_factor.value,mean_factor.value,returns_to_use.shape[1])

        grid_stress = DataGrid(
            pd.DataFrame(
                stress_matrix,
                columns=dataframe.columns,
                index=dataframe.columns
            ),
            editable=True,
            layout={"height": "250px"}
        )
        
        grid_mean_shock = DataGrid(
            pd.DataFrame(
                stress_mean,
                columns=['Mean Shock'],
                index=dataframe.columns
            ),
            editable=True,
            layout={"height": "250px"}
        )
        
        grid_stress.observe(update_stress_grid)
        grid_mean_shock.observe(update_stress_data)
        
        update_stress_data(None)
        

    stress_grid_output = widgets.Output()
    stress_grid_button = widgets.Button(description="Refresh")
    stress_grid_button.on_click(reset_stress)

    
    def get_var_metrics(_):
        global var_scenarios, cvar_scenarios, fund_results
        
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with var_output:
                var_output.clear_output()   
                print("⚠️ Invalid start or end date.")
            return
            
        with var_output:
            var_output.clear_output()   
            if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
                print("⚠️ Error with date range.")
                return
            if returns_to_use.empty:
                print('⚠️Load Prices.')
                return
            if dataframe.empty or returns_to_use.empty or grid.data.empty:
                print("⚠️ Please compute optimization results first.")
                return
                
            horizon = 1/250
            spot = dataframe.iloc[-1]
            theta = 2
            stress_matrix=grid_stress.data.to_numpy()
            mean_shock_vec=grid_mean_shock.data['Mean Shock']
            
            distrib_functions = {
                'multivariate_distribution': (iterations.value, stress_matrix,mean_shock_vec),
                'gaussian_copula': (iterations.value, stress_matrix,mean_shock_vec),
                't_copula': (iterations.value, stress_matrix,mean_shock_vec),
                'gumbel_copula': (iterations.value, theta,stress_factor.value,mean_shock_vec),
                'monte_carlo': (spot, horizon, iterations.value, stress_matrix,mean_shock_vec)
            }
            
            range_prices=dataframe.loc[start_ts:end_ts]
            range_returns=range_prices.pct_change(fill_method=None)
            
            portfolio = RiskAnalysis(range_returns)
            var_scenarios, cvar_scenarios, fund_results = {}, {}, {}

            display(loading_bar_var)
            
            def process_index(index):
                vs, cvs = {}, {}
                for func_name, args in distrib_functions.items():
                    func = getattr(portfolio, func_name)
                    scenarios = {}
    
                    for i in range(num_scenarios.value):
                        if func_name == 'monte_carlo':
                            distrib = pd.DataFrame(func(*args)[1], columns=portfolio.returns.columns)
                        else:
                            distrib = pd.DataFrame(func(*args), columns=portfolio.returns.columns)
    
                        distrib = distrib * grid.data.loc[index]
                        distrib = distrib[distrib.columns[grid.data.loc[index] > 0]]
                        distrib['Portfolio'] = distrib.sum(axis=1)
    
                        results = distrib.sort_values(by='Portfolio').iloc[int(distrib.shape[0] * var_centile.value)]
                        scenarios[i] = results
    
                    scenario = pd.DataFrame(scenarios).T
                    mean_scenario = scenario.mean()
                    index_cvar = scenario['Portfolio'] < mean_scenario['Portfolio']
                    cvar = scenario.loc[index_cvar].mean()
    
                    vs[func_name] = mean_scenario
                    cvs[func_name] = cvar
    
                fund_result = {
                    'Value At Risk': mean_scenario.loc['Portfolio'],
                    'CVaR': cvar.loc['Portfolio']
                }
    
                return index, vs, cvs, fund_result
    
            # Threaded execution
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_index, idx): idx for idx in grid.data.index}
                for future in as_completed(futures):
                    idx, vs, cvs, fund_result = future.result()
                    var_scenarios[idx] = vs
                    cvar_scenarios[idx] = cvs
                    fund_results[idx] = fund_result
                    loading_bar_var.value += 100 / len(grid.data.index)
    
            loading_bar_var.value = 0
    
        display_var_results(selected_fund_var.value)
        loading_bar_var.value=0
        
    
    global result_var,result_cvar
    
    var_function_names={'Multivariate':'multivariate_distribution',
                   'Gaussian Copula':'gaussian_copula',
                   'Monte Carlo':'monte_carlo',
                   'Gumbel Copula':'gumbel_copula',
                   'T-Copula':'t_copula'}
    
    result_var=pd.DataFrame()
    result_cvar=pd.DataFrame()
    
    value_at_risk_trajectory_output=widgets.Output()
    selected_fund_to_decompose_var=widgets.Dropdown(options=['Fund','Historical Portfolio'],value='Fund',description='Fund:')
    var_risk_trajectory_refresh_button=widgets.Button(description='Refresh')
    var_trajectory_button=widgets.Button(description='Get VaR',button_style='success')
    func_name=widgets.Dropdown(description='Method',options=list(var_function_names.keys()),value='Gumbel Copula')

    window_var=widgets.IntText(
    value=252,
    description='Window:',
    disabled=False)
    
    def value_at_risk_trajectory(_):
        
        global result_var,result_cvar,series_dict,current_underlying_returns

        
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with value_at_risk_trajectory_output:
                value_at_risk_trajectory_output.clear_output(wait=True)
                print("⚠️ Invalid start or end date.")
            return
            
        with value_at_risk_trajectory_output:
            value_at_risk_trajectory_output.clear_output(wait=True)

            if dataframe.empty:
                print('⚠️Load Prices.')
                return
            if dataframe.empty or returns_to_use.empty or grid.data.empty:
                print("⚠️ Please compute optimization results first.")
                return
            
            if len(returns_to_use.loc[start_ts:end_ts]) < window_risk.value:
                print("⚠️ Date range is shorter than rolling window.")
                return    
            else:
                display(loading_bar)
                display(loading_bar_risk)
                
        range_prices=dataframe.loc[start_ts:end_ts]
        range_returns=range_prices.pct_change(fill_method=None)
        series_dict={}

        for key in grid.data.index:
            
            rebalanced_series=rebalanced_portfolio(dataframe,grid.data.loc[key])
            rebalanced_series_weights=rebalanced_series.apply(lambda x: x/rebalanced_series.sum(axis=1))
            buy_and_hold_series=buy_and_hold(dataframe,grid.data.loc[key])
            buy_and_hold_series_weights=buy_and_hold_series.apply(lambda x: x/buy_and_hold_series.sum(axis=1))
            series_dict['Rebalanced '+key]=rebalanced_series_weights.loc[start_ts:end_ts]
            series_dict['Buy and Hold '+key]=buy_and_hold_series_weights.loc[start_ts:end_ts]
        
        weights_ex_post=positions.copy()
        weights_ex_post=weights_ex_post.drop(columns=['USDTUSDT'])
        weights_ex_post=weights_ex_post.apply(lambda x: x/weights_ex_post['Total'])
        weights_ex_post=weights_ex_post.drop(columns=['Total'])
        weights_ex_post=weights_ex_post.fillna(0.0)
        
        if not quantities.empty:
            portfolio=quantities*dataframe
            model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
            series_dict['Fund']=model_weights.loc[start_ts:end_ts]
            
        if not quantities_core.empty:
            portfolio=quantities_core*dataframe
            model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
            series_dict['Core']=model_weights.loc[start_ts:end_ts]

        if not quantities_overlay.empty:
            portfolio=quantities_overlay*dataframe
            model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
            series_dict['Overlay']=model_weights.loc[start_ts:end_ts]

        tickers_combined=list(quantities.columns)+list(weights_ex_post.columns)
        tickers_combined=list(set(tickers_combined))
        
        current_underlying_prices=get_price_threading(tickers_combined,weights_ex_post.index[0].date())    
        current_underlying_returns=current_underlying_prices.pct_change(fill_method=None)

        horizon = 1/250
        spot = dataframe.iloc[-1]
        theta = 2
        stress_matrix=grid_stress.data.to_numpy()
        mean_shock_vec=grid_mean_shock.data['Mean Shock']

        distrib_functions = {
        'multivariate_distribution': (iterations.value, stress_factor.value,mean_factor.value),
        'gaussian_copula': (iterations.value, stress_factor.value,mean_factor.value),
        't_copula': (iterations.value, stress_factor.value,mean_factor.value),
        'gumbel_copula': (iterations.value, theta,stress_factor.value,mean_factor.value),
        'monte_carlo': (spot, horizon, iterations.value, stress_factor.value,mean_factor.value)}
        
        method=var_function_names[func_name.value]
        args=distrib_functions[method]      

        tasks=[(key,method,args,returns_to_use.loc[start_ts:end_ts],series_dict[key],window_var.value,var_centile.value) for key in series_dict]

        series_dict['Historical Portfolio']=weights_ex_post.loc[start_ts:end_ts]
        
        if method in ['gumbel_copula']:
            tasks.append(
             (
              'Historical Portfolio',
              method,
              args,
              current_underlying_returns.loc[weights_ex_post.index].loc[start_ts:end_ts],
              weights_ex_post.loc[start_ts:end_ts],
              window_var.value,
              var_centile.value
             )
            )
            
        results_dict_var={}
        results_dict_cvar={}
        
        loading_bar_risk.value = 0
        loading_bar_risk.max=len(tasks)
        
        for name,func,arg,returns,weight,window,centile in tasks:

            common_col=returns.columns.intersection(weight.columns)
            common_index=returns.index.intersection(weight.index)
            
            var_data,cvar_data=get_var_contribution(func,arg,returns.loc[common_index,common_col],weight.loc[common_index,common_col],window,centile)
            results_dict_var[name]=var_data['Portfolio']
            results_dict_cvar[name] =cvar_data['Portfolio']
            
            loading_bar_risk.value += 1
                        
        loading_bar_risk.value = loading_bar_risk.max
        
        with value_at_risk_trajectory_output:

            if not results_dict_var or not results_dict_cvar:
                print("⚠️ No risk results were computed.")
                return
            
            result_var = pd.concat(results_dict_var.values(), axis=1)
            result_cvar = pd.concat(results_dict_cvar.values(), axis=1)
            
            if result_var.shape[1] == 0 or result_cvar.shape[1] == 0:                
                print("⚠️ Risk results are empty.")
                return  

        result_var.columns=results_dict_var.keys()
        result_cvar.columns=results_dict_cvar.keys()

        selected_fund_to_decompose_var.options=result_var.columns
        selected_fund_to_decompose_var.value='Fund'
        
        show_var_graph(None)
        
    var_trajectory_button.on_click(value_at_risk_trajectory)
    
    def show_var_graph(_):
        
        global result_var,result_cvar
        
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with value_at_risk_trajectory_output:
                value_at_risk_trajectory_output.clear_output(wait=True)
                print("⚠️ Invalid start or end date.")
            return
            
        output1=widgets.Output()
        output2=widgets.Output()
        
        with value_at_risk_trajectory_output:
            
            value_at_risk_trajectory_output.clear_output(wait=True)
            
            if dataframe.empty:
                print('⚠️Load Prices.')
                return
            if dataframe.empty or returns_to_use.empty or grid.data.empty or global_returns.empty:
                print("⚠️ Please compute optimization results first.")
                return

            if result_var.empty or result_cvar.empty:
                print("⚠️ Load VaR History.")
                return
            
            if len(returns_to_use.loc[start_ts:end_ts]) < window_risk.value:
                print("⚠️ Date range is shorter than rolling window.")
                return    
                
            if not series_dict:
                print("⚠️ Weights Empty.")
                return
            else:

                horizon = 1/250
                spot = dataframe.iloc[-1]
                theta = 2
                stress_matrix=grid_stress.data.to_numpy()
                mean_shock_vec=grid_mean_shock.data['Mean Shock']
            
                distrib_functions = {
                'multivariate_distribution': (iterations.value, stress_factor.value,mean_factor.value),
                'gaussian_copula': (iterations.value, stress_factor.value,mean_factor.value),
                't_copula': (iterations.value, stress_factor.value,mean_factor.value),
                'gumbel_copula': (iterations.value, theta,stress_factor.value,mean_factor.value),
                'monte_carlo': (spot, horizon, iterations.value, stress_factor.value,mean_factor.value)}

                value_at_risk_trajectory_output.clear_output(wait=True)
                series_weights=series_dict[selected_fund_to_decompose_var.value]
                method=var_function_names[func_name.value]
                args=distrib_functions[method]

                if selected_fund_to_decompose_var.value!='Historical Portfolio':
                    common=series_weights.columns.intersection(returns_to_use.columns)
                    common_index=series_weights.index.intersection(returns_to_use.index)
                    var,cvar=get_var_contribution(method,args,returns_to_use.loc[common_index,common].loc[start_ts:end_ts],series_weights.loc[common_index,common].loc[start_ts:end_ts],window_var.value,var_centile.value)
                    # ret=global_returns[selected_fund_to_decompose_var.value]

                else:
                    #ret=performance_ex_post[selected_fund_to_decompose_var.value]
                    common=series_weights.columns.intersection(current_underlying_returns.columns)
                    common_index=series_weights.index.intersection(current_underlying_returns.index)
                    var,cvar=get_var_contribution(method,args,current_underlying_returns.loc[common_index].loc[start_ts:end_ts,common],series_weights.loc[common_index].loc[start_ts:end_ts,common],window_var.value,var_centile.value)

                
                # historical_var=ret.rolling(window_var.value).quantile(var_centile.value)
                # historical_cvar = ret.rolling(window_var.value).apply(
                #     lambda x: x[x <= x.quantile(var_centile.value)].mean(),
                #     raw=False
                # )
                    
                # var_model_tested=pd.concat([historical_var,historical_cvar,var['Portfolio'],cvar['Portfolio'],ret],axis=1)
                # var_model_tested.columns=['Historical VaR','Historical CVaR','Model VaR','Model CVaR','Return']

                with output1:
                    
                    fig = px.line(result_var.loc[start_ts:end_ts], title='Portfolios Value At Risk', width=800, height=400, render_mode = 'svg')
                    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Historical Portfolio","Fund"])
                    fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig.show()

                    fig4 = px.line(var, title='Value at Risk History', width=800, height=400, render_mode = 'svg')
                    
                    fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Portfolio"])
                    fig4.show()   
                        
                with output2:
                    
                    fig2 = px.line(result_cvar, title='Portfolio Expected Shortfall', width=800, height=400, render_mode = 'svg')
                    fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Historical Portfolio","Fund"])
                    fig2.show()
                    
                    fig3 = px.line(cvar, title='Expected Shortfall History', width=800, height=400, render_mode = 'svg')
                    fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Portfolio"])
                    fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig3.show()
                    
                    # fig5 = px.line(var_model_tested.dropna(), title='Model Backtest', width=800, height=400, render_mode = 'svg')
                    # fig5.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    # fig5.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Historical VaR","Model VaR"])
                    # fig5.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    # fig5.show()
                    
                ui=widgets.HBox([output1,output2])

                display(ui)
                
    var_risk_trajectory_refresh_button.on_click(show_var_graph)
          
    def display_var_results(fund_name):
        if fund_name not in var_scenarios:
            with var_output:
                var_output.clear_output(wait=True)
                print(f"⚠️ No VaR data found for '{fund_name}'. Run simulation first.")
            return

        columns = ['Multivariate', 'Gaussian Copula', 'T-Student Copula', 'Gumbel Copula', 'Monte Carlo']
        var_dataframe = pd.DataFrame(var_scenarios[fund_name])
        var_dataframe.columns = columns

        cvar_dataframe = pd.DataFrame(cvar_scenarios[fund_name])
        cvar_dataframe.columns = columns

        fund_results_dataframe = pd.DataFrame(fund_results).T

        with var_output:
            
            var_output.clear_output(wait=True)
            display(Markdown(f"### Summary"))
            display(display_scrollable_df(fund_results_dataframe))
            display(Markdown(f"### VaR Results for **{fund_name}**"))
            display(display_scrollable_df(var_dataframe))
            display(Markdown(f"### CVaR Results for **{fund_name}**"))
            display(display_scrollable_df(cvar_dataframe))

    def update_var_metrics(change):
        if change['name'] == 'value' and change['new'] in var_scenarios:
            display_var_results(change['new'])
    
    selected_fund_var.observe(update_var_metrics, names='value')

    get_var_button = widgets.Button(description='Run Simulation', button_style='info')
    get_var_button.on_click(get_var_metrics)

    # ---------- Layout ----------
    ex_ante_ui = widgets.VBox([widgets.HBox([start_date_perf_risk, end_date_perf_risk]),
        widgets.VBox([selected_fund, selected_bench,rebalancing_frequency_pnl]),
        risk_output,
        ex_ante_output
    ])

    var_ui = widgets.VBox([widgets.HBox([start_date_perf_risk, end_date_perf_risk]),
        widgets.VBox([selected_fund_var, stress_factor,mean_factor,iterations,num_scenarios,var_centile,stress_grid_output,widgets.HBox([get_var_button,stress_grid_button])]),
        var_output
    ])

    var_history_ui = widgets.VBox([widgets.HBox([start_date_perf_risk,end_date_perf_risk,var_trajectory_button,var_risk_trajectory_refresh_button]),
                                   selected_fund_to_decompose_var,func_name,window_var,var_centile,value_at_risk_trajectory_output])
    
    # ---------- Market Risk Metrics ----------
    global quantities_eigen,perf_index_eigen

    pca_output=widgets.Output()
    pca_components=widgets.Output()
    start_date_market_risk = widgets.DatePicker(value=start_perf_date, layout=widgets.Layout(width='350px'))
    end_date_market_risk = widgets.DatePicker(value=datetime.date.today(), layout=widgets.Layout(width='350px'))


    quantities_eigen=pd.DataFrame()
    perf_index_eigen=pd.DataFrame()

    
    def get_market_risk_metrics(_):

        try:
            start_ts = pd.to_datetime(start_date_market_risk.value)
            end_ts = pd.to_datetime(end_date_market_risk.value)
        except Exception:
            with pca_output:
                pca_output.clear_output()   
            with pca_components:
                pca_components.clear_output()   
                print("⚠️ Invalid start or end date.")
            return
        
        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
            with pca_output:
                pca_output.clear_output()   
            with pca_components:
                pca_components.clear_output()   
                print("⚠️ Error with date range.")
            return

        if returns_to_use.empty:

            with pca_output:
                pca_output.clear_output()            
            with pca_components:
                pca_components.clear_output()
                print('⚠️ Load Prices.')
            return
            
        market_tickers=[t for t in tickers if t in dataframe.columns]

        range_returns=returns_to_use.loc[start_ts:end_ts,market_tickers]
        portfolio=RiskAnalysis(range_returns)

        eigval,eigvec,portfolio_components=portfolio.pca(num_components=num_components.value)
        selected_components.options=portfolio_components.columns
        num_components.max=len(range_returns.columns)+1
        num_closest_to_pca.max=len(range_returns.columns)
        
        variance_explained=eigval/eigval.sum()
        variance_explained_dataframe=pd.DataFrame(variance_explained,index=portfolio_components.columns,columns=['Variance Explained'])
        
        pca_weight=dict((portfolio_components[selected_components.value]/(portfolio_components[selected_components.value]).sum()))
        pca_portfolio=pd.DataFrame(portfolio_components[selected_components.value]).sort_values(by=selected_components.value,ascending=False)
        
        historical_PCA=pd.DataFrame(np.array(list(pca_weight.values())).dot(np.transpose(portfolio.returns)),index=portfolio.returns.index,columns=['PCA'])
        historical_PCA=historical_PCA.dropna()
        historical_PCA.iloc[0]=0
        
        comparison=portfolio.returns.copy()
        comparison['PCA']=historical_PCA
        distances=np.sqrt(np.sum(comparison.apply(lambda y:(y-historical_PCA['PCA'])**2),axis=0)).sort_values()
        
        pca_similarity=comparison[distances.index[:num_closest_to_pca.value]]
        pca_similarity.iloc[0]=0
        pca_similarity=(1+pca_similarity).cumprod()*100
    
        with pca_components:
            
            pca_components.clear_output(wait=True)
            
            fig=px.bar(variance_explained_dataframe,title='Variance Explanation in %')
            fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400) 
            fig.update_traces(textfont=dict(family="Arial Narrow", size=15))

            fig2=px.bar(pca_portfolio,title='Eigen Weights')
            fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400) 
            fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))

            fig.show()
            fig2.show()
            
        with pca_output:
            pca_output.clear_output(wait=True)
            
            fig3=px.line((1+historical_PCA).cumprod()*100,title='Eigen Index', render_mode = 'svg')
            fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
            fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))

            fig4=px.line(pca_similarity,title='PCA Similarity', render_mode = 'svg')
            fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
            fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))

            fig3.show()
            fig4.show()

    asset_output_corr = widgets.Output()
    button_corr = widgets.Button(description="Show Correlation", button_style="success")
    
    window_corr=widgets.IntText(
    value=252,
    description='Rolling Correlation:',
    disabled=False,style={'description_width': '150px'}
    )

    
    def update_correlation(change=None):

        try:
            start_ts = pd.to_datetime(start_date_market_risk.value)
            end_ts = pd.to_datetime(end_date_market_risk.value)
            
        except Exception:
            with asset_output_corr:
                asset_output_corr.clear_output()
                print("⚠️ Invalid start or end date.")
            return
            
        if returns_to_use.empty:
            with asset_output_corr:
                asset_output_corr.clear_output()
                print('⚠️Load Prices.')
                return
            
        if dropdown_asset1.value==dropdown_asset2.value:
            with asset_output_corr:

                asset_output_corr.clear_output()
                print('⚠️Same asset')
                return    
            
        with asset_output_corr:
            asset_output_corr.clear_output()
            if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
                print("⚠️ Error with date range.")
                return
   
        range_returns=returns_to_use.loc[start_ts:end_ts]
        pca_over_time=first_pca_over_time(returns=range_returns,window=window_corr.value)

        rolling_corr_output=widgets.Output()
        correlation_matrix=widgets.Output()
        pca_overtime_output=widgets.Output()
        mean_returns_output=widgets.Output()
        
        with asset_output_corr:
            asset_output_corr.clear_output(wait=True)
            
            rolling_correlation = range_returns[dropdown_asset1.value].rolling(window_corr.value).corr(
                range_returns[dropdown_asset2.value]
            ).dropna()

            rolling_mean_returns=range_returns.rolling(window_corr.value).mean().dropna()*252
            
            with rolling_corr_output:
                fig = px.line(rolling_correlation, title=f"{dropdown_asset1.value}/{dropdown_asset2.value} Correlation", render_mode = 'svg')
                fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
                fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
    
                fig.show()

            with correlation_matrix:
                fig2 = px.imshow(range_returns.corr().round(2), title='Correlation Matrix',color_continuous_scale='Blues', text_auto=True, aspect="auto")
                fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig2.update_traces(xgap=2, ygap=2)
                fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                fig2.show()
            with pca_overtime_output:
                fig3=px.line(pca_over_time,title='First principal component (Variance Explained in %)', render_mode = 'svg')
                fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig3.update_layout(xaxis_title=None, yaxis_title=None)
                fig3.show()

            with mean_returns_output:
                fig4=px.line(rolling_mean_returns,title='Mean Return', render_mode = 'svg')
                fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig4.update_layout(xaxis_title=None, yaxis_title=None)
                fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in [dropdown_asset1.value,dropdown_asset2.value])
                fig4.show()
                
            ui=widgets.HBox([widgets.VBox([rolling_corr_output,mean_returns_output]),widgets.VBox([pca_overtime_output,correlation_matrix])])
            display(ui)
            
    selected_components=widgets.Dropdown(options=['PC1'],description='Select PCA:',style={'description_width': '150px'})
    num_components=widgets.BoundedIntText(min=1,max=5,value=5,description='PCA Components:',style={'description_width': '150px'})
    num_closest_to_pca=widgets.BoundedIntText(min=1,max=20,value=5,description='PCA Closest:',style={'description_width': '150px'})
    market_button=widgets.Button(description='Market Risk Analysis',button_style='info',style={'description_width': '150px'})
    market_button.on_click(get_market_risk_metrics)    
    correlation_button=widgets.Button(description='Get Correlation',button_style='info',style={'description_width': '150px'})
    correlation_button.on_click(update_correlation)

    market_ui=widgets.VBox([widgets.HBox([start_date_market_risk,
                                          end_date_market_risk,market_button]),
                            num_components,selected_components,num_closest_to_pca,
                            widgets.HBox([pca_components,pca_output])])
    
    correlation_ui=widgets.VBox([widgets.HBox([start_date_market_risk,end_date_market_risk,correlation_button]),dropdown_asset1,dropdown_asset2,window_corr,asset_output_corr])    

    
    frequency_eigen=widgets.Dropdown(description='Frequency:', options=['Yearly','Quarterly','Monthly'], value='Quarterly',style={'description_width': '150px'})
    selected_pca_market=widgets.Dropdown(description='PCA:', options=['PC1','PC2','PC3'], value='PC1',style={'description_width': '150px'})
    market_vol_window=widgets.IntText(
        value=252,
        description='Window:',
        disabled=False,style={'description_width': '150px'})
    
    market_factor_output=widgets.Output()
    market_factors_button=widgets.Button(description="Get Market Drivers", button_style="info",style={'description_width': '150px'})


    def show_market_graph(_):
        try:
            start_ts = pd.to_datetime(start_date_market_risk.value)
            end_ts = pd.to_datetime(end_date_market_risk.value)
            
        except Exception:
            with market_factor_output:
                market_factor_output.clear_output()
                print("⚠️ Invalid start or end date.")
            return
            
        if returns_to_use.empty:
            with market_factor_output:
                asset_output_corr.clear_output()
                print('⚠️Load Prices.')
                return
        if quantities_eigen.empty:
            with market_factor_output:
                market_factor_output.clear_output()
                print('⚠️Load Factors.')
                return
        with market_factor_output:
            market_factor_output.clear_output()
            if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
                print("⚠️ Error with date range.")
                return 
                
        range_prices=dataframe.loc[start_ts:end_ts]
        range_returns=returns_to_use.loc[start_ts:end_ts]
        
        market_portfolio=(quantities_eigen.loc[range_prices.index]*range_prices)
        
        market_index=market_portfolio.sum(axis=1).to_frame()
        market_index.columns=['Market Index']
        market_cost=rebalanced_book_cost(range_prices,quantities_eigen.loc[range_prices.index])
        
        weights_series=market_portfolio.copy()
        weights_series=weights_series.apply(lambda x: x/market_portfolio.sum(axis=1))
        
        updated_perf_index_eigen=perf_index_eigen.copy()
        updated_perf_index_eigen=updated_perf_index_eigen.loc[start_ts:end_ts]
        updated_perf_index_eigen.iloc[0]=0
    
        market_results=(1+updated_perf_index_eigen).cumprod()*100
    
        market_pnl=market_portfolio-rebalanced_book_cost(range_prices,quantities_eigen.loc[range_prices.index])
        market_pnl['Market Index']=market_pnl.sum(axis=1)
    
        vol_contribution=get_ex_ante_vol_contribution(weights_series,range_returns,window=market_vol_window.value)
        correlation_contribution=get_correlation_contribution(weights_series,range_returns,window=market_vol_window.value)
        idiosyncratic_contribution=get_idiosyncratic_contribution(weights_series,range_returns,window=market_vol_window.value)
            
        output1=widgets.Output()
        output2=widgets.Output()
        
        with market_factor_output:
            
            market_factor_output.clear_output(wait=True)
            
            with output1:
            
                fig = px.line(market_results, title='Performance Comparison', width=800, height=400, render_mode = 'svg')
                fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Market Index","Fund","Bitcoin","Historical Portfolio"])
                fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
                fig.show()
                
                fig2 = px.line(market_pnl, title='Market Drivers', width=800, height=400, render_mode = 'svg')
                fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Market Index"])
                fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                fig2.show()
                
                fig3 = px.line(correlation_contribution, title='Market Correlation', width=800, height=400, render_mode = 'svg')
                fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Correlation"])
                fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
                fig3.show()
                
                
            with output2:
                
        
                fig4 = px.line(vol_contribution, title='Market Volatility', width=800, height=400, render_mode = 'svg')
                fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Vol"])
                fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
                fig4.show()    
                
                fig5 = px.line(idiosyncratic_contribution, title='Market Intrinsic Volatility', width=800, height=400, render_mode = 'svg')
                fig5.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig5.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Idiosyncratic Vol"])
                fig5.update_traces(textfont=dict(family="Arial Narrow", size=15))
                fig5.show()
                
                fig6 = px.line(weights_series, title='Market Weights', width=800, height=400, render_mode = 'svg')
                fig6.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig6.update_traces(visible="legendonly", selector=lambda t: not t.name in ["BTCUSDT"])
                fig6.update_traces(textfont=dict(family="Arial Narrow", size=15))
                fig6.show()    
                
            ui=widgets.HBox([output1,output2])
        
            display(ui)
        
    def get_market_factors(_):
        global quantities_eigen,perf_index_eigen
        try:
            start_ts = pd.to_datetime(start_date_market_risk.value)
            end_ts = pd.to_datetime(end_date_market_risk.value)
            
        except Exception:
            with market_factor_output:
                market_factor_output.clear_output()
                print("⚠️ Invalid start or end date.")
            return
            
        if returns_to_use.empty:
            with market_factor_output:
                market_factor_output.clear_output()
                print('⚠️Load Prices.')
                return
            
        with market_factor_output:
            market_factor_output.clear_output()
            if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
                print("⚠️ Error with date range.")
                return 
                
        results={}
        
        range_prices=dataframe.loc[start_ts:end_ts]
        range_returns=returns_to_use.loc[start_ts:end_ts]
        
        dates=get_rebalancing_dates(range_returns,frequency=frequency_eigen.value)
        tasks = [(returns_to_use.loc[dates[i]:dates[i+1]],dates[i], dates[i+1]) for i in range(len(dates)-1)]
        # Run with threads
        results = {}
        loading_bar_market=widgets.IntProgress(description='Loading Drivers...',min=0, max=len(tasks),style={'description_width': '150px'})
        
        loading_bar_market.value=0
        
        with market_factor_output:
            display(loading_bar_market)
            
        def worker(subset,start, end):

            if subset.empty or len(subset) < 2:
                return None
            try:
                risk = RiskAnalysis(subset)
                eigval,eigvec,portfolio_components=risk.pca(num_components=5)
                weights=np.real(portfolio_components[selected_pca_market.value].to_numpy())
                
                return subset.index[-1], np.round(weights, 6)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {executor.submit(worker,subset, start, end): (subset,start, end) for subset,start, end in tasks}
            for future in as_completed(futures):
                out = future.result()
                if out is not None:
                    date_key, weights = out
                    results[date_key] = weights
                    loading_bar_market.value+=1
                    
        if not results:
            print("⚠️ No valid Eigen values computed.")
            return
        
        weights=pd.DataFrame(results).T
        
        quantities_eigen=rebalanced_dynamic_quantities(range_prices,weights)
        
        market_portfolio=(quantities_eigen*range_prices)
        
        market_index=market_portfolio.sum(axis=1).to_frame()
        market_index.columns=['Market Index']
        perf_index_eigen=market_index.pct_change(fill_method=None)

        if not performance_ex_post.empty:
            perf_index_eigen=pd.concat([perf_index_eigen,performance_ex_post],axis=1)
    
        elif not global_returns.empty:
            perf_index_eigen=pd.concat([perf_index_eigen,global_returns],axis=1)
    
    
        show_market_graph(None)
    
    refresh_market_dates_button=widgets.Button(description='Refresh')
    market_factors_button.on_click(get_market_factors)
    refresh_market_dates_button.on_click(show_market_graph)
    
    market_factors_ui=widgets.VBox([widgets.HBox([start_date_market_risk,end_date_market_risk,market_factors_button,refresh_market_dates_button]),
                                    widgets.VBox([frequency_eigen,selected_pca_market,market_vol_window,market_factor_output])])
    
    global daily_pnl,pnl_history,historical_ptf,performance_ex_post,positions,quantities_holding
    
    daily_pnl=pd.DataFrame()
    pnl_history=pd.DataFrame()
    historical_ptf=pd.DataFrame()
    performance_ex_post=pd.DataFrame()
    positions=pd.DataFrame()
    quantities_holding=pd.DataFrame()
        
    def check_connection(_):
        global quantities_holding,positions
        url_positions='https://github.com/niroojane/Risk-Management/raw/refs/heads/main/Positions.xlsx'
        url_quantities='https://github.com/niroojane/Risk-Management/raw/refs/heads/main/Quantities.xlsx'
        
        with ex_post_perf:
            
            ex_post_perf.clear_output(wait=True)
            
            position = read_excel_from_url(url_positions,index_col=0)
            if position is None:
                raise FileNotFoundError("Positions.xlsx could not be loaded. Execution stopped.")
                print('Positions Not Found in Repository')
                
            quantities_history = read_excel_from_url(url_quantities,index_col=0)
            if quantities_history is None:
                raise FileNotFoundError("Quantities.xlsx could not be loaded. Execution stopped.")
                print('Quantities Not Found in Repository')
            
            # position=pd.read_excel('Positions.xlsx',index_col=0)
            positions,quantities_holding=Binance.get_positions_history(enddate=datetime.datetime.today())
            positions=positions.sort_index()
            positions.index=pd.to_datetime(positions.index)
            positions=pd.concat([position,positions])
            positions.index=pd.to_datetime(positions.index)
            positions=pd.concat([position,positions]).sort_index()
            positions=positions.loc[~positions.index.duplicated(keep='last'),:]
            positions['Total']=positions.loc[:,positions.columns!='Total'].sum(axis=1)
            
            # quantities_history=pd.read_excel('Quantities.xlsx',index_col=0)
            
            quantities_holding.index=pd.to_datetime(quantities_holding.index)
            quantities_holding=pd.concat([quantities_holding,quantities_history])
            quantities_holding=quantities_holding.loc[~quantities_holding.index.duplicated(),:]
        
            quantities_holding=quantities_holding.sort_index()
            start_date_perf_ex_post.value=positions.index[0].date()
            
    start_date_perf_ex_post = widgets.DatePicker(value=datetime.date.today(), layout=widgets.Layout(width='350px'))
    end_date_perf_ex_post = widgets.DatePicker(value=datetime.date.today(), layout=widgets.Layout(width='350px'))
    ex_post_perf=widgets.Output()
    ex_post_calendar=widgets.Output()
    
    fund_ex_post=widgets.Dropdown(value='Historical Portfolio',options=['Historical Portfolio','Fund'],description='Select Fund:',style={'description_width': '150px'})
    benchmark_ex_post=widgets.Dropdown(value='Historical Portfolio',options=['Historical Portfolio','Fund'],description='Select Benchmark:',style={'description_width': '150px'})
    frequency_graph_ex_post=widgets.Dropdown(options=['Year','Month'],value='Year',description='Select Frequency:',style={'description_width': '150px'})
    calendar_button_ex_post=widgets.Button(description='Update', button_style='info')
    
    def show_graph_ex_post(_):
        
        try:
            start_ts = pd.to_datetime(start_date_perf_ex_post.value)
            end_ts = pd.to_datetime(end_date_perf_ex_post.value)
        except Exception:
            with ex_post_calendar:
                ex_post_calendar.clear_output()
                print("⚠️ Invalid start or end date.")
            return
    
        with ex_post_calendar:
            ex_post_calendar.clear_output()
    
            if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
                print("⚠️ Error with date range.")
                return
    
            if pnl_history.empty:
                print("⚠️ P&L not computed.")
                return  
        cumulative_performance_ex_post=pd.DataFrame()
        
        if global_returns.empty:
            
            performance_ex_post=historical_ptf['Historical Portfolio'].copy()
            performance_ex_post=performance_ex_post.to_frame()
        else:
            
            performance_ex_post=historical_ptf['Historical Portfolio'].copy()
            performance_ex_post=pd.concat([performance_ex_post,global_returns],axis=1).sort_index()
        
            options = list(performance_ex_post.columns)
            fund_ex_post.options = options
            benchmark_ex_post.options = options
            
        with ex_post_calendar:
            
            return_and_vol_graph=widgets.Output()
            sharpe_and_te_graph=widgets.Output()
            
            ex_post_calendar.clear_output()
            
            if performance_ex_post.empty:
                print("⚠️ Load Ex Post Performance.")
                return
            if fund_ex_post.value==benchmark_ex_post.value:
                print("⚠️ Benchmark and Fund must be different.")
                return
            if performance_ex_post.empty or performance_ex_post.shape[1]<2:
                print("⚠️ No performance data available yet. Please run an optimization first.")
                return

            cumulative_performance_ex_post=performance_ex_post.loc[start_date_perf_ex_post.value:end_date_perf_ex_post.value].copy()
            cumulative_performance_ex_post.iloc[0]=0
            cumulative_performance_ex_post=(1+cumulative_performance_ex_post).cumprod()*100   

            graphs=get_calendar_graph(cumulative_performance_ex_post, 
                               freq=frequency_graph_ex_post.value, 
                               benchmark=benchmark_ex_post.value, 
                               fund=fund_ex_post.value)
            
            # for name, fig in graphs.items():
            #     fig.show()
            keys=list(graphs.keys())
            with return_and_vol_graph:
                graphs[keys[0]].show()
                graphs[keys[2]].show()
            with sharpe_and_te_graph:
                graphs[keys[1]].show()
                graphs[keys[3]].show()   

            ui=widgets.HBox([return_and_vol_graph,sharpe_and_te_graph])
            
            display(ui)
            
            update_ex_post_chart(None)

    calendar_button_ex_post.on_click(show_graph_ex_post)


    def update_ex_post_chart(_):
        
        global performance_ex_post
        
        try:
            start_ts = pd.to_datetime(start_date_perf_ex_post.value)
            end_ts = pd.to_datetime(end_date_perf_ex_post.value)
        except Exception:
            with ex_post_perf:
                ex_post_perf.clear_output()
                print("⚠️ Invalid start or end date.")

            return
    
        with ex_post_perf:
            ex_post_perf.clear_output()
    
            if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
                print("⚠️ Error with date range.")
                return
    
            if pnl_history.empty:
                print("⚠️ P&L not computed.")
                return

        selected_cumulative_pnl=daily_pnl.loc[start_ts:end_ts,'Total'].copy()
        selected_cumulative_pnl.iloc[0]=0
        
        selected_history=pd.concat([selected_cumulative_pnl.cumsum(),pnl_history['Total'].loc[start_ts:end_ts]],axis=1)
        selected_history.columns=['Cumulative P&L','Total P&L']
        
        selected_daily_pnl=daily_pnl.loc[start_ts:end_ts].copy()
        selected_positions=positions.loc[start_ts:end_ts]
        
        if global_returns.empty:
            performance_ex_post=historical_ptf['Historical Portfolio'].copy()
            performance_ex_post=performance_ex_post.to_frame()
        else:
            performance_ex_post=historical_ptf['Historical Portfolio'].copy()
            performance_ex_post=pd.concat([performance_ex_post,global_returns],axis=1).sort_index()  
        
        cumulative_performance_ex_post=performance_ex_post.loc[start_ts:end_ts].copy()
        cumulative_performance_ex_post.iloc[0]=0
        cumulative_performance_ex_post=(1+cumulative_performance_ex_post).cumprod()*100
        pnl_contribution=(pnl_history-pnl_history.shift(1)).loc[start_ts:end_ts]
        
        git_output=widgets.Output()
        
        def git_push(_):
            
            with git_output:
                git_output.clear_output(wait=True)
                
                quantities_holding.to_excel('Quantities.xlsx',index=True)
                positions.to_excel('Positions.xlsx',index=True)
                if not trades.empty:
                    trades.to_excel('Trade History Reconstructed.xlsx',index=True)
                    git.push_or_update_file(trades,'Trade History Reconstructed')

                git.push_or_update_file(positions,'Positions')
                git.push_or_update_file(quantities_holding,'Quantities')
                    

        push_button=widgets.Button(description='Upload Files',button_style='success')
        push_button.on_click(git_push)

        expost_output=widgets.Output()
        expost_output1=widgets.Output()

        with ex_post_perf:
            ex_post_perf.clear_output(wait=True)

            with expost_output:
            
                fig=px.line(selected_positions,title='Portfolio Value', render_mode = 'svg')
                fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig.update_traces(visible="legendonly", selector=lambda t:  not t.name in ['Total'])
                fig.update_layout(xaxis_title=None, yaxis_title=None)
                fig.show()
                
                fig2=px.line(selected_history,title='Cumulative P&L', render_mode = 'svg')
                fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Cumulative P&L'])
    
                fig2.update_layout(xaxis_title=None, yaxis_title=None)
                fig2.show()
                

                    
                fig3 = px.line(pnl_contribution.cumsum(),x=pnl_contribution.index,y=pnl_contribution.columns,
                               title="Cumulative P&L Contribution")
                fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig3.update_traces(visible="legendonly", selector=lambda t:  not t.name in ['Total'])

                fig3.update_layout(xaxis_title=None, yaxis_title=None,showlegend=True)
                fig3.show()
                
            with expost_output1:

                fig4=px.line(cumulative_performance_ex_post,title='Cumulative Return', render_mode = 'svg')
                fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Historical Portfolio','Fund','Bitcoin'])
                fig4.update_layout(xaxis_title=None, yaxis_title=None)
                fig4.show()
                
                drawdown = (cumulative_performance_ex_post - cumulative_performance_ex_post.cummax()) / cumulative_performance_ex_post.cummax()
                
                fig5=px.line(drawdown,title='Drawdown', render_mode = 'svg')
                fig5.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig5.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Historical Portfolio','Fund','Bitcoin'])
                fig5.update_layout(xaxis_title=None, yaxis_title=None)
                fig5.show()

                fig6 = px.bar(pnl_contribution,x=pnl_contribution.index,y=pnl_contribution.columns,
                     title="Daily P&L",barmode="relative")
                fig6.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig6.update_traces(visible="legendonly", selector=lambda t:  t.name in ['Total'])
                fig6.update_layout(xaxis_title=None, yaxis_title=None,showlegend=True)
                fig6.show()
                
            ui=widgets.HBox([expost_output,expost_output1])
            
            display(ui)
            display(display_scrollable_df(pnl_contribution))
            display(push_button)
            display(git_output)
    
    def get_ex_post_returns(_):
        
        global daily_pnl,pnl_history,historical_ptf,performance_ex_post
 
        loading_bar.value=0

        with ex_post_perf:
            ex_post_perf.clear_output()
            
            display(loading_bar_pnl)
            display(loading_bar)
        if book_cost.empty:

            get_pnl_on_click(None)  
          
        quantities_tickers=list(quantities_holding.columns)
        daily_book_cost=book_cost.resample("D").last().dropna().sort_index()
        book_cost_history=pd.DataFrame()
        book_cost_history.index=set(daily_book_cost.index.append(quantities_holding.index))
        
        book_cost_history=book_cost_history.sort_index()
        cols= quantities_holding.columns[quantities_holding.columns!='USDCUSDT']
        
        for col in cols:
            book_cost_history[col]=daily_book_cost[col]

        book_cost_history=book_cost_history.ffill()
        book_cost_history=book_cost_history.loc[quantities_holding.index] 
        
        today = datetime.date.today()
        start_pnl=quantities_holding.index[0]
        days_total = (today - start_pnl.date()).days
        
        weights_ex_post=positions.copy()
        weights_ex_post=weights_ex_post.drop(columns=['USDTUSDT'])
        weights_ex_post=weights_ex_post.apply(lambda x: x/weights_ex_post['Total'])
        
        start_date=weights_ex_post.index[0].date()

        binance_data=get_price_threading(quantities_tickers,start_date)
        binance_data=binance_data.sort_index()
        binance_data=binance_data.loc[~binance_data.index.duplicated(keep='last')]
        
        pnl_history=pd.DataFrame()
        pnl_history.index=quantities_holding.index
        pnl_history=pnl_history.sort_index()

        for col in cols:
            pnl_history[col]=quantities_holding[col]*(binance_data[col]-book_cost_history[col])

        pnl_history['Total']=pnl_history.sum(axis=1)
    
        daily_pnl=pnl_history['Total']-pnl_history['Total'].shift(1)
        daily_pnl=pd.DataFrame(daily_pnl)
        colors = ['green' if value >= 0 else 'red' for value in daily_pnl.values]
        
        daily_pnl['color'] = daily_pnl['Total'].apply(lambda v: 'green' if v >= 0 else 'red')
    
        binance_data_return=np.log(1+binance_data.pct_change(fill_method=None))
        weight_date=set(weights_ex_post.index)
        binance_date=set(binance_data_return.index)
        common_date=weight_date.intersection(binance_date)
        
        binance_data2=binance_data_return.loc[list(common_date)].copy().sort_index()
        weights_ex_post2=weights_ex_post.loc[list(common_date)].copy().sort_index()
        historical_ptf=pd.DataFrame()
        
        for col in binance_data:
            historical_ptf[col]=weights_ex_post2[col]*binance_data2[col]
            
        historical_ptf['Historical Portfolio']=historical_ptf.sum(axis=1)   
    
        if global_returns.empty:
            performance_ex_post=historical_ptf['Historical Portfolio'].copy()
            performance_ex_post=performance_ex_post.to_frame()
        else:
            performance_ex_post=historical_ptf['Historical Portfolio'].copy()
            performance_ex_post=pd.concat([performance_ex_post,global_returns],axis=1).sort_index()
        
            options = list(performance_ex_post.columns)
            fund_ex_post.options = options
            benchmark_ex_post.options = options
            fund_ex_post.value = 'Historical Portfolio'
            benchmark_ex_post.value ='Fund'

        update_ex_post_chart(None)
        show_graph_ex_post(None)
    
    
    ex_post_button=widgets.Button(description='Get P&L',button_style='info')
    ex_post_button.on_click(get_ex_post_returns)
    refresh_ex_post_button=widgets.Button(description='Refresh')
    refresh_ex_post_button.on_click(update_ex_post_chart)
    
    ex_post_ui=widgets.VBox([widgets.HBox([start_date_perf_ex_post,end_date_perf_ex_post,ex_post_button,refresh_ex_post_button]),ex_post_perf])    
    calendar_ui_ex_post=widgets.VBox([widgets.HBox([frequency_graph_ex_post,fund_ex_post,benchmark_ex_post,calendar_button_ex_post]),ex_post_calendar])


    global results_vol,series_dict,current_underlying_returns,results_tracking_error,spread_weights

    results_vol=pd.DataFrame()
    results_tracking_error=pd.DataFrame()
    current_underlying_returns=pd.DataFrame()
    series_dict={}
    spread_weights={}
    
    risk_trajectory_output=widgets.Output()
    tracking_error_trajectory_output=widgets.Output()

    selected_fund_to_decompose=widgets.Dropdown(options=['Fund','Historical Portfolio'],value='Historical Portfolio',description='Fund:')
    selected_bench_risk=widgets.Dropdown(options=['Fund','Historical Portfolio'],value='Fund',description='Bench:')
    
    window_risk=widgets.IntText(
    value=252,
    description='Window:',
    disabled=False)   
    
    window_te=widgets.IntText(
    value=252,
    description='Window:',
    disabled=False)   
        
    loading_bar_risk = widgets.IntProgress(description='Loading Vol...',min=0, max=100,style={'description_width': '150px'})
    loading_bar_tracking_error = widgets.IntProgress(description='Loading TE...',min=0, max=100,style={'description_width': '150px'})
    
    risk_trajectory_button=widgets.Button(description="Get Ex Ante Vol", button_style="success")
    tracking_error_trajectory_button=widgets.Button(description="Get Ex Ante TE", button_style="success")

    risk_trajectory_refresh_button=widgets.Button(description="Refresh")
    tracking_error_refresh_button=widgets.Button(description="Refresh")
    

    def show_risk_graph(_):
        
        global results_vol
        
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with risk_trajectory_output:
                risk_trajectory_output.clear_output(wait=True)
                print("⚠️ Invalid start or end date.")
            return
        output1=widgets.Output()
        output2=widgets.Output()
        
        with risk_trajectory_output:
            
            risk_trajectory_output.clear_output(wait=True)
            
            if dataframe.empty:
                print('⚠️Load Prices.')
                return
            if dataframe.empty or returns_to_use.empty or grid.data.empty:
                print("⚠️ Please compute optimization results first.")
                return

            if results_vol.empty:
                print("⚠️ Load Ex Ante Vol.")
                return
            
            if len(returns_to_use.loc[start_ts:end_ts]) < window_risk.value:
                print("⚠️ Date range is shorter than rolling window.")
                return    

            if not series_dict:
                print("⚠️ Weights Empty.")
                return
            else:
                
                risk_trajectory_output.clear_output(wait=True)
                
                series_weights=series_dict[selected_fund_to_decompose.value]
                if selected_fund_to_decompose.value!='Historical Portfolio':
                    contribution_to_vol=get_ex_ante_vol_contribution(series_weights.loc[start_ts:end_ts],returns_to_use.loc[start_ts:end_ts],window_risk.value)
                    correlation_contrib=get_correlation_contribution(series_weights.loc[start_ts:end_ts],returns_to_use.loc[start_ts:end_ts],window_risk.value)
                    idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights.loc[start_ts:end_ts],returns_to_use.loc[start_ts:end_ts],window_risk.value)

                else:
                    
                    contribution_to_vol=get_ex_ante_vol_contribution(series_weights.loc[start_ts:end_ts],current_underlying_returns.loc[series_weights.index].loc[start_ts:end_ts],window_risk.value)
                    correlation_contrib=get_correlation_contribution(series_weights.loc[start_ts:end_ts],current_underlying_returns.loc[series_weights.index].loc[start_ts:end_ts],window_risk.value)
                    idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights.loc[start_ts:end_ts],current_underlying_returns.loc[series_weights.index].loc[start_ts:end_ts],window_risk.value)
                    
                with output1:
                    
                    fig = px.line(results_vol.loc[start_ts:end_ts], title='Ex Ante Volatility', width=800, height=400, render_mode = 'svg')
                    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Historical Portfolio","Fund"])
                    fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig.show()

                    fig4 = px.line(idiosyncratic_contrib, title='Idiosyncratic Contribution', width=800, height=400, render_mode = 'svg')
                    fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Idiosyncratic Vol"])
                    fig4.show()   
                        
                with output2:
                    
                    fig2 = px.line(contribution_to_vol, title='Volatility Contribution', width=800, height=400, render_mode = 'svg')
                    fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Total Vol'])
                    fig2.show()
                    
                    fig3 = px.line(correlation_contrib, title='Correlation Contribution', width=800, height=400, render_mode = 'svg')
                    fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Correlation"])
                    fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig3.show()

                ui=widgets.HBox([output1,output2])

                display(ui)
                
                    
    def get_risk_trajectory(_):
        
        global results_vol,series_dict,current_underlying_returns

        results_vol=pd.DataFrame()
        
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with risk_trajectory_output:
                risk_trajectory_output.clear_output(wait=True)
                print("⚠️ Invalid start or end date.")
            return
            
        with risk_trajectory_output:
            risk_trajectory_output.clear_output(wait=True)

            if dataframe.empty:
                print('⚠️Load Prices.')
                return
            if dataframe.empty or returns_to_use.empty or grid.data.empty:
                print("⚠️ Please compute optimization results first.")
                return
            
            if len(returns_to_use.loc[start_ts:end_ts]) < window_risk.value:
                print("⚠️ Date range is shorter than rolling window.")
                return    
            else:
                display(loading_bar)
                display(loading_bar_risk)

        series_dict={}

        for key in grid.data.index:
            
            rebalanced_series=rebalanced_portfolio(dataframe,grid.data.loc[key])
            rebalanced_series_weights=rebalanced_series.apply(lambda x: x/rebalanced_series.sum(axis=1))
            buy_and_hold_series=buy_and_hold(dataframe,grid.data.loc[key])
            buy_and_hold_series_weights=buy_and_hold_series.apply(lambda x: x/buy_and_hold_series.sum(axis=1))
            series_dict['Rebalanced '+key]=rebalanced_series_weights.loc[start_ts:end_ts]
            series_dict['Buy and Hold '+key]=buy_and_hold_series_weights.loc[start_ts:end_ts]
        
        weights_ex_post=positions.copy()
        weights_ex_post=weights_ex_post.drop(columns=['USDTUSDT'])
        weights_ex_post=weights_ex_post.apply(lambda x: x/weights_ex_post['Total'])
        weights_ex_post=weights_ex_post.drop(columns=['Total'])
        weights_ex_post=weights_ex_post.fillna(0.0)
        
        if not quantities.empty:
            portfolio=quantities*dataframe
            model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
            series_dict['Fund']=model_weights.loc[start_ts:end_ts]
            
        if not quantities_core.empty:
            portfolio=quantities_core*dataframe
            model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
            series_dict['Core']=model_weights.loc[start_ts:end_ts]

        if not quantities_overlay.empty:
            portfolio=quantities_overlay*dataframe
            model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
            series_dict['Overlay']=model_weights.loc[start_ts:end_ts]

        tickers_combined=list(quantities.columns)+list(weights_ex_post.columns)
        tickers_combined=list(set(tickers_combined))
        
        current_underlying_prices=get_price_threading(tickers_combined,weights_ex_post.index[0].date())    
        current_underlying_returns=current_underlying_prices.pct_change(fill_method=None)

        tasks=[(key,series_dict[key],returns_to_use.loc[start_ts:end_ts],window_risk.value) for key in series_dict]

        series_dict['Historical Portfolio']=weights_ex_post.loc[start_ts:end_ts]

        tasks.append(('Historical Portfolio',weights_ex_post.loc[start_ts:end_ts],current_underlying_returns.loc[weights_ex_post.index].loc[start_ts:end_ts],window_risk.value))

        loading_bar_risk.value = 0
        loading_bar_risk.max=len(tasks)   
        
        results_dict = {}
        
        for name,weight,returns,window in tasks:
            results_dict[name]=get_ex_ante_vol(weight, returns, window)
            loading_bar_risk.value += 1
                
        loading_bar_risk.value = loading_bar_risk.max
        
        with risk_trajectory_output:

            if not results_dict:
                print("⚠️ No risk results were computed.")
                return
            
            results_vol = pd.concat(results_dict.values(), axis=1)
            
            if results_vol.shape[1] == 0:
                print("⚠️ Risk results are empty.")
                return  

        results_vol.columns=results_dict.keys()
        selected_fund_to_decompose.options=results_vol.columns
        selected_fund_to_decompose.value='Historical Portfolio'
        
        show_risk_graph(None)

    show_risk_graph(None)
    
    def show_tracking_error_graph(_):
        
        global results_tracking_error
        
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with tracking_error_trajectory_output:
                tracking_error_trajectory_output.clear_output(wait=True)
                print("⚠️ Invalid start or end date.")
            return
        output1=widgets.Output()
        output2=widgets.Output()
        
        with tracking_error_trajectory_output:
            
            tracking_error_trajectory_output.clear_output(wait=True)
            
            if dataframe.empty:
                print('⚠️Load Prices.')
                return
            if dataframe.empty or returns_to_use.empty or grid.data.empty:
                print("⚠️ Please compute optimization results first.")
                return

            if results_tracking_error.empty:
                print("⚠️ Load Ex Ante TE.")
                return

            if not spread_weights:
                print("⚠️ Weights Empty.")
                return

            if len(returns_to_use.loc[start_ts:end_ts]) < window_te.value:
                print("⚠️ Date range is shorter than rolling window.")
                return   
                
            else:
                
                tracking_error_trajectory_output.clear_output(wait=True)
                
                series_weights=spread_weights[selected_fund_to_decompose.value]
                
                if selected_fund_to_decompose.value!='Historical Portfolio':
                    contribution_to_vol=get_ex_ante_vol_contribution(series_weights.loc[start_ts:end_ts],returns_to_use.loc[series_weights.index].loc[start_ts:end_ts],window_te.value)
                    correlation_contrib=get_correlation_contribution(series_weights.loc[start_ts:end_ts],returns_to_use.loc[series_weights.index].loc[start_ts:end_ts],window_te.value)
                    idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights.loc[start_ts:end_ts],returns_to_use.loc[series_weights.index].loc[start_ts:end_ts],window_te.value)

                else:
                    
                    contribution_to_vol=get_ex_ante_vol_contribution(series_weights.loc[start_ts:end_ts],current_underlying_returns.loc[series_weights.index].loc[start_ts:end_ts],window_te.value)
                    correlation_contrib=get_correlation_contribution(series_weights.loc[start_ts:end_ts],current_underlying_returns.loc[series_weights.index].loc[start_ts:end_ts],window_te.value)
                    idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights.loc[start_ts:end_ts],current_underlying_returns.loc[series_weights.index].loc[start_ts:end_ts],window_te.value)
                    
                with output1:
                    
                    fig = px.line(results_tracking_error.loc[start_ts:end_ts], title='Ex Ante Tracking Error', width=800, height=400, render_mode ='svg')
                    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Historical Portfolio","Fund"])
                    fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig.show()

                    fig4 = px.line(idiosyncratic_contrib, title='Idiosyncratic Contribution', width=800, height=400, render_mode = 'svg')
                    fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Idiosyncratic Vol"])
    
                    fig4.show()   
                        
                with output2:
                    
                        
                    fig2 = px.line(contribution_to_vol, title='Tracking Error Contribution', width=800, height=400, render_mode = 'svg')
                    fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Vol"])
                    
                    fig2.show()
                    
                    fig3 = px.line(correlation_contrib, title='Correlation Contribution', width=800, height=400, render_mode = 'svg')
                    fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Correlation"])
                    fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig3.show()

                ui=widgets.HBox([output1,output2])

                display(ui)
                    
    def get_tracking_error_trajectory(_):
        
        global results_tracking_error,spread_weights,current_underlying_returns

        results_tracking_error=pd.DataFrame()
        
        try:
            start_ts = pd.to_datetime(start_date_perf_risk.value)
            end_ts = pd.to_datetime(end_date_perf_risk.value)
        except Exception:
            with tracking_error_trajectory_output:
                risk_trajectory_output.clear_output(wait=True)
                print("⚠️ Invalid start or end date.")
            return
            
        with tracking_error_trajectory_output:
            tracking_error_trajectory_output.clear_output(wait=True)
                
            if dataframe.empty:
                print('⚠️Load Prices.')
                return
            if dataframe.empty or returns_to_use.empty or grid.data.empty:
                print("⚠️ Please compute optimization results first.")
                return
                        
            if len(returns_to_use.loc[start_ts:end_ts]) < window_te.value:
                print("⚠️ Date range is shorter than rolling window.")
                return   
            else:
                display(loading_bar)
                display(loading_bar_risk)

        series_dict={}

        for key in grid.data.index:
            
            rebalanced_series=rebalanced_portfolio(dataframe,grid.data.loc[key])
            rebalanced_series_weights=rebalanced_series.apply(lambda x: x/rebalanced_series.sum(axis=1))
            buy_and_hold_series=buy_and_hold(dataframe,grid.data.loc[key])
            buy_and_hold_series_weights=buy_and_hold_series.apply(lambda x: x/buy_and_hold_series.sum(axis=1))
            series_dict['Rebalanced '+key]=rebalanced_series_weights.loc[start_ts:end_ts]
            series_dict['Buy and Hold '+key]=buy_and_hold_series_weights.loc[start_ts:end_ts]
        
        weights_ex_post=positions.copy()
        weights_ex_post=weights_ex_post.drop(columns=['USDTUSDT'])
        weights_ex_post=weights_ex_post.apply(lambda x: x/weights_ex_post['Total'])
        weights_ex_post=weights_ex_post.drop(columns=['Total'])
        weights_ex_post=weights_ex_post.fillna(0.0)
        
        if not quantities.empty:
            portfolio=quantities*dataframe
            model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
            series_dict['Fund']=model_weights.loc[start_ts:end_ts]
            
        if not quantities_core.empty:
            portfolio=quantities_core*dataframe
            model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
            series_dict['Core']=model_weights.loc[start_ts:end_ts]

        if not quantities_overlay.empty:
            portfolio=quantities_overlay*dataframe
            model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
            series_dict['Overlay']=model_weights.loc[start_ts:end_ts]
            
        tickers_combined=list(quantities.columns)+list(weights_ex_post.columns)
        tickers_combined=list(set(tickers_combined))
        
        current_underlying_prices=get_price_threading(tickers_combined,weights_ex_post.index[0].date())
        current_underlying_returns=current_underlying_prices.pct_change(fill_method=None)

        series_dict['Historical Portfolio']=weights_ex_post.loc[start_ts:end_ts]
        selected_weights=series_dict[selected_bench_risk.value]
        
        not_in_bench=list(set(weights_ex_post.columns)-set(selected_weights.columns))
        not_in_fund=list(set(selected_weights.columns)-set(weights_ex_post.columns))
        
        selected_weights = selected_weights.copy()
        weights_ex_post = weights_ex_post.copy()
        
        weights_ex_post[not_in_fund] = 0
        selected_weights[not_in_bench] = 0

        spread_weights={}

        for key in series_dict:
            spread_weights[key]=(series_dict[key]-selected_weights).fillna(0)
        
        tasks=[(key,spread_weights[key].loc[start_ts:end_ts],returns_to_use.loc[spread_weights[key].loc[start_ts:end_ts].index],window_te.value) for key in series_dict if key!='Historical Portfolio']
        
        spread_ex_post=(weights_ex_post-selected_weights).loc[weights_ex_post.index].loc[start_ts:end_ts].fillna(0)
        spread_weights['Historical Portfolio']=spread_ex_post
        tasks.append(('Historical Portfolio',spread_ex_post,current_underlying_returns.loc[spread_ex_post.index].loc[start_ts:end_ts],window_te.value))

        loading_bar_risk.value = 0
        loading_bar_risk.max=len(tasks)   
        
        results_dict = {}

        for name,weight,returns,window in tasks:
            
            results_dict[name]=get_ex_ante_vol(weight, returns, window)
            loading_bar_risk.value += 1
            
        loading_bar_risk.value = loading_bar_risk.max
        
        with tracking_error_trajectory_output:
            
            tracking_error_trajectory_output.clear_output(wait=True)

            if not results_dict:
                print("⚠️ No risk results were computed.")
                return
            
            results_tracking_error = pd.concat(results_dict.values(), axis=1)
            
            if results_tracking_error.shape[1] == 0:
                print("⚠️ Risk results are empty.")
                return       
                
        results_tracking_error.columns=results_dict.keys()       
        selected_bench_risk.options=results_tracking_error.columns
        selected_fund_to_decompose.options=results_tracking_error.columns
        selected_fund_to_decompose.value='Historical Portfolio'
        selected_bench_risk.value='Fund'
        
        show_tracking_error_graph(None)

    show_tracking_error_graph(None)    
    
    risk_trajectory_button.on_click(get_risk_trajectory)
    risk_trajectory_refresh_button.on_click(show_risk_graph)
    risk_exposure_ui=widgets.VBox([widgets.HBox([start_date_perf_risk,end_date_perf_risk,risk_trajectory_button,risk_trajectory_refresh_button]),
                                   selected_fund_to_decompose,window_risk,risk_trajectory_output])

    tracking_error_trajectory_button.on_click(get_tracking_error_trajectory)
    tracking_error_refresh_button.on_click(show_tracking_error_graph)
    tracking_error_exposure_ui=widgets.VBox([widgets.HBox([start_date_perf_risk,end_date_perf_risk,tracking_error_trajectory_button,tracking_error_refresh_button]),
                                   selected_fund_to_decompose,selected_bench_risk,window_te,tracking_error_trajectory_output])   
    
    check_connection(None)

    investment_universe_tab = widgets.Output()
    strategy_tab = widgets.Output()
    positioning_tab = widgets.Output()
    ex_post_tab = widgets.Output()
    risk_analysis_tab = widgets.Output()
    market_risk_tab = widgets.Output()
    

    main_tabs = widgets.Tab(children=[
        investment_universe_tab,
        strategy_tab,
        ex_post_tab,
        risk_analysis_tab,
        market_risk_tab
    ])
    
    main_tabs.set_title(0, 'Investment Universe')
    main_tabs.set_title(1, 'Strategy')
    main_tabs.set_title(2, 'Current Portfolio')
    main_tabs.set_title(3, 'Risk Analysis')
    main_tabs.set_title(4, 'Market Risk')
    
    with investment_universe_tab:
        display(universe_ui)

    strategy_constraints_tab = widgets.Output()
    strategy_positions_tab = widgets.Output()
    strategy_returns_tab = widgets.Output()
    
    strategy_subtabs = widgets.Tab(children=[
        strategy_constraints_tab,
        strategy_positions_tab,
        strategy_returns_tab
    ])
    
    strategy_subtabs.set_title(0, 'Strategy')
    strategy_subtabs.set_title(1, 'Positioning')
    strategy_subtabs.set_title(2, 'Strategy Return')
    
    with strategy_tab:
        display(strategy_subtabs)
    
    with strategy_constraints_tab:
        display(constraint_ui)
    
    with strategy_positions_tab:
        display(positions_ui)
    
    with strategy_returns_tab:
        display(calendar_perf)


    pnl_sub_tab = widgets.Output()
    positions_subtab = widgets.Output()
    calendar_ex_post_subtab = widgets.Output()

    
    ex_post_subtabs = widgets.Tab(children=[
        pnl_sub_tab,
        positions_subtab,
        calendar_ex_post_subtab
    ])
    
    ex_post_subtabs.set_title(0, 'P&L')
    ex_post_subtabs.set_title(1, 'Positioning')
    ex_post_subtabs.set_title(2, 'Calendar Return')
    
    with ex_post_tab:
        display(ex_post_subtabs)
        
    with pnl_sub_tab:
        display(ex_post_ui)
        
    with positions_subtab:
        display(positions_ui)

    with calendar_ex_post_subtab:
        display(calendar_ui_ex_post)


    risk_contribution_tab = widgets.Output()
    var_tab = widgets.Output()
    var_history_tab = widgets.Output()
    
    risk_exposure=widgets.Output()
    tracking_error_exposure=widgets.Output()

    
    risk_subtabs = widgets.Tab(children=[
        risk_contribution_tab,
        risk_exposure,
        tracking_error_exposure])
    
    var_subtabs=widgets.Tab(children=[var_tab,var_history_tab])
    var_subtabs.set_title(0, 'Value at Risk')
    var_subtabs.set_title(1, 'Value at Risk History')

    risk_subtabs.set_title(0, 'Risk Contribution')
    risk_subtabs.set_title(1, 'Risk Trajectory')
    risk_subtabs.set_title(2, 'Tracking Error')

    with risk_analysis_tab:
        risk_tabs=widgets.Tab(children=[risk_subtabs,var_subtabs])
        risk_tabs.set_title(0,'Volatility Analysis')
        risk_tabs.set_title(1,'Value at Risk')
        display(risk_tabs)
        
    with risk_contribution_tab:
        display(ex_ante_ui)
        
    with var_tab:
        display(var_ui)  
        
    with var_history_tab:
        display(var_history_ui)
        
    with risk_exposure:
        display(risk_exposure_ui)

    with tracking_error_exposure:
        display(tracking_error_exposure_ui)
        
    market_risk_detail_tab = widgets.Output()
    correlation_tab = widgets.Output()
    market_factors_tab = widgets.Output()
    
    market_risk_subtabs = widgets.Tab(children=[
        market_risk_detail_tab,
        correlation_tab,
        market_factors_tab])
    
    market_risk_subtabs.set_title(0, 'Market Risk')
    market_risk_subtabs.set_title(1, 'Correlation')
    market_risk_subtabs.set_title(2, 'Market Drivers')

    with market_risk_tab:
        display(market_risk_subtabs)
    
    with market_risk_detail_tab:
        display(market_ui)
    
    with correlation_tab:
        display(correlation_ui)
        
    with market_factors_tab:
        display(market_factors_ui)
        
    display(main_tabs)