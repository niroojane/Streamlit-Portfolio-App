# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


import streamlit as st
import pandas as pd
import random
import numpy as np
import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from RiskMetrics import *
from Price_Endpoint import *
from Stock_Data import get_close
from Rebalancing import *

from Metrics import *



def load_data(tickers,start_date=datetime.datetime(2023,1,1),today=None):
    if today is None:
        today=datetime.datetime.today()
    
    days=(today-start_date).days
    
    remaining=days%500
    numbers_of_table=days//500

    temp_end=start_date
    scope_prices=pd.DataFrame()
    for i in range(numbers_of_table+1):
        data=get_price(tickers,temp_end)
        temp_end=temp_end+datetime.timedelta(500)
        scope_prices=scope_prices.combine_first(data)
        
    temp_end=(today-datetime.timedelta(remaining))
    data=get_price(tickers,temp_end)
    scope_prices=scope_prices.combine_first(data)
    scope_prices=scope_prices.sort_index()
    scope_prices = scope_prices[~scope_prices.index.duplicated(keep='first')]
    scope_prices.index=pd.to_datetime(scope_prices.index)

    trx=get_close(['TRX-USD'],start=(start_date-datetime.timedelta(1)).strftime("%Y-%m-%d"),end=today.strftime("%Y-%m-%d"))
    trx.index=pd.to_datetime(trx.index)
    trx = trx[~trx.index.duplicated(keep='first')]
    trx=trx.sort_index().dropna()
    trx_returns=trx.pct_change().sort_index()
    scope_prices=pd.concat([trx,scope_prices],axis=1)

    returns=np.log(1+scope_prices.pct_change(fill_method=None))
    returns.index=pd.to_datetime(returns.index)
    with_no_na=returns.columns[np.where((returns.isna().sum()<30))]
    returns_to_use=returns[with_no_na].sort_index()


    dataframe=scope_prices[with_no_na].sort_index()
    dataframe.index=pd.to_datetime(dataframe.index)
    
    returns_to_use.index=pd.to_datetime(returns_to_use.index)
    returns_to_use = returns_to_use[~returns_to_use.index.duplicated(keep='first')]
    st.session_state.dataframe = dataframe.ffill()
    st.session_state.returns_to_use = returns_to_use.fillna(0)

tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(["Investment Universe", "Strategy","Strategy Return","Risk Contribution","Value at Risk","Market Risk","Correlation"])

            
with tab1:
    
    selected_number = st.slider(
        "Number of Crypto:",
        min_value=1,
        max_value=40,
        value=20,     
        step=1           
    )
    tickers=get_market_cap()['Ticker'].iloc[:selected_number].to_list()
    market_cap_table=get_market_cap()[['Long name','Ticker','Market Cap','Supply']].set_index('Ticker').iloc[:selected_number]
    selected = st.multiselect("Select Crypto:", tickers,default=tickers)
    
    st.dataframe(market_cap_table)
    
    starting_date= st.date_input("Starting Date", datetime.datetime(2020, 1, 1))
    dt = datetime.datetime.combine(starting_date, datetime.datetime.min.time())

    price_button=st.button(label='Get Prices')
        
    if price_button:
        with st.spinner("Loading market data...",show_time=True):
            load_data(tickers,dt)
            st.success("Done!")
            
    if "dataframe" not in st.session_state:
        st.info("Click the button to load data ⬆️")
    else:
        
        dataframe=st.session_state.dataframe
        returns_to_use=st.session_state.returns_to_use
        
        max_value = dataframe.index.max().strftime('%Y-%m-%d')
        min_value = dataframe.index.min().strftime('%Y-%m-%d')
        max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
        min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')
        value=(min_value,max_value)
        
        Model = st.slider(
            'Date:',
            min_value=min_value,
            max_value=max_value,
            value=value,key='tab1')
        
        selmin, selmax = Model
        selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
        selmaxd = selmax.strftime('%Y-%m-%d')
        
        # Filter data by selected date range
        mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
        
        asset_returns=get_asset_returns(dataframe.loc[mask])
        asset_risk=get_asset_risk(dataframe.loc[mask])
        
        st.dataframe(dataframe.loc[mask])
    
        st.dataframe(asset_returns)
        st.dataframe(asset_risk)
        
        fig = px.line(dataframe.loc[mask], title='Price', width=800, height=400)
        fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
        fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
        fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["BTCUSDT"])
    
        st.plotly_chart(fig)

        cumulative_returns=returns_to_use.loc[mask].copy()
        cumulative_returns.iloc[0]=0
        cumulative_returns=(1+cumulative_returns).cumprod()*100
        
        fig2 = px.line(cumulative_returns, title='Cumulative Performance', width=800, height=400)
        fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
        fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
        fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["BTCUSDT"])
    
        st.plotly_chart(fig2)

with tab2:

    dico_strategies = {
    'Minimum Variance': 'minimum_variance',
    'Risk Parity': 'risk_parity',
    'Sharpe Ratio': 'sharpe_ratio',
    'Maximum Diversification':'maximum_diversification'}
    
    if "dataframe" not in st.session_state:
        st.info("Load data first ⬅️")
    else:
        dataframe = st.session_state.dataframe
        returns_to_use = st.session_state.returns_to_use
        max_value = dataframe.index.max().strftime('%Y-%m-%d')
        min_value = dataframe.index.min().strftime('%Y-%m-%d')
        max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
        min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')
        value=(min_value,max_value)
        
        Model2 = st.slider(
            'Date:',
            min_value=min_value,
            max_value=max_value,
            value=value,key='tab2')
    
        selmin, selmax = Model2
        selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
        selmaxd = selmax.strftime('%Y-%m-%d')
        
        # Filter data by selected date range
        mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
        
        range_prices=dataframe.loc[mask].copy()
        range_returns=returns_to_use.loc[mask].copy()
        
        portfolio=RiskAnalysis(range_returns)
        
        asset_returns=get_asset_returns(range_prices)
        asset_risk=get_asset_risk(range_prices)
        
        st.dataframe(asset_returns)
        st.dataframe(asset_risk)
                
    
        st.subheader("Constraints")  
                
        data = pd.DataFrame({'Asset':[None],
        'Sign':[None],
        'Limit':[None]
        })
        drop_down_list=list(range_returns.columns)+['All']
        # Define dropdown options for the 'Risk Level' column
        column_config = {'Asset':st.column_config.SelectboxColumn(
            options=drop_down_list),
        'Sign': st.column_config.SelectboxColumn(
            options=["=", "≥", "≤"],  # Dropdown options
            help="Select the risk level for each asset."  # Tooltip for the column
        )
        }
        
        # Create the editable data editor with dropdown
        editable_data = st.data_editor(
        data,
        column_config=column_config,
        num_rows="dynamic",  # Allow rows to be added dynamically
        )
    
        constraint_matrix=editable_data.to_numpy()
        constraints=[]

        try:
            for row in range(constraint_matrix.shape[0]):
                temp = constraint_matrix[row, :]
                ticker = temp[0]
                
                if ticker not in drop_down_list:
                    continue
                    
                sign = temp[1]
                limit = float(temp[2])

                if ticker=='All':
                    constraint= diversification_constraint(sign,limit)
                else:
                    position = np.where(range_prices.columns == ticker)[0][0]
                    constraint = create_constraint(sign, limit, position)
                    
                constraints.extend(constraint)
                
        
        except Exception as e:
            pass



        st.subheader("Portfolio Construction")

        allocation={}
        
        optimized_weights_constraint = portfolio.optimize(objective="sharpe_ratio",constraints=constraints)
        minvar_weights_constraint = portfolio.optimize(objective="minimum_variance",constraints=constraints)
        risk_parity_weights_constraint = portfolio.optimize(objective="risk_parity",constraints=constraints)
        max_diversification_weights_constraint=portfolio.optimize("maximum_diversification",constraints=constraints)
        
        optimized_weights = portfolio.optimize(objective="sharpe_ratio")
        minvar_weights = portfolio.optimize(objective="minimum_variance")
        risk_parity_weights = portfolio.optimize(objective="risk_parity")
        max_diversification=portfolio.optimize(objective="maximum_diversification")
        
        allocation['Optimal Portfolio']=optimized_weights.tolist()
        allocation['Optimal Constrained Portfolio']=optimized_weights_constraint.tolist()

        allocation['Minimum Variance Portfolio']=minvar_weights.tolist()
        allocation['Minimum Variance Constrained Portfolio']=minvar_weights_constraint.tolist()
        
        allocation['Maximum Diversification Portfolio']=max_diversification.tolist()
        allocation['Maximum Diversification Constrained Portfolio']=max_diversification_weights_constraint.tolist()
        
        allocation['Risk Parity Portfolio']=risk_parity_weights.tolist()
        allocation['Risk Parity Constrained Portfolio']=risk_parity_weights_constraint.tolist()

        
        
        allocation_dataframe = pd.DataFrame(
                allocation,
                index=dataframe.columns
            ).T.round(6)
        
        
        st.session_state.allocation_dataframe = st.data_editor(
            allocation_dataframe,
            num_rows="dynamic",
        key='allocation_editor')

    options_strat = list(dico_strategies.keys())
    rebalancing_frequency = ['Monthly', 'Quarterly', 'Yearly']
    
    selected_strategy = st.selectbox("Strategy:", options_strat, index=0)
    selected_frequency = st.selectbox("Rebalancing Frequency:", rebalancing_frequency, index=0)
    benchmark_tracking_error = st.selectbox("Benchmark:", list(allocation_dataframe.index), index=0)
    window_vol = st.number_input("Sliding Window Size:", min_value=1, value=252, step=1)
    
    if "run_optimization" not in st.session_state:
        st.session_state.run_optimization = False
    if "results" not in st.session_state:
        st.session_state.results = None
    

    if st.button("Run Optimization"):
        st.session_state.run_optimization = True
        st.session_state.results = None  
    
    if st.session_state.run_optimization and st.session_state.results is None:
    

        freq_map = {
            'Monthly': pd.offsets.BMonthEnd(),
            'Quarterly': pd.offsets.BQuarterEnd(),
            'Yearly': pd.offsets.BYearEnd()
        }
        offset = freq_map.get(selected_frequency, pd.offsets.BMonthEnd())
    
        range_prices.index = pd.to_datetime(range_prices.index)
        range_returns.index = pd.to_datetime(range_returns.index)
        returns_to_use.index = pd.to_datetime(returns_to_use.index)
    
        candidate_anchors = pd.DatetimeIndex(sorted(set(range_prices.index + offset)))
        if candidate_anchors.empty:
            candidate_anchors = pd.DatetimeIndex([range_returns.index[-1]])
    
        idx = range_returns.index.get_indexer(candidate_anchors, method='nearest')
        idx = idx[idx >= 0]
    
        selected_dates = sorted(list(set(range_returns.index[idx].tolist() + [returns_to_use.index[-1]])))
        dates_end = selected_dates
    
        if len(dates_end) < 2:
            st.warning("⚠️ Not enough anchor dates for rolling optimization.")
    
        results_dict = {}
        for i in range(len(dates_end) - 1):
            dataset = range_returns.loc[dates_end[i]:dates_end[i+1]]
            risk = RiskAnalysis(dataset)
            date = dataset.index[-1]
    
            optimal = risk.optimize(
                objective=dico_strategies[selected_strategy],
                constraints=constraints
            )
            results_dict[date] = np.round(optimal, 6)
    
        rolling_optimization = pd.DataFrame(results_dict, index=dataframe.columns).T
        rolling_optimization.loc[dates_end[0]] = 1 / len(dataframe.columns)
        rolling_optimization = rolling_optimization.sort_index()

        model = pd.DataFrame(rolling_optimization.iloc[-2])
        model.columns = ["Model"]
        alloc_df = st.session_state.allocation_dataframe.copy()
    
        if "Model" in alloc_df.index:
            alloc_df.loc["Model"] = model.T
        else:
            alloc_df = pd.concat([alloc_df, model.T], axis=0)
    
        quantities = rebalanced_dynamic_quantities(dataframe, rolling_optimization)
        performance_fund = pd.DataFrame({'Fund': (quantities * dataframe).sum(axis=1)})
    
        if 'BTCUSDT' in range_prices.columns:
            performance_fund['Bitcoin'] = range_prices['BTCUSDT']
    
        performance_pct = performance_fund.pct_change(fill_method=None)
        cumulative = (1 + performance_pct).cumprod() * 100
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    
        date_drawdown = drawdown.idxmin().dt.date
        max_drawdown = drawdown.min()
    
        metrics = {}
        metrics['Tracking Error'] = ((performance_pct['Fund'] - performance_pct['Bitcoin']).std() * np.sqrt(252)).round(4)
        metrics['Fund Vol'] = (performance_pct['Fund'].std() * np.sqrt(252)).round(4)
        metrics['Bitcoin Vol'] = (performance_pct['Bitcoin'].std() * np.sqrt(252)).round(4)
        metrics['Fund Return'] = (performance_fund['Fund'].iloc[-2] / performance_fund['Fund'].iloc[0]).round(4)
        metrics['Bitcoin Return'] = (performance_fund['Bitcoin'].iloc[-2] / performance_fund['Bitcoin'].iloc[0]).round(4)
        metrics['Sharpe Ratio'] = ((1 + metrics['Fund Return']) ** (1 / len(set(returns_to_use.index.year))) / metrics['Fund Vol']).round(4)
        metrics['Bitcoin Sharpe Ratio'] = ((1 + metrics['Bitcoin Return']) ** (1 / len(set(returns_to_use.index.year))) / metrics['Bitcoin Vol']).round(4)
        metrics['Fund Drawdown'] = max_drawdown['Fund'].round(4)
        metrics['Bitcoin Drawdown'] = max_drawdown['Bitcoin'].round(4)
        metrics['Fund Date Drawdown'] = date_drawdown['Fund']
        metrics['Bitcoin Date Drawdown'] = date_drawdown['Bitcoin']
    
        indicators = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Indicators'])
    
        cumulative_performance = performance_pct.loc[mask]
        cumulative_performance.iloc[0] = 0
        cumulative_results = (1 + cumulative_performance).cumprod() * 100

        portfolio_returns = rebalanced_time_series(range_prices, alloc_df, frequency=selected_frequency)
        cumulative_results = pd.concat([cumulative_results, portfolio_returns], axis=1)
        drawdown = (cumulative_results - cumulative_results.cummax()) / cumulative_results.cummax()
        rolling_vol_ptf = cumulative_results.pct_change().rolling(window_vol).std() * np.sqrt(260)

        frontier_indicators, fig4 = get_frontier(range_returns, alloc_df)


        fig = px.line(cumulative_results, title='Performance', width=800, height=400)
        fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
        fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
        fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
    
        fig2 = px.line(drawdown, title='Drawdown', width=800, height=400)
        fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
        fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
        fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))

    
        fig3 = px.line(rolling_vol_ptf, title="Portfolio Rolling Volatility").update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
        fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400) 
        fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
        fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
    

        fig4.update_layout(width=800, height=400,title={'text': "Efficient Frontier"})
        fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))    

        st.session_state.results = {
            "rolling_optimization": rolling_optimization,
            "alloc_df": alloc_df,
            "quantities": quantities,
            "performance_pct": performance_pct,
            "cumulative_results": cumulative_results,
            "drawdown": drawdown,
            "rolling_vol_ptf": rolling_vol_ptf,
            "indicators": indicators,
            "frontier_indicators": frontier_indicators,
            "fig_performance": fig,
            "fig_drawdown": fig2,
            "fig_rolling_vol": fig3,
            "fig_frontier": fig4
        }
    if st.session_state.results is not None:
        res = st.session_state.results
    
        st.subheader("Weights Matrix")
        st.dataframe(res["rolling_optimization"])
        st.subheader("Allocation Table")
        st.dataframe(res["alloc_df"])
    
        st.subheader("Metrics")
        st.dataframe(res["indicators"])
        st.dataframe(rebalanced_metrics(res['cumulative_results']))
        st.dataframe(get_portfolio_risk(res["alloc_df"], range_prices, res['cumulative_results'], benchmark_tracking_error))

        st.subheader("Charts")
        if "fig_performance" in res:
            st.plotly_chart(res["fig_performance"], use_container_width=True)
        if "fig_drawdown" in res:
            st.plotly_chart(res["fig_drawdown"], use_container_width=True)
        if "fig_rolling_vol" in res:
            st.plotly_chart(res["fig_rolling_vol"], use_container_width=True)
        if "fig_frontier" in res:
            st.plotly_chart(res["fig_frontier"], use_container_width=True)
            
        st.subheader("Time Series")
        st.dataframe(res["cumulative_results"])

        
with tab3:
    if st.session_state.run_optimization == False and st.session_state.results is None:
        st.info("Compute Optimization first ⬅️")

    elif "dataframe" not in st.session_state:
        st.info("Load data first ⬅️")
        
    else:

        rebalancing_frequency=['Month', 'Year']
        allocation_dataframe=st.session_state.allocation_dataframe
        cumulative_results=st.session_state.results['cumulative_results']
        
        selected_frequency_calendar = st.selectbox("Rebalancing Frequency:", rebalancing_frequency,index=1,key='selected_frequency_calendar')
        fund_calendar=st.selectbox("Fund:", list(cumulative_results.columns),index=0,key='fund_calendar')
        benchmark_calendar=st.selectbox("Benchmark:", list(cumulative_results.columns),index=1,key='benchmark_calendar')

        if benchmark_calendar==fund_calendar:
            st.info("Benchmark and Fund must be different ⬅️")
        else:
            graphs=get_calendar_graph(cumulative_results, 
                               freq=selected_frequency_calendar, 
                               benchmark=benchmark_calendar, 
                               fund=fund_calendar)
            for name, fig in graphs.items():
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{name}")

with tab4:    

    if "dataframe" not in st.session_state:
        st.info("Load data first ⬅️")
        
    elif "allocation_dataframe" not in st.session_state:
        st.info("Compute Optimization first ⬅️")

    else:

        
        dataframe = st.session_state.dataframe
        returns_to_use = st.session_state.returns_to_use
        allocation_dataframe=st.session_state.allocation_dataframe
        
        st.subheader("Allocation")
        
        st.dataframe(allocation_dataframe)
        
        max_value = dataframe.index.max().strftime('%Y-%m-%d')
        min_value = dataframe.index.min().strftime('%Y-%m-%d')
        max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
        min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')
        value=(min_value,max_value)
        
        st.subheader("Risk Decomposition")

        Model3 = st.slider(
            'Date:',
            min_value=min_value,
            max_value=max_value,
            value=value,key='tab3')
    
        selmin, selmax = Model3
        selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
        selmaxd = selmax.strftime('%Y-%m-%d')
        
        mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
        
        range_prices=dataframe.loc[mask].copy()
        range_returns=returns_to_use.loc[mask].copy()

        portfolio = RiskAnalysis(range_returns)     

        fund_risk=st.selectbox("Fund:", list(allocation_dataframe.index),index=0,key='fund_risk')
        benchmark_risk=st.selectbox("Benchmark:", list(allocation_dataframe.index),index=1,key='benchmark_risk')
        
        selected_weights = allocation_dataframe.loc[fund_risk]
        
        decomposition = pd.DataFrame(portfolio.var_contrib_pct(selected_weights))*100
        
        quantities_rebalanced = rebalanced_portfolio(range_prices, selected_weights) / range_prices
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
        profit_and_loss_simulated=profit_and_loss_simulated.sort_values(by='Variance Contribution in %', ascending=False)
    
        vol_ex_ante = {}
        tracking_error_ex_ante = {}
        
        for idx in allocation_dataframe.index:
            vol_ex_ante[idx] = portfolio.variance(allocation_dataframe.loc[idx])
            tracking_error_ex_ante[idx] = portfolio.variance(allocation_dataframe.loc[idx] - allocation_dataframe.loc[benchmark_risk])

        data = {
            'Vol Ex Ante': vol_ex_ante,
            'Tracking Error Ex Ante': tracking_error_ex_ante
        }
        
        ex_ante_dataframe = pd.DataFrame(data)


        st.dataframe(profit_and_loss_simulated)

        st.subheader("Ex Ante Metrics")

        st.dataframe(ex_ante_dataframe)


        
    