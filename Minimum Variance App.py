# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


import streamlit as st
import pandas as pd
import random
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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


st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Global font override */
    html, body, .stApp, [class*="css"]  {
        font-family: "Arial Narrow", Arial, sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def load_data(tickers,start_date=datetime.datetime(2023,1,1),today=None):
    
    if today is None:
        today=datetime.datetime.today()
    
    days=(today-start_date).days
    
    remaining=days%500
    numbers_of_table=days//500
    start_dt = datetime.datetime.combine(start_date, datetime.time())
    
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

    scope_prices = None

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(get_price, tickers,d) for d in end_dates]

            for future in as_completed(futures):
                data = future.result()

                if scope_prices is None:
                    scope_prices = data
                else:
                    scope_prices = scope_prices.combine_first(data)

    except Exception as e:
        print("❌ Error while fetching prices:", e)
        return
        
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

    
def process_index(index,allocation,dataframe,iterations,stress_factor,var_centile,num_scenarios):
    
    horizon = 1 / 250
    spot = dataframe.iloc[-1]
    theta = 2
    
    range_returns=dataframe.pct_change()

    distrib_functions = {
    'multivariate_distribution': (iterations, stress_factor),
    'gaussian_copula': (iterations, stress_factor),
    't_copula': (iterations, stress_factor),
    'gumbel_copula': (iterations, theta),
    'monte_carlo': (spot, horizon, iterations, stress_factor)
    }
    
    portfolio = RiskAnalysis(range_returns)

    vs, cvs = {}, {}
    for func_name, args in distrib_functions.items():
        func = getattr(portfolio, func_name)
        scenarios = {}

        for i in range(num_scenarios):
            if func_name == 'monte_carlo':
                distrib = pd.DataFrame(func(*args)[1], columns=portfolio.returns.columns)
            else:
                distrib = pd.DataFrame(func(*args), columns=portfolio.returns.columns)

            distrib = distrib * allocation.loc[index]
            distrib = distrib[distrib.columns[allocation.loc[index] > 0]]
            distrib['Portfolio'] = distrib.sum(axis=1)

            results = distrib.sort_values(by='Portfolio').iloc[int(distrib.shape[0] * var_centile)]
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

main_tabs=st.tabs(["Investment Universe","Strategy","Risk Analysis","Market Risk"])
    

with main_tabs[0]:
    
    selected_number = st.slider(
        "Number of Crypto:",
        min_value=1,
        max_value=40,
        value=20,     
        step=1           
    )

    
    tickers_market_cap=get_market_cap()
    market_cap_table=tickers_market_cap.iloc[:selected_number].set_index('Ticker')

    tickers=tickers_market_cap['Ticker'].iloc[:selected_number].to_list()
    
    selected = st.multiselect("Select Crypto:", tickers,default=tickers)
    
    st.dataframe(market_cap_table)
    
    starting_date= st.date_input("Starting Date", datetime.datetime(2020, 1, 1))
    dt = datetime.datetime.combine(starting_date, datetime.datetime.min.time())
    
    price_button=st.button(label='Get Prices')
           
    if price_button:
        with st.spinner("Loading market data...",show_time=True):
            load_data(selected,dt)
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
            value=value,key='investment_tab')
        
        selmin, selmax = Model
        selmind = selmin.strftime('%Y-%m-%d')
        selmaxd = selmax.strftime('%Y-%m-%d')
        
        mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
        
        asset_returns=get_asset_returns(dataframe.loc[mask])
        asset_risk=get_asset_risk(dataframe.loc[mask])
        
    
        st.dataframe(asset_returns,width='stretch')
        st.dataframe(asset_risk,width='stretch')
        
        fig = px.line(dataframe.loc[mask], title='Price', width=800, height=400, render_mode = 'svg')
        fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
        fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
        fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["BTCUSDT"])
        


        cumulative_returns=returns_to_use.loc[mask].copy()
        cumulative_returns.iloc[0]=0
        cumulative_returns=(1+cumulative_returns).cumprod()*100
        
        fig2 = px.line(cumulative_returns, title='Cumulative Performance', width=800, height=400, render_mode = 'svg')
        fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
        fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
        fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["BTCUSDT"])
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(fig,width='content')
        with col2:
            st.plotly_chart(fig2,width='content')
        st.dataframe(dataframe.loc[mask],width='stretch')

with main_tabs[1]:

    dico_strategies = {
    'Minimum Variance': 'minimum_variance',
    'Risk Parity': 'risk_parity',
    'Sharpe Ratio': 'sharpe_ratio',
    'Maximum Diversification':'maximum_diversification'}
    
    if "dataframe" not in st.session_state:
        st.info("Load data first ⬅️")
        
    else:

        sub_tabs=st.tabs(["Strategy","Strategy Return"])

        with sub_tabs[0]:

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
                value=value,key='strategy_tab')
        
            selmin, selmax = Model2
            selmind = selmin.strftime('%Y-%m-%d') 
            selmaxd = selmax.strftime('%Y-%m-%d')
            
            mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
            
            range_prices=dataframe.loc[mask].copy()
            range_returns=returns_to_use.loc[mask].copy()
            
            portfolio=RiskAnalysis(range_returns)
            
            asset_returns=get_asset_returns(range_prices)
            asset_risk=get_asset_risk(range_prices)
            
            st.dataframe(asset_returns,width='stretch')
            st.dataframe(asset_risk,width='stretch')
        
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
                options=["=", "≥", "≤"],
                help="Select the risk level for each asset." 
            )
            }

            
            editable_data = st.data_editor(
            data,
            column_config=column_config,
            num_rows="dynamic",
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
            equal_weights = np.ones(returns_to_use.shape[1]) / returns_to_use.shape[1]
    
            allocation['Optimal Portfolio']=optimized_weights.tolist()
            allocation['Optimal Constrained Portfolio']=optimized_weights_constraint.tolist()
    
            allocation['Minimum Variance Portfolio']=minvar_weights.tolist()
            allocation['Minimum Variance Constrained Portfolio']=minvar_weights_constraint.tolist()
            
            allocation['Maximum Diversification Portfolio']=max_diversification.tolist()
            allocation['Maximum Diversification Constrained Portfolio']=max_diversification_weights_constraint.tolist()
            
            allocation['Risk Parity Portfolio']=risk_parity_weights.tolist()
            allocation['Risk Parity Constrained Portfolio']=risk_parity_weights_constraint.tolist()
            allocation['Equal Weighted']=equal_weights.tolist()
            
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
            benchmark_tracking_error = st.selectbox("Benchmark:", list(allocation_dataframe.index), index=0)
            selected_frequency = st.selectbox("Rebalancing Frequency:", rebalancing_frequency, index=0)
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
        
                st.session_state.results = {
                    "rolling_optimization": rolling_optimization,
                    "alloc_df": alloc_df,
                    "quantities": quantities,
                    "performance_pct": performance_pct,
                    "cumulative_results":cumulative_results,
                    "indicators":indicators}
                
            if st.session_state.results is not None:
                
                selmin, selmax = st.session_state['strategy_tab']
                selmind = selmin.strftime('%Y-%m-%d') 
                selmaxd = selmax.strftime('%Y-%m-%d')
                
                res=st.session_state.results
                mask = (res['cumulative_results'].index >= selmind) & (res['cumulative_results'].index <= selmaxd)

                res=st.session_state.results
                
                cumulative_performance=res['cumulative_results'].loc[mask].pct_change()
                cumulative_performance.iloc[0] = 0
                cumulative_results = (1 + cumulative_performance).cumprod() * 100
                
                drawdown = (cumulative_results - cumulative_results.cummax()) / cumulative_results.cummax()
                rolling_vol_ptf = cumulative_results.pct_change().rolling(window_vol).std() * np.sqrt(260)
                
                frontier_indicators, fig4 = get_frontier(range_returns, res['alloc_df'])
        
                fig = px.line(cumulative_results, title='Performance', width=800, height=400, render_mode = 'svg')
                fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
                fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
            
                fig2 = px.line(drawdown, title='Drawdown', width=800, height=400, render_mode = 'svg')
                fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
                fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
        
            
                fig3 = px.line(rolling_vol_ptf, title="Portfolio Rolling Volatility", render_mode = 'svg').update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
                fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400) 
                fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Fund","Bitcoin"])
                fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
            
                fig4.update_layout(width=800, height=400,title={'text': "Efficient Frontier"})
                fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))    
            
                res = st.session_state.results
                
                st.subheader("Weights Matrix")
                st.dataframe(res["rolling_optimization"],width='stretch')
                st.subheader("Allocation Table")
                st.dataframe(res["alloc_df"],width='stretch')
    
                
                st.subheader("Expected Returns")
                st.dataframe(frontier_indicators,width='stretch')
    
                st.subheader("Systematic Fund Metrics")
                st.dataframe(res["indicators"],width='stretch')
                
                st.subheader("Backtested Metrics")
                st.dataframe(rebalanced_metrics(cumulative_results),width='stretch')
                st.dataframe(get_portfolio_risk(res["alloc_df"], range_prices, cumulative_results, benchmark_tracking_error),width='stretch')
        
                st.subheader("Charts")
                col1, col2 = st.columns([1, 1])
    
                with col1:
                        st.plotly_chart(fig, width='content')
                        st.plotly_chart(fig2, width='content')
                with col2:
                        st.plotly_chart(fig3, width='content')
                        st.plotly_chart(fig4, width='content')
                    
                st.subheader("Time Series")
                st.dataframe(cumulative_results,width='stretch')
            else:
                st.info("Compute Optimization first ⬅️")

        
        with sub_tabs[1]:
            
            # if "dataframe" not in st.session_state:
            #     st.info("Load data first ⬅️")
            
            if st.session_state.results is None:
                st.info("Compute Optimization first ⬅️")
                
            else:
        
                rebalancing_frequency=['Month', 'Year']
                res=st.session_state.results
                allocation_dataframe=res['alloc_df']
                cumulative_results=st.session_state.results['cumulative_results']
                
                col1, col2, col3 = st.columns([1, 1, 1])
            
                with col1:
                    selected_frequency_calendar = st.selectbox("Frequency:", rebalancing_frequency,index=1,key='selected_frequency_calendar')
        
                with col2:
                    fund_calendar=st.selectbox("Fund:", list(cumulative_results.columns),index=0,key='fund_calendar')
                            
                with col3:
                    benchmark_calendar=st.selectbox("Benchmark:", list(cumulative_results.columns),index=1,key='benchmark_calendar')
                    
        
                if benchmark_calendar==fund_calendar:
                    st.info("Benchmark and Fund must be different ⬅️")
                else:
                    graphs=get_calendar_graph(cumulative_results, 
                                       freq=selected_frequency_calendar, 
                                       benchmark=benchmark_calendar, 
                                       fund=fund_calendar)
                    # for name, fig in graphs.items():
                    #     st.plotly_chart(fig, width='content', key=f"plot_{name}")
                col1, col2 = st.columns([1, 1])
                keys=list(graphs.keys())
                with col1:
                    st.plotly_chart(graphs[keys[0]], width='content', key=f"plot_{keys[0]}")
                    st.plotly_chart(graphs[keys[2]], width='content', key=f"plot_{keys[1]}")
                with col2:
                    st.plotly_chart(graphs[keys[1]], width='content', key=f"plot_{keys[2]}")
                    st.plotly_chart(graphs[keys[3]], width='content', key=f"plot_{keys[3]}")           

        
with main_tabs[2]: 
    
    if "dataframe" not in st.session_state:
        st.info("Load data first ⬅️")
        
    elif st.session_state.results is None:
        st.info("Compute Optimization first ⬅️")

    else:
        sub_tabs_risk=st.tabs(['Risk Analysis','Value At Risk'])
        with sub_tabs_risk[0]:
            
            dataframe = st.session_state.dataframe
            returns_to_use = st.session_state.returns_to_use
            res=st.session_state.results
            allocation_dataframe=res["alloc_df"]
            
            max_value = dataframe.index.max().strftime('%Y-%m-%d')
            min_value = dataframe.index.min().strftime('%Y-%m-%d')
            
            max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
            min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')
            
            value=(min_value,max_value)
            
            Model3 = st.slider(
                'Date:',
                min_value=min_value,
                max_value=max_value,
                value=value,key='risk_tab')
        
            selmin, selmax = Model3
            selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
            selmaxd = selmax.strftime('%Y-%m-%d')
            
            mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
            
            range_prices=dataframe.loc[mask].copy()
            range_returns=returns_to_use.loc[mask].copy()
    
            portfolio = RiskAnalysis(range_returns)                
            
            st.subheader("Allocation")
            
            st.dataframe(allocation_dataframe,width='stretch')
            
            st.subheader("Risk Decomposition")
            
            col1, col2, col3 = st.columns([1, 1, 1])
        
            with col1:
                fund_risk=st.selectbox("Fund:", list(allocation_dataframe.index),index=0,key='fund_risk')
    
            with col2:
                benchmark_risk=st.selectbox("Benchmark:", list(allocation_dataframe.index),index=1,key='benchmark_risk')

            with col3:
                frequency_pnl=st.selectbox("Rebalancing Frequency:", ['Yearly','Quarterly','Monthly'],index=1,key='frequency_pnl')
            
            selected_weights = allocation_dataframe.loc[fund_risk]
            
            decomposition = pd.DataFrame(portfolio.var_contrib(selected_weights)[0])*100
 


            quantities_rebalanced = rebalanced_portfolio(range_prices, selected_weights,frequency=frequency_pnl) / range_prices
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
            profit_and_loss_simulated=profit_and_loss_simulated.sort_values(by='Vol Contribution', ascending=False)
        
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
    
            st.dataframe(profit_and_loss_simulated,width='stretch')

            st.subheader("Ex Ante Metrics")
    
            st.dataframe(ex_ante_dataframe,width='stretch')

        with sub_tabs_risk[1]:
        
            dataframe = st.session_state.dataframe
            returns_to_use = st.session_state.returns_to_use
            res=st.session_state.results
            allocation_dataframe=res["alloc_df"]
                
            max_value = dataframe.index.max().strftime('%Y-%m-%d')
            min_value = dataframe.index.min().strftime('%Y-%m-%d')
            max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
            min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')  
            value=(min_value,max_value)
    
            Model4 = st.slider(
            'Date:',
            min_value=min_value,
            max_value=max_value,
            value=value,key='var_tab')
        
            selmin, selmax = Model3
            selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
            selmaxd = selmax.strftime('%Y-%m-%d')
            
            mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
            
            range_prices=dataframe.loc[mask].copy()
            range_returns=returns_to_use.loc[mask].copy()
    
            
            stress_factor=st.number_input("Stress Factor:", min_value=1.0, value=1.0, step=1.0)
            iterations=st.number_input("Iterations:", min_value=1, value=10000, step=1)
            num_scenarios=st.number_input("Scenarios:", min_value=1, value=100, step=1)
            var_centile=st.number_input("Centile:", min_value=0.00, value=0.05, step=0.01)
    
            var_button=st.button("Run Simulation")
            var_status=st.empty()

            selected_fund_var=st.selectbox("Fund:", list(allocation_dataframe.index),index=0,key='selected_fund_var')

            
            var_scenarios, cvar_scenarios, fund_results = {}, {}, {}
            
            portfolio = RiskAnalysis(range_returns)
            
            if "fund_results" not in st.session_state:
                st.session_state.fund_results = None
                st.session_state.var_scenarios=None
                st.session_state.cvar_scenarios=None
                
            if var_button:
                with st.spinner("Computing VaR...",show_time=True):

                    st.session_state.fund_results=None
                    st.session_state.var_scenarios=None
                    st.session_state.cvar_scenarios=None
    
                    tasks=[(idx,allocation_dataframe,range_prices,iterations,stress_factor,var_centile,num_scenarios) for idx in allocation_dataframe.index]
                    
                    var_scenarios={}
                    cvar_scenarios={}
                    fund_results={}
                    
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        futures = {executor.submit(process_index,idx,allocation_dataframe,range_prices,iterations,stress_factor,var_centile,num_scenarios): (idx,allocation_dataframe,range_prices,iterations,stress_factor,var_centile,num_scenarios)
                                   for idx,allocation_dataframe,range_prices,iterations,stress_factor,var_centile,num_scenarios in tasks}
                        for future in as_completed(futures):
                            idx, vs, cvs, fund_result = future.result()
                            var_scenarios[idx] = vs
                            cvar_scenarios[idx] = cvs
                            fund_results[idx] = fund_result

                    st.session_state.var_scenarios = var_scenarios
                    st.session_state.cvar_scenarios = cvar_scenarios
                    st.session_state.fund_results = fund_results
                    var_status.success('Done!')
    
        
            if st.session_state.fund_results is not None:
                
                var_scenarios=st.session_state.var_scenarios
                cvar_scenarios=st.session_state.cvar_scenarios
                fund_results=st.session_state.fund_results   
                
                columns = ['Multivariate', 'Gaussian Copula', 'T-Student Copula', 'Gumbel Copula', 'Monte Carlo']
            
                var_dataframe = pd.DataFrame(var_scenarios[selected_fund_var])
                var_dataframe.columns = columns
            
                cvar_dataframe = pd.DataFrame(cvar_scenarios[selected_fund_var])
                cvar_dataframe.columns = columns
            
                fund_results_dataframe = pd.DataFrame(fund_results).T
                
                st.dataframe(var_dataframe,width='stretch')
                st.dataframe(cvar_dataframe,width='stretch')
                st.dataframe(fund_results_dataframe,width='stretch')

with main_tabs[3]:
    
    if "dataframe" not in st.session_state:
        st.info("Load data first ⬅️")
    else:

        sub_tabs_market=st.tabs(['Market Risk','Correlation'])

        with sub_tabs_market[0]:
            dataframe = st.session_state.dataframe
            returns_to_use = st.session_state.returns_to_use
            market_tickers=[t for t in tickers if t in dataframe.columns]
    
            max_value = dataframe.index.max().strftime('%Y-%m-%d')
            min_value = dataframe.index.min().strftime('%Y-%m-%d')
            max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
            min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')  
            value=(min_value,max_value)
            
            Model5 = st.slider(
            'Date:',
            min_value=min_value,
            max_value=max_value,
            value=value,key='market_tab')
        
            selmin, selmax = Model5
            selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
            selmaxd = selmax.strftime('%Y-%m-%d')
            
            mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
            
            range_prices=dataframe.loc[mask].copy()
            range_returns=returns_to_use.loc[mask].copy()
    
            portfolio=RiskAnalysis(range_returns)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                num_components=st.number_input("PCA Components:",min_value=1,value=min(5,range_returns.shape[1]),max_value=range_returns.shape[1]+1)
    
            eigval,eigvec,portfolio_components=portfolio.pca(num_components=num_components)

            with col2:
                selected_components=st.selectbox("Select PCA:", list(portfolio_components.columns),index=0,key='selected_pca')
                        
            with col3:
                num_closest_to_pca=st.number_input("Closest to PCA:",min_value=1,value=min(5,range_returns.shape[1]),max_value=range_returns.shape[1]+1)

            variance_explained=eigval/eigval.sum()
            variance_explained_dataframe=pd.DataFrame(variance_explained,index=portfolio_components.columns,columns=['Variance Explained'])
            
            pca_weight=dict((portfolio_components[selected_components]/(portfolio_components[selected_components]).sum()))
            pca_portfolio=pd.DataFrame(portfolio_components[selected_components]).sort_values(by=selected_components,ascending=False)
            
            historical_PCA=pd.DataFrame(np.array(list(pca_weight.values())).dot(np.transpose(portfolio.returns)),index=portfolio.returns.index,columns=['PCA'])
            historical_PCA=historical_PCA.dropna()
        
            comparison=portfolio.returns.copy()
            comparison['PCA']=historical_PCA
            distances=np.sqrt(np.sum(comparison.apply(lambda y:(y-historical_PCA['PCA'])**2),axis=0)).sort_values()
            pca_similarity=(1+comparison[distances.index[:num_closest_to_pca]]).cumprod()
    
    
            fig=px.bar(variance_explained_dataframe,title='Variance Explanation in %')
            fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400) 
            fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
    
            fig2=px.bar(pca_portfolio,title='Eigen Weights')
            fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400) 
            fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
            
            fig3=px.line((1+historical_PCA).cumprod(),title='Eigen Index', render_mode = 'svg')
            fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
            fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
    
            fig4=px.line(pca_similarity,title='PCA Similarity', render_mode = 'svg')
            fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
            fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
            
            with col1:
                st.plotly_chart(fig,width='content')
            with col2:
                st.plotly_chart(fig2,width='content')
                st.plotly_chart(fig4,width='content')

            with col3:
                st.plotly_chart(fig3,width='content')
            
        with sub_tabs_market[1]:
      
            dataframe = st.session_state.dataframe
            returns_to_use = st.session_state.returns_to_use
            market_tickers=[t for t in tickers if t in dataframe.columns]
    
            max_value = dataframe.index.max().strftime('%Y-%m-%d')
            min_value = dataframe.index.min().strftime('%Y-%m-%d')
            max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
            min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')  
            value=(min_value,max_value)
    
            Model6 = st.slider(
            'Date:',
            min_value=min_value,
            max_value=max_value,
            value=value,key='correlation_tab')
        
            selmin, selmax = Model6
            selmind = selmin.strftime('%Y-%m-%d') 
            selmaxd = selmax.strftime('%Y-%m-%d')

            dropdown_asset1=st.selectbox("Asset 1:",options=range_returns.columns,index=0)
            dropdown_asset2=st.selectbox("Asset 2:",options=range_returns.columns,index=1)
            window_corr=st.number_input("Window Correlation",min_value=0,value=252)
    
            mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
            col1, col2, col3 = st.columns([1, 1, 1])

            
            range_prices=dataframe.loc[mask].copy()
            range_returns=returns_to_use.loc[mask].copy()
    
            pca_over_time=first_pca_over_time(returns=range_returns,window=window_corr)
    
            rolling_correlation = range_returns[dropdown_asset1].rolling(window_corr).corr(
                range_returns[dropdown_asset2]
            ).dropna()
            
            fig = px.line(rolling_correlation, title=f"{dropdown_asset1}/{dropdown_asset2} Correlation", render_mode = 'svg')
            fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
            fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
    
    
            fig2 = px.imshow(range_returns.corr().round(2), title='Correlation Matrix',color_continuous_scale='blues', text_auto=True, aspect="auto")
            fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
            fig2.update_traces(xgap=2, ygap=2)
            fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
            
            fig3=px.line(pca_over_time,title='First principal component (Variance Explained in %)', render_mode = 'svg')
            fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
            fig3.update_layout(xaxis_title=None, yaxis_title=None)

            with col1:
                st.plotly_chart(fig,width='content')
            with col2:
                st.plotly_chart(fig3,width='content')
            with col3:
                st.plotly_chart(fig2,width='content')
