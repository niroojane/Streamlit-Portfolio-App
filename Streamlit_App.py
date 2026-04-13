import streamlit as st
import pandas as pd
import random
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.RiskMetrics import *
from src.Price_Endpoint import *
from src.Stock_Data import get_close
from src.Rebalancing import *
from src.Metrics import *


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



def process_index(index,allocation,dataframe,iterations,stress_factor,mean_shock_vec,var_centile,num_scenarios):
    
    horizon = 1 / 250
    spot = dataframe.iloc[-1]
    theta = 2
    
    range_returns=dataframe.pct_change(fill_method=None)

    distrib_functions = {
        'multivariate_distribution': (iterations, stress_factor,mean_shock_vec),
        'gaussian_copula': (iterations, stress_factor,mean_shock_vec),
        't_copula': (iterations, stress_factor,mean_shock_vec),
        'gumbel_copula': (iterations, theta,np.diag(stress_factor),mean_shock_vec),
        'monte_carlo': (spot, horizon, iterations, stress_factor,mean_shock_vec)
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


uploaded_file = st.file_uploader("Upload an Excel file with time series", type="xlsx")
main_tabs=st.tabs(["Investment Universe","Strategy","Risk Analysis","Market Risk"])

if uploaded_file:
    # Create tabs for Portfolio Analysis and Efficient Frontier
        # Load and prepare the data
    st.session_state.dataframe = pd.read_excel(uploaded_file, index_col=0)
    st.session_state.returns_to_use=st.session_state.dataframe.pct_change(fill_method=None)
    tickers=st.session_state.dataframe.columns
    
    with main_tabs[0]:   
        
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
               
                st.subheader("Core Strategy")
    
                selected_strategy = st.selectbox("Strategy:", options_strat, index=0)
                benchmark_tracking_error = st.selectbox("Benchmark:", list(allocation_dataframe.index), index=0)
                selected_frequency = st.selectbox("Rebalancing Frequency:", rebalancing_frequency, index=0)
                window_vol = st.number_input("Sliding Window Size:", min_value=1, value=252, step=1)
                            
                st.subheader("Overlay")
                drop_down_list_strat=list(dico_strategies.keys())
    
                column_config = {
                    'Strategy': st.column_config.SelectboxColumn(
                        options=drop_down_list_strat
                    ),
                    'Limit': st.column_config.NumberColumn()  # optional but recommended
                }
    
                data_overlay = pd.DataFrame({
                    'Strategy': [None],
                    'Limit': [None]
                })            
                
                # Create the editable data editor with dropdown
                overlay_dataframe = st.data_editor(
                data_overlay,
                column_config=column_config,
                num_rows="dynamic",  # Allow rows to be added dynamically
                )
                
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

                    strategy_limits = overlay_dataframe.set_index("Strategy")["Limit"].to_dict()

                    strategy_key = dico_strategies[selected_strategy]
                    tasks = [(returns_to_use.loc[dates_end[i]:dates_end[i+1]],dates_end[i], dates_end[i+1],strategy_key) for i in range(len(dates_end)-1)]
                    
                    overlays_tasks = [
                        (
                            returns_to_use.loc[dates_end[i]:dates_end[i+1]],
                            dates_end[i],
                            dates_end[i+1],
                            dico_strategies[key]
                        )
                        for i in range(len(dates_end)-1)
                        for key in strategy_limits if pd.notna(key) and key in dico_strategies
                    ]
                    
                    all_tasks = tasks + overlays_tasks
                    
                    results = {}
                    
                    def worker(subset,start, end,strategy_key):
                        if subset.empty or len(subset) < 2:
                            return None
                        try:
                            risk = RiskAnalysis(subset)
                            if constraints:
                                opt = risk.optimize(objective=strategy_key, constraints=constraints)
                            else:
                                opt = risk.optimize(objective=strategy_key)
                            return subset.index[-1], np.round(opt, 6),strategy_key
                        except Exception:
                            return None
    
                    with ThreadPoolExecutor(max_workers=8) as executor:
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
                    
                    
                    rolling_optimization=pd.DataFrame(results[strategy_key], index=dataframe.columns).T.sort_index()
                    total_overlay = pd.DataFrame(0, index=rolling_optimization.index, columns=rolling_optimization.columns)
                    core_weights = 1
                    core_strat = rolling_optimization.copy()
                                        
                    for strat_name, limit in strategy_limits.items():
                    
                        if strat_name not in dico_strategies:
                            continue
                    
                        strat_key_overlay = dico_strategies[strat_name]
                    
                        if strat_key_overlay not in results:
                            continue  # skip if failed
                    
                        overlay_df = (
                            pd.DataFrame(results[strat_key_overlay], index=dataframe.columns).T.sort_index()
                            * limit
                        )
                    
                        total_overlay = total_overlay.add(overlay_df, fill_value=0)
                        core_weights=core_weights-limit
                    
                    rolling_optimization = core_strat * core_weights + total_overlay
                             
                    
                    if not rolling_optimization.empty:
                        first_row = pd.Series(1 / len(dataframe.columns), index=dataframe.columns, name=dates_end[0])
                        rolling_optimization = pd.concat([pd.DataFrame([first_row]), rolling_optimization])
                        core_strat= pd.concat([pd.DataFrame([first_row]), core_strat])
                        total_overlay= pd.concat([pd.DataFrame([first_row]), total_overlay/(1-core_weights)])
    
            
                    model = pd.DataFrame(rolling_optimization.iloc[-2])
                    model.columns = ["Model"]
                    alloc_df = st.session_state.allocation_dataframe.copy()
                
                    if "Model" in alloc_df.index:
                        alloc_df.loc["Model"] = model.T
                    else:
                        alloc_df = pd.concat([alloc_df, model.T], axis=0)
                
                    quantities = rebalanced_dynamic_quantities(dataframe, rolling_optimization)
                    quantities_core = rebalanced_dynamic_quantities(dataframe, core_strat)
                    quantities_overlay = rebalanced_dynamic_quantities(dataframe, total_overlay)
        
                    performance_fund = pd.DataFrame({'Fund': (quantities * dataframe).sum(axis=1),
                                                     'Core':(quantities_core * dataframe).sum(axis=1),
                                                     'Overlay':(quantities_overlay * dataframe).sum(axis=1)})
                    
                    if '^GSPC' in range_prices.columns:
                        performance_fund['S&P'] = range_prices['^GSPC']
                    
                    performance_pct = performance_fund.pct_change(fill_method=None)
                    
                    cumulative = (1 + performance_pct).cumprod() * 100
                    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
                
                    date_drawdown = drawdown.idxmin().dt.date
                    max_drawdown = drawdown.min()
                
                    metrics=pd.DataFrame()
                    metrics['Returns']=(performance_fund.iloc[-2]/performance_fund.iloc[0]).round(4)
                    metrics['Volatility']=(performance_pct.std()*np.sqrt(252)).round(4)
                    metrics['Sharpe Ratio']=((1+metrics['Returns'])**(1/len(set(returns_to_use.index.year)))/metrics['Volatility']).round(4)
                    metrics['Drawdown']=(max_drawdown).round(4)
                    metrics['Date Drawdown']=date_drawdown
                    excess_returns_to_btc = performance_pct.loc[:, performance_pct.columns != 'S&P'].sub(
                        performance_pct['S&P'], axis=0
                    )
                    metrics['Tracking Error to S&P']=((excess_returns_to_btc).std()*np.sqrt(252)).round(4)
                    
                    excess_returns_to_core = performance_pct.loc[:, performance_pct.columns != 'Core'].sub(
                        performance_pct['Core'], axis=0
                    )
                    metrics['Tracking Error to Core']=((excess_returns_to_core).std()*np.sqrt(252)).round(4)
                    metrics=metrics.fillna(0).T
                    indicators=metrics
    
                    cumulative_performance = performance_pct.loc[mask]
                    cumulative_performance.iloc[0] = 0
                    cumulative_results = (1 + cumulative_performance).cumprod() * 100
    
                    portfolio_returns = rebalanced_time_series(range_prices, alloc_df, frequency=selected_frequency)
                    cumulative_results = pd.concat([cumulative_results, portfolio_returns], axis=1)
                    drawdown = (cumulative_results - cumulative_results.cummax()) / cumulative_results.cummax()
                    rolling_vol_ptf = cumulative_results.pct_change().rolling(window_vol).std() * np.sqrt(260)
            
                    st.session_state.results = {
                        "rolling_optimization": rolling_optimization,
                        "core_strat":core_strat,
                        "total_overlay":total_overlay,
                        "alloc_df": alloc_df,
                        "quantities": quantities,
                        "quantities_core":quantities_core,
                        "quantities_overlay":quantities_overlay,
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
                    
                    frontier_indicators, fig4 = get_frontier(range_returns, res['alloc_df'],constraints)
            
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
                    
                    col1,col2,col3=st.columns(3)
                    with col1:
                        st.subheader("Strategy Matrix")
                        st.dataframe(res["rolling_optimization"],width='stretch')
                    with col2:
    
                        st.subheader("Core Matrix")
                        st.dataframe(res["core_strat"],width='stretch')
                    with col3:
                        st.subheader("Overlay Matrix")
                        st.dataframe(res["total_overlay"],width='stretch')
    
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
                
                vol_composition_tabs=st.tabs(['Risk Decomposition','Risk Trajectory','Tracking Error Trajectory'])
                with vol_composition_tabs[0]:
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
                    
                with vol_composition_tabs[1]:
        
                    dataframe = st.session_state.dataframe
                    returns_to_use = st.session_state.returns_to_use
                    res=st.session_state.results
                    allocation_dataframe=res["alloc_df"]
                    
                    quantities=res['quantities']
                    quantities_core=res['quantities_core']
                    quantities_overlay=res['quantities_overlay']
    
            
                    max_value = dataframe.index.max().strftime('%Y-%m-%d')
                    min_value = dataframe.index.min().strftime('%Y-%m-%d')
                    max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
                    min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')  
                    value=(min_value,max_value)
            
                    Model_trajectory = st.slider(
                    'Date:',
                    min_value=min_value,
                    max_value=max_value,
                    value=value,key='risk_path_tab')
                
                    selmin, selmax = Model_trajectory
                    selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
                    selmaxd = selmax.strftime('%Y-%m-%d')
                    
                    mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
                    
                    range_prices=dataframe.loc[mask].copy()
                    range_returns=returns_to_use.loc[mask].copy()
                    
                    if "results_vol" not in st.session_state:
                        st.session_state.results_vol=None
                                        
                    series_dict={}
        
                    for key in allocation_dataframe.index:
                        
                        rebalanced_series=rebalanced_portfolio(range_prices,allocation_dataframe.loc[key])
                        rebalanced_series_weights=rebalanced_series.apply(lambda x: x/rebalanced_series.sum(axis=1))
                        buy_and_hold_series=buy_and_hold(range_prices,allocation_dataframe.loc[key])
                        buy_and_hold_series_weights=buy_and_hold_series.apply(lambda x: x/buy_and_hold_series.sum(axis=1))
                        series_dict['Rebalanced '+key]=rebalanced_series_weights
                        series_dict['Buy and Hold '+key]=buy_and_hold_series_weights
        
                    
                    if not quantities.empty:
                        portfolio=quantities.loc[range_prices.index]*range_prices
                        model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
                        series_dict['Fund']=model_weights
    
                    if not quantities_overlay.empty:
                        portfolio=quantities_overlay.loc[range_prices.index]*range_prices
                        model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
                        series_dict['Overlay']=model_weights                
                        
                    if not quantities_core.empty:
                        portfolio=quantities_core.loc[range_prices.index]*range_prices
                        model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
                        series_dict['Core']=model_weights         
                        
                    
                    options_vol=list(series_dict.keys())
                    selected_fund_to_decompose=st.selectbox("Fund:",options=options_vol,index=len(options_vol)-1,key='selected_fund_risk_decomposition')
                    window_risk=st.number_input("Window Vol:", min_value=7, value=252, step=1)
                    ex_ante_vol_button=st.button("Get Risk History")
                    ex_ante_vol_status=st.empty()
        
                    if ex_ante_vol_button:
                        st.session_state.results_vol=None
        
                        with st.spinner("Computing Ex Ante Vol...",show_time=True):
                                                
                            tasks=[(key,series_dict[key],range_returns,window_risk) for key in series_dict]
        
                            results_dict = {}
                            
                            with ThreadPoolExecutor(max_workers=8) as executor:
                                futures = {
                                    executor.submit(get_ex_ante_vol, weights, returns, window): name
                                    for name, weights, returns, window in tasks
                                }
                            
                                for future in as_completed(futures):
                                    name = futures[future]
                                    results_dict[name] = future.result()
                    
                            results_vol=pd.concat(results_dict.values(), axis=1)
                            results_vol.columns=results_dict.keys()
        
                            st.session_state.results_vol= results_vol
                            
                            ex_ante_vol_status.success('Done!')
            
                    if st.session_state.results_vol is not None:
                        series_weights=series_dict[selected_fund_to_decompose]
                        mask = (series_weights.index >= selmind) & (series_weights.index <= selmaxd)
                        results_vol=st.session_state.results_vol
        
                        contribution_to_vol=get_ex_ante_vol_contribution(series_weights,range_returns.loc[series_weights.index],window_risk)
                        correlation_contrib=get_correlation_contribution(series_weights,range_returns.loc[series_weights.index],window_risk)
                        idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights,range_returns.loc[series_weights.index],window_risk)
        
                        col1, col2 = st.columns([1, 1])
            
                        with col1:
                            
                            mask = (results_vol.index >= selmind) & (results_vol.index <= selmaxd)
            
                            fig = px.line(results_vol.loc[mask], title='Ex Ante Volatility', width=800, height=400, render_mode = 'svg')
                            fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                            fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Historical Portfolio","Fund"])
                            fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
                            st.plotly_chart(fig,width='content')
                
                            fig4 = px.line(idiosyncratic_contrib, title='Idiosyncratic Contribution', width=800, height=400, render_mode = 'svg')
                            fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                            fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
                            fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Idiosyncratic Vol"])
                            st.plotly_chart(fig4,width='content')
                            
                        with col2:
                            
                            fig2 = px.line(contribution_to_vol, title='Volatility Contribution', width=800, height=400, render_mode = 'svg')
                            fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                            fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                            fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Vol"])
                            st.plotly_chart(fig2,width='content')
                
                            
                            fig3 = px.line(correlation_contrib, title='Correlation Contribution', width=800, height=400, render_mode = 'svg')
                            fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                            fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Correlation"])
                            fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
                            st.plotly_chart(fig3,width='content')
                    else:
                        st.info('Load Ex Ante Data')
                        
                with vol_composition_tabs[2]:
                    
                    dataframe = st.session_state.dataframe
                    returns_to_use = st.session_state.returns_to_use
                    res=st.session_state.results
                    allocation_dataframe=res["alloc_df"]
                    
                    quantities=res['quantities']
                    quantities=res['quantities']
                    quantities_core=res['quantities_core']
                    quantities_overlay=res['quantities_overlay']
                    
                    max_value = dataframe.index.max().strftime('%Y-%m-%d')
                    min_value = dataframe.index.min().strftime('%Y-%m-%d')
                    max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
                    min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')  
                    value=(min_value,max_value)
            
                    Model_tracking_error_trajectory = st.slider(
                    'Date:',
                    min_value=min_value,
                    max_value=max_value,
                    value=value,key='te_path_tab')
                
                    selmin, selmax = Model_tracking_error_trajectory
                    selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
                    selmaxd = selmax.strftime('%Y-%m-%d')
                    
                    mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
                    
                    range_prices=dataframe.loc[mask].copy()
                    range_returns=returns_to_use.loc[mask].copy()
                    
                    if "results_tracking_error" not in st.session_state:
                        st.session_state.results_tracking_error=None
                                        
                    series_dict={}
        
                    for key in allocation_dataframe.index:
                        
                        rebalanced_series=rebalanced_portfolio(range_prices,allocation_dataframe.loc[key])
                        rebalanced_series_weights=rebalanced_series.apply(lambda x: x/rebalanced_series.sum(axis=1))
                        buy_and_hold_series=buy_and_hold(range_prices,allocation_dataframe.loc[key])
                        buy_and_hold_series_weights=buy_and_hold_series.apply(lambda x: x/buy_and_hold_series.sum(axis=1))
                        series_dict['Rebalanced '+key]=rebalanced_series_weights
                        series_dict['Buy and Hold '+key]=buy_and_hold_series_weights
        
                    
                    if not quantities.empty:
                        portfolio=quantities.loc[range_prices.index]*range_prices
                        model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
                        series_dict['Fund']=model_weights
    
                    if not quantities_overlay.empty:
                        portfolio=quantities_overlay.loc[range_prices.index]*range_prices
                        model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
                        series_dict['Overlay']=model_weights                
                        
                    if not quantities_core.empty:
                        portfolio=quantities_core.loc[range_prices.index]*range_prices
                        model_weights=portfolio.apply(lambda x: x/portfolio.sum(axis=1))
                        series_dict['Core']=model_weights   
                    
                    options_te=list(series_dict.keys())
        
                    selected_fund_to_decompose=st.selectbox("Fund:", options=options_te,index=len(options_te)-1,key='selected_fund_te_decomposition')
                    select_benchmark_te=st.selectbox("Bench:", options=options_te,index=len(options_te)-2,key='selected_bench_risk_decomposition')
                    window_te=st.number_input("Window Tracking Error:", min_value=7, value=252, step=1)
        
                    selected_weights=series_dict[select_benchmark_te]            
            
                    spread_weights={}
            
                    for key in series_dict:
                        spread_weights[key]=(series_dict[key]-selected_weights).fillna(0)
                                            
                    ex_ante_te_button=st.button("Get Tracking Error History")
                    ex_ante_te_status=st.empty()
        
                    if ex_ante_te_button:
        
                        st.session_state.results_tracking_error=None
        
                        with st.spinner("Computing Ex Ante TE...",show_time=True):
                                                                                        
                            tasks=[(key,spread_weights[key],range_returns.loc[spread_weights[key].index],window_te) for key in series_dict]
        
                            results_dict = {}
                            
                            with ThreadPoolExecutor(max_workers=8) as executor:
                                futures = {
                                    executor.submit(get_ex_ante_vol, weights, returns, window): name
                                    for name, weights, returns, window in tasks
                                }
                            
                                for future in as_completed(futures):
                                    name = futures[future]
                                    results_dict[name] = future.result()
                    
                            results_tracking_error=pd.concat(results_dict.values(), axis=1)
                            results_tracking_error.columns=results_dict.keys()
                            
                            st.session_state.results_tracking_error= results_tracking_error
                            
                            ex_ante_te_status.success('Done!')
        
                    if st.session_state.results_tracking_error is not None:
                        series_weights=spread_weights[selected_fund_to_decompose]
                        mask = (series_weights.index >= selmind) & (series_weights.index <= selmaxd)
                        results_tracking_error=st.session_state.results_tracking_error
        
                        contribution_to_vol=get_ex_ante_vol_contribution(series_weights,range_returns.loc[series_weights.index],window_te)
                        correlation_contrib=get_correlation_contribution(series_weights,range_returns.loc[series_weights.index],window_te)
                        idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights,range_returns.loc[series_weights.index],window_te)
         
                        col1, col2 = st.columns([1, 1])
            
                        with col1:
                            
                            mask = (results_tracking_error.index >= selmind) & (results_tracking_error.index <= selmaxd)
            
                            fig = px.line(results_tracking_error.loc[mask], title='Ex Ante Tracking Error', width=800, height=400, render_mode = 'svg')
                            fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                            fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Historical Portfolio","Fund"])
                            fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
                            st.plotly_chart(fig,width='content')
                
                            fig4 = px.line(idiosyncratic_contrib, title='Idiosyncratic Contribution', width=800, height=400, render_mode = 'svg')
                            fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                            fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
                            fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Idiosyncratic Vol"])
                            st.plotly_chart(fig4,width='content')
                            
                        with col2:
                            
                            fig2 = px.line(contribution_to_vol, title='Tracking Error Contribution', width=800, height=400, render_mode = 'svg')
                            fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                            fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                            fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Vol"])
                            st.plotly_chart(fig2,width='content')
                
                            
                            fig3 = px.line(correlation_contrib, title='Correlation Contribution', width=800, height=400, render_mode = 'svg')
                            fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                            fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Correlation"])
                            fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
                            st.plotly_chart(fig3,width='content')
                    else:
                        st.info('Load Tracking Error Data')
        
                
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
                
                stress_factor=st.number_input("Stress Factor:", min_value=1.0, value=1.0, step=0.1)
                mean_factor=st.number_input("Mean Shock Factor:", min_value=0.0, value=1.0, step=0.1)
                iterations=st.number_input("Iterations:", min_value=1, value=10000, step=1)
                num_scenarios=st.number_input("Scenarios:", min_value=1, value=100, step=1)
                var_centile=st.number_input("Centile:", min_value=0.00, value=0.05, step=0.01)
        
                stress_vec=np.linspace(stress_factor,stress_factor,returns_to_use.shape[1])
                stress_matrix = np.diag(stress_vec)
                
                stress_mean=np.linspace(mean_factor,mean_factor,returns_to_use.shape[1])
                
                selected_fund_var=st.selectbox("Fund:", list(allocation_dataframe.index),index=0,key='selected_fund_var')
    
                st.session_state.mean_data=pd.DataFrame(
                    stress_mean,
                    columns=['Mean Shock'],
                    index=dataframe.columns)
    
                st.session_state.corr_data=pd.DataFrame(
                    stress_matrix,
                    columns=dataframe.columns,
                    index=dataframe.columns
                )
                    
                st.subheader("Mean Return Shock")
                
                def sync_mean_data():
                    edited = st.session_state.editable_mean_data
                    df = st.session_state.mean_data
                    for row_idx, col_changes in edited["edited_rows"].items():
                        for col_name, new_val in col_changes.items():
                            df.iloc[row_idx][col_name] = new_val
                            
                    st.session_state.mean_data = df     
                    
                editable_mean_data = st.data_editor(
                    st.session_state.mean_data,
                    num_rows="static",
                    key='editable_mean_data',
                    on_change=sync_mean_data
                )
                
                st.subheader("Correlation and Volatility Shock")
                
                def enforce_symmetry():
                    """Force correlation matrix to be symmetric using the widget's edited value"""
                    edited = st.session_state.corr_editor_widget
                    df = st.session_state.corr_data.copy()
        
                    for row_idx, col_changes in edited["edited_rows"].items():
                        for col_name, new_val in col_changes.items():
                            df.iloc[row_idx][col_name] = new_val
                            
                    sym = set_symmetric(df.to_numpy(), limit=2)
                    
                    st.session_state.corr_data = pd.DataFrame(
                        sym, index=df.index, columns=df.columns
                    )
                    
    
                    
                edited_corr = st.data_editor(
                    st.session_state.corr_data,
                    key="corr_editor_widget",
                    num_rows="static",
                    on_change=enforce_symmetry
                )
    
    
                cov=range_returns.cov()
                stress_diag=np.diag(np.diag(st.session_state.corr_data))
                stressed_cov = stress_diag @ cov @ stress_diag
                stressed_std=np.sqrt(np.diag(stressed_cov))
                vol = stressed_std*np.sqrt(250)
                shocked_means=(range_returns.mean()*editable_mean_data['Mean Shock'])*250
                
                corr_matrix = stressed_cov / np.outer(stressed_std, stressed_std)
                corr_matrix=corr_matrix+np.tril(edited_corr)+np.tril(edited_corr).T
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
    
                col1,col2=st.columns([1,1])
                
                st.subheader("Shocked Correlation")
    
                with col1:
                    
                    fig = px.imshow(original_corr.round(4), title='Original Correlation Matrix',color_continuous_scale='blues', text_auto=True, aspect="auto")
                    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig.update_traces(xgap=2, ygap=2)
                    fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
    
                    st.plotly_chart(fig)
                    # st.dataframe(original_corr)
                with col2:
                    # st.subheader("Shocked Correlation Matrix")
                    fig1 = px.imshow(corr_dataframe.round(4), title='Shocked Correlation Matrix',color_continuous_scale='blues', text_auto=True, aspect="auto")
                    fig1.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig1.update_traces(xgap=2, ygap=2)
                    fig1.update_traces(textfont=dict(family="Arial Narrow", size=15))
    
                    st.plotly_chart(fig1)
                    # st.dataframe(corr_dataframe)
                
                st.subheader("Shocked Means and Volatilities ")
                st.dataframe(expected_data)
                
                col1, col2,_ = st.columns([1,1,8])
                with col1:
                    var_button = st.button("Run Simulation")
                with col2:
                    refresh_assumption = st.button("Reset Shocks")
                            
                var_status = st.empty()
                  
                var_scenarios, cvar_scenarios, fund_results = {}, {}, {}
    
    
                if refresh_assumption:
                    
                    new_mean_df = pd.DataFrame(
                        np.full(returns_to_use.shape[1], mean_factor),
                        columns=['Mean Shock'],
                        index=dataframe.columns
                    )
                
                    new_corr_df = pd.DataFrame(
                        np.diag(np.full(returns_to_use.shape[1], stress_factor)),
                        columns=dataframe.columns,
                        index=dataframe.columns
                    )
                
                    # Reset BOTH the source data AND the widget state
                    st.session_state.mean_data = new_mean_df
                    st.session_state.corr_data = new_corr_df
    
                    st.rerun()
                    
                if "fund_results" not in st.session_state:
                    st.session_state.fund_results = None
                    st.session_state.var_scenarios=None
                    st.session_state.cvar_scenarios=None
                
                if var_button:
                    with st.spinner("Computing VaR...",show_time=True):
                        st.session_state.fund_results=None
                        st.session_state.var_scenarios=None
                        st.session_state.cvar_scenarios=None
        
                        tasks = [
                            (
                                idx,
                                allocation_dataframe,
                                range_prices,
                                iterations,
                                edited_corr.to_numpy(),
                                editable_mean_data['Mean Shock'],
                                var_centile,
                                num_scenarios
                            )
                            for idx in allocation_dataframe.index
                        ]
                        
                        var_scenarios = {}
                        cvar_scenarios = {}
                        fund_results = {}
                        
                        with ThreadPoolExecutor(max_workers=8) as executor:
                            futures = {
                                executor.submit(process_index, *task): task[0]
                                for task in tasks
                            }
                        
                            for future in futures:
                                idx = futures[future]
                                try:
                                    var, cvar, result = future.result()
                                    var_scenarios[idx] = var
                                    cvar_scenarios[idx] = cvar
                                    fund_results[idx] = result
                                except Exception as e:
                                    print(f"Error processing index {idx}: {e}")
                            
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
    
            sub_tabs_market=st.tabs(['Market Risk','Correlation','Market Drivers'])
    
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
                historical_PCA.iloc[0]=0
                
                comparison=portfolio.returns.copy()
                comparison['PCA']=historical_PCA
                comparison.iloc[0]=0
                distances=np.sqrt(np.sum(comparison.apply(lambda y:(y-historical_PCA['PCA'])**2),axis=0)).sort_values()
                pca_similarity=(1+comparison[distances.index[:num_closest_to_pca]]).cumprod()*100
        
    
                fig=px.bar(variance_explained_dataframe,title='Variance Explanation in %')
                fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400) 
                fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
        
                fig2=px.bar(pca_portfolio,title='Eigen Weights')
                fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400) 
                fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                
                fig3=px.line((1+historical_PCA).cumprod()*100,title='Eigen Index', render_mode = 'svg')
                fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
                fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
        
                fig4=px.line(pca_similarity,title='PCA Similarity', render_mode = 'svg')
                fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
                fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
                col1, col2 = st.columns([1, 1])
    
    
                with col1:
                    st.plotly_chart(fig,width='content')
                    st.plotly_chart(fig2,width='content')
    
                with col2:
                    st.plotly_chart(fig4,width='content')
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
                col1, col2 = st.columns([1, 1])
                
                range_prices=dataframe.loc[mask].copy()
                range_returns=returns_to_use.loc[mask].copy()
        
                pca_over_time=first_pca_over_time(returns=range_returns,window=window_corr)
        
                rolling_correlation = range_returns[dropdown_asset1].rolling(window_corr).corr(
                    range_returns[dropdown_asset2]
                ).dropna()
                
                rolling_mean_returns=range_returns.rolling(window_corr).mean().dropna()*252
    
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
    
                fig4=px.line(rolling_mean_returns,title='Mean Return', render_mode = 'svg')
                fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                fig4.update_layout(xaxis_title=None, yaxis_title=None)
                fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in [dropdown_asset1,dropdown_asset2])
                
                with col1:
                    st.plotly_chart(fig,width='content')
                    st.plotly_chart(fig4,width='content')
    
                with col2:
                    st.plotly_chart(fig3,width='content')
                    st.plotly_chart(fig2,width='content')
                    
            with sub_tabs_market[2]:
    
                dataframe = st.session_state.dataframe
                returns_to_use = st.session_state.returns_to_use
                market_tickers=[t for t in tickers if t in dataframe.columns]
                            
                max_value = dataframe.index.max().strftime('%Y-%m-%d')
                min_value = dataframe.index.min().strftime('%Y-%m-%d')
                max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
                min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')  
                value=(min_value,max_value)
        
                
                Model_market_driver = st.slider(
                'Date:',
                min_value=min_value,
                max_value=max_value,
                value=value,key='market_driver_tab')
            
                selmin, selmax = Model_market_driver
                selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
                selmaxd = selmax.strftime('%Y-%m-%d')
                
                mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
    
                range_prices=dataframe.loc[mask].copy()
                range_returns=returns_to_use.loc[mask].copy()
            
                rebalancing_frequency_marker=st.selectbox("Rebalacing Frequency:",options=['Monthly','Quarterly','Yearly'],index=1,key='market_frequency_eigen_cov_matrix')
                selected_pca_market=st.selectbox("PCA:",options=['PC1','PC2','PC3'])
                window_vol_market=st.number_input("Window Vol:", min_value=7, value=252, step=1,key='window_vol_market')
                market_factors_button=st.button("Get Market Drivers")
                market_factors_status=st.empty()    
                
                if market_factors_button:
                    
                    st.session_state.eigen_weights=None
                    # st.session_state.quantities_eigen=None
                    # st.session_state.market_pnl=None
                    with st.spinner("Computing Market Drivers...",show_time=True):
                        
                        dates=get_rebalancing_dates(returns_to_use,frequency=rebalancing_frequency_marker)
                        tasks = [(returns_to_use.loc[dates[i]:dates[i+1]],dates[i], dates[i+1]) for i in range(len(dates)-1)]
                        # Run with threads
                        results = {}
                        def worker(subset,start, end):
                
                            if subset.empty or len(subset) < 2:
                                return None
                            try:
                                risk = RiskAnalysis(subset)
                                eigval,eigvec,portfolio_components=risk.pca(num_components=5)
                                weights=np.real(portfolio_components[selected_pca_market].to_numpy())
                                
                                return subset.index[-1], np.round(weights, 6)
                            except Exception:
                                return None
                
                        with ThreadPoolExecutor(max_workers=8) as executor:
                            futures = {executor.submit(worker,subset, start, end): (subset,start, end) for subset,start, end in tasks}
                            for future in as_completed(futures):
                                out = future.result()
                                if out is not None:
                                    date_key, weights = out
                                    results[date_key] = weights
                                    
                        if not results:
                            print("⚠️ No valid Eigen values computed.")
                            
                        weights=pd.DataFrame(results).T
    
                        st.session_state.eigen_weights=weights
                        market_factors_status.success('Done!')
                        
                if ('eigen_weights' in st.session_state and st.session_state.eigen_weights is not None):   
                    
                    weights=st.session_state.eigen_weights
                    
                    mask = (weights.index >= selmind) & (weights.index <= selmaxd)
    
                    quantities_eigen=rebalanced_dynamic_quantities(range_prices,weights.loc[mask])
                    
                    market_portfolio=(quantities_eigen*range_prices)
                    market_pnl=market_portfolio-rebalanced_book_cost(range_prices,quantities_eigen)
                    market_pnl['Market Index']=market_pnl.sum(axis=1)
                                    
                    weights_series=market_portfolio.copy()
                    weights_series=weights_series.apply(lambda x: x/market_portfolio.sum(axis=1))  
                    
                    market_index=market_portfolio.sum(axis=1).to_frame()
                    market_index=market_index.pct_change(fill_method=None)
                    market_index.columns=['Market Index']
                    
                    vol_contribution=get_ex_ante_vol_contribution(weights_series,range_returns,window=window_vol_market)
                    correlation_contribution=get_correlation_contribution(weights_series,range_returns,window=window_vol_market)
                    idiosyncratic_contribution=get_idiosyncratic_contribution(weights_series,range_returns,window=window_vol_market)
                    col1, col2 = st.columns([1, 1])
                    
                    perf_index_eigen=pd.DataFrame()
                    
    
                    if 'results' in st.session_state and st.session_state.results is not None:                        
                        
                        res=st.session_state.results
                        global_returns=res['cumulative_results'].pct_change(fill_method=None)
                        perf_index_eigen=market_index.pct_change(fill_method=None)
                        perf_index_eigen=pd.concat([market_index,global_returns],axis=1)
                    else:
                        perf_index_eigen=market_index
                    
                    mask = (perf_index_eigen.index >= selmind) & (perf_index_eigen.index <= selmaxd)
    
                    perf_index_eigen=perf_index_eigen.loc[mask]            
                    perf_index_eigen.iloc[0]=0
                    market_results=(1+perf_index_eigen).cumprod()*100
            
                    with col1:
                        fig = px.line(market_results, title='Performance Comparison', width=800, height=400, render_mode = 'svg')
                        fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                        fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Market Index","Fund","Bitcoin","Historical Portfolio"])
                        fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
                        
                        fig2 = px.line(market_pnl, title='Market Drivers', width=800, height=400, render_mode = 'svg')
                        fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                        fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Market Index"])
                        fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                        
                        fig3 = px.line(correlation_contribution, title='Market Correlation', width=800, height=400, render_mode = 'svg')
                        fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                        fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Correlation"])
                        fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
                         
                        st.plotly_chart(fig,width='content')
                        st.plotly_chart(fig2,width='content')
                        st.plotly_chart(fig3,width='content')         
                        
                    with col2:
                
                        fig4 = px.line(vol_contribution, title='Market Volatility', width=800, height=400, render_mode = 'svg')
                        fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                        fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Vol"])
                        fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
                        
                        fig5 = px.line(idiosyncratic_contribution, title='Market Intrinsic Volatility', width=800, height=400, render_mode = 'svg')
                        fig5.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                        fig5.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Total Idiosyncratic Vol"])
                        fig5.update_traces(textfont=dict(family="Arial Narrow", size=15))
                        
                        fig6 = px.line(weights_series, title='Market Weights', width=800, height=400, render_mode = 'svg')
                        fig6.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                        fig6.update_traces(visible="legendonly", selector=lambda t: not t.name in ["BTCUSDT"])
                        fig6.update_traces(textfont=dict(family="Arial Narrow", size=15))
                        
                        st.plotly_chart(fig4,width='content')
                        st.plotly_chart(fig5,width='content')
                        st.plotly_chart(fig6,width='content')
                else:
                    
                    st.info("Load Market Drivers ⬅️")