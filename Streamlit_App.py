# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


import streamlit as st
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
from keys import *

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


    
def load_data(tickers,start=datetime.datetime(2023,1,1),today=None):

    scope_prices=Binance.get_price_threading(tickers,start)
    scope_prices = scope_prices.sort_index()
    scope_prices = scope_prices[~scope_prices.index.duplicated(keep='first')]
    scope_prices.index = pd.to_datetime(scope_prices.index)
    prices = scope_prices.loc[:, scope_prices.columns != 'USDCUSDT']

    returns = np.log(1 + prices.pct_change(fill_method=None))
    returns.index = pd.to_datetime(returns.index)
    valid_cols = returns.columns[returns.isna().sum() < 30]

    returns_to_use = returns[valid_cols].sort_index()
    dataframe = prices[valid_cols].sort_index().dropna()
    dataframe.index = pd.to_datetime(dataframe.index)
    returns_to_use = returns_to_use[~returns_to_use.index.duplicated(keep='first')]

    st.session_state.dataframe = dataframe.ffill()
    st.session_state.returns_to_use = returns_to_use.fillna(0)
    
    
def get_positions():
    
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
    amount=current_positions.loc['Total']['Current Portfolio in USDT']

    st.session_state.current_weights=current_weights
    st.session_state.current_positions=current_positions
    st.session_state.current_quantities=current_quantities
    st.session_state.amount=amount
    st.session_state.holding_tickers=holding_tickers
    st.session_state.condition=condition

def get_pnl(url):
    
    # url='https://github.com/niroojane/Risk-Management/raw/refs/heads/main/Trade%20History%20Reconstructed.xlsx'
    trade_history = read_excel_from_url(url)
    
    if trade_history is None:
        raise FileNotFoundError("Trade history could not be loaded. Execution stopped.")  
        
    trades=Pnl_calculation.get_trade_in_usdt(trade_history)
    book_cost=Pnl_calculation.get_book_cost(trades)
    realized_pnl,profit_and_loss=Pnl_calculation.get_pnl(book_cost,trades)
    book_cost['MANTRAUSDT']=book_cost['OMUSDT']/4
    st.session_state.book_cost=book_cost
    st.session_state.realized_pnl=realized_pnl
    st.session_state.profit_and_loss=profit_and_loss
    trades=trades.set_index('Date(UTC)')
    st.session_state.trades=trades

    get_positions()

def check_connection(url_positions,url_quantities,url_trades):
    
    # url_positions='https://github.com/niroojane/Risk-Management/raw/refs/heads/main/Positions.xlsx'
    # url_quantities='https://github.com/niroojane/Risk-Management/raw/refs/heads/main/Quantities.xlsx'
    
    position = read_excel_from_url(url_positions,index_col=0)
    if position is None:
        raise FileNotFoundError("Positions.xlsx could not be loaded. Execution stopped.")
        print('Positions Not Found in Repository')
        
    quantities_history = read_excel_from_url(url_quantities,index_col=0)
    if quantities_history is None:
        raise FileNotFoundError("Quantities.xlsx could not be loaded. Execution stopped.")
        print('Quantities Not Found in Repository')
    
    trade_history = read_excel_from_url(url_trades,index_col=0)
    
    if trade_history is None:
        raise FileNotFoundError("Trade history could not be loaded. Execution stopped.")  
        print('Trades Not Found in Repository')

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

    st.session_state.quantities_holding=quantities_holding
    st.session_state.positions=positions
    
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
    
main_tabs=st.tabs(["Investment Universe","Strategy","Current Portfolio","Risk Analysis","Market Risk"])
    
Binance = None
Pnl_calculation = None
git = None    

with st.sidebar:
    
    st.title('Account Details')

    st.subheader('P&L URL')
    
    trades_url=st.text_input(label='Trades URL',value='https://github.com/niroojane/Risk-Management/raw/refs/heads/main/Trade%20History%20Reconstructed.xlsx')
    position_url=st.text_input(label='Position URL',value='https://github.com/niroojane/Risk-Management/raw/refs/heads/main/Positions.xlsx')
    quantities_url=st.text_input(label='Quantities URL',value='https://github.com/niroojane/Risk-Management/raw/refs/heads/main/Quantities.xlsx')
    files_status = st.empty() 

    try:
        position = read_excel_from_url(position_url,index_col=0)
        if position is None:
            raise FileNotFoundError("Positions.xlsx could not be loaded. Execution stopped.")
            print('Positions Not Found in Repository')
            
        quantities_history = read_excel_from_url(quantities_url,index_col=0)
        if quantities_history is None:
            raise FileNotFoundError("Quantities.xlsx could not be loaded. Execution stopped.")
            print('Quantities Not Found in Repository')
            
        trade_history = read_excel_from_url(trades_url)
        if trade_history is None:
            raise FileNotFoundError("Trade history could not be loaded. Execution stopped.")  
            print('Trades Not Found in Repository')
        files_status.success('Files Retrieved')

    except Exception as e:
        files_status.error(f"❌ Files were not retrieved: {e}")
    
    
    st.subheader('Binance Keys')
    
    binance_streamlit_api=st.text_input(label='Binance API Key',value=binance_api_key)
    binance_streamlit_secret=st.text_input(label='Binance Secret Key',value=binance_api_secret)
    binance_status = st.empty()     
    
    st.subheader('Github Keys')
    
    token_input=st.text_input(label='Github Token',value=token)
    owner=st.text_input(label='Github Owner',value=repo_owner)
    repo=st.text_input(label='Repository',value=repo_name)
    branch_name=st.text_input(label='Branch',value=branch)
    github_status = st.empty()     

    
    try:
        Binance = BinanceAPI(
            binance_streamlit_api,
            binance_streamlit_secret
        )
        
        Pnl_calculation = PnL(
            binance_streamlit_api,
            binance_streamlit_secret
        )


        binance_status.success('Binance API Connected')
        get_positions()

    except Exception as e:
        binance_status.error(f"❌ Binance API initialization failed: {e}")
        st.stop()

    try:
        git = GitHub(
            token,
            owner,
            repo,
            branch_name
        )

        github_status.success('Github Connected')

    except Exception as e:
        github_status.error(f"❌ GitHub connection failed: {e}")
        st.stop()
    

# Binance=BinanceAPI(binance_api_key,binance_api_secret)
# Pnl_calculation=PnL(binance_api_key,binance_api_secret)
# git=GitHub(token,repo_owner,repo_name,branch)
# get_positions()

with main_tabs[0]:
    
    selected_number = st.slider(
        "Number of Crypto:",
        min_value=1,
        max_value=40,
        value=20,     
        step=1           
    )
    
    # tickers_market_cap=Binance.get_market_cap()

    tickers_market_cap = (
    Binance.get_market_cap()
    .loc[lambda df: ~df['Long name'].str.contains(r'\(bStocks\)', na=False)]
    )
    market_cap_table=tickers_market_cap.iloc[:selected_number].set_index('Ticker')

    tickers=tickers_market_cap['Ticker'].iloc[:selected_number].to_list()
    holding_tickers=st.session_state.holding_tickers
    combined_tickers=sorted(list(set(tickers+holding_tickers)))
    
    selected = st.multiselect("Select Crypto:", combined_tickers,default=combined_tickers)
    
    st.dataframe(market_cap_table)
    
    starting_date= st.date_input("Starting Date", datetime.datetime(2020, 1, 1))
    dt = datetime.datetime.combine(starting_date, datetime.datetime.min.time())

    price_button=st.button(label='Get Prices')
           
    if price_button:
        with st.spinner("Loading market data...",show_time=True):
            load_data(selected,dt.date())
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
        selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
        selmaxd = selmax.strftime('%Y-%m-%d')
        
        # Filter data by selected date range
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
        'Maximum Diversification':'maximum_diversification',
        'Eigen Strategy':'eigenportfolio'}
    
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
            selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
            selmaxd = selmax.strftime('%Y-%m-%d')
            
            # Filter data by selected date range
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
            eigen_portfolio__constraint=portfolio.optimize("eigenportfolio",constraints=constraints)

            optimized_weights = portfolio.optimize(objective="sharpe_ratio")
            minvar_weights = portfolio.optimize(objective="minimum_variance")
            risk_parity_weights = portfolio.optimize(objective="risk_parity")
            max_diversification=portfolio.optimize(objective="maximum_diversification")
            eigen_portfolio=portfolio.optimize("eigenportfolio")

            equal_weights = np.ones(returns_to_use.shape[1]) / returns_to_use.shape[1]
    
            allocation['Optimal Portfolio']=optimized_weights.tolist()
            allocation['Optimal Constrained Portfolio']=optimized_weights_constraint.tolist()
    
            allocation['Minimum Variance Portfolio']=minvar_weights.tolist()
            allocation['Minimum Variance Constrained Portfolio']=minvar_weights_constraint.tolist()
            
            allocation['Maximum Diversification Portfolio']=max_diversification.tolist()
            allocation['Maximum Diversification Constrained Portfolio']=max_diversification_weights_constraint.tolist()
            
            allocation['Risk Parity Portfolio']=risk_parity_weights.tolist()
            allocation['Risk Parity Constrained Portfolio']=risk_parity_weights_constraint.tolist()
            
            allocation['Eigen Portfolio']= eigen_portfolio.tolist()
            allocation['Eigen Portfolio Constrained']= eigen_portfolio__constraint.tolist()
            
            allocation['Equal Weighted']=equal_weights.tolist()
            
            allocation_dataframe = pd.DataFrame(
                    allocation,
                    index=dataframe.columns
                ).T.round(6)

            current_weights=st.session_state.current_weights
            
            if set(current_weights.index).issubset(dataframe.columns):
                allocation_dataframe = allocation_dataframe.combine_first(current_weights.T).fillna(0)
                allocation_dataframe = allocation_dataframe.iloc[::-1]
            
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
                res_status=st.empty()

                st.session_state.run_optimization = True
                st.session_state.results = None  
                

            if st.session_state.run_optimization and st.session_state.results is None:

                with st.spinner("Computing Results...",show_time=True):

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
                        
                    # rolling_optimization.loc[dates_end[0]] = 1 / len(dataframe.columns)
                    # rolling_optimization = rolling_optimization.sort_index()
            
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
                    
                    if 'BTCUSDT' in range_prices.columns:
                        performance_fund['Bitcoin'] = range_prices['BTCUSDT']
                    
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
                    excess_returns_to_btc = performance_pct.loc[:, performance_pct.columns != 'Bitcoin'].sub(
                        performance_pct['Bitcoin'], axis=0
                    )
                    metrics['Tracking Error to Bitcoin']=((excess_returns_to_btc).std()*np.sqrt(252)).round(4)
                    
                    excess_returns_to_core = performance_pct.loc[:, performance_pct.columns != 'Core'].sub(
                        performance_pct['Core'], axis=0
                    )
                    metrics['Tracking Error to Core']=((excess_returns_to_core).std()*np.sqrt(252)).round(4)
                    metrics=metrics.fillna(0).T
                    indicators=metrics
                    # indicators = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Indicators'])
                    
                    cumulative_performance = performance_pct.loc[mask]
                    cumulative_performance.iloc[0] = 0
                    cumulative_results = (1 + cumulative_performance).cumprod() * 100
            
                    portfolio_returns = rebalanced_time_series(range_prices, alloc_df, frequency=selected_frequency)
                    cumulative_results = pd.concat([cumulative_results, portfolio_returns], axis=1)
                    drawdown = (cumulative_results - cumulative_results.cummax()) / cumulative_results.cummax()
                    rolling_vol_ptf = cumulative_results.pct_change(fill_method=None).rolling(window_vol).std() * np.sqrt(260)
            
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
                    
                    res_status.success('Done!')


            if st.session_state.results is not None:
                selmin, selmax = st.session_state['strategy_tab']
                selmind = selmin.strftime('%Y-%m-%d') 
                selmaxd = selmax.strftime('%Y-%m-%d')
                
                res=st.session_state.results
                mask = (res['cumulative_results'].index >= selmind) & (res['cumulative_results'].index <= selmaxd)

                cumulative_performance=res['cumulative_results'].loc[mask].pct_change(fill_method=None)
                cumulative_performance.iloc[0] = 0
                cumulative_results = (1 + cumulative_performance).cumprod() * 100
                
                drawdown = (cumulative_results - cumulative_results.cummax()) / cumulative_results.cummax()
                rolling_vol_ptf = cumulative_results.pct_change(fill_method=None).rolling(window_vol).std() * np.sqrt(260)
                
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

        
with main_tabs[3]: 
    
    if "dataframe" not in st.session_state:
        st.info("Load data first ⬅️")
        
    elif st.session_state.results is None:
        st.info("Compute Optimization first ⬅️")

    else:
        
        sub_tabs_risk=st.tabs(['Risk Analysis','Value at Risk'])
        check_connection(position_url,quantities_url,trades_url)

        with sub_tabs_risk[0]:
            
            risk_decomposition_tab=st.tabs(['Risk Decomposition','Risk Trajectory','Tracking Error Trajectory'])
            
            with risk_decomposition_tab[0]:
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

            with risk_decomposition_tab[1]:
                
                
                dataframe = st.session_state.dataframe
                returns_to_use = st.session_state.returns_to_use
                res=st.session_state.results
                positions=st.session_state.positions

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
                    
                if "current_underlying_returns" not in st.session_state:
                    st.session_state.current_underlying_returns=None
                    
                series_dict={}
    
                for key in allocation_dataframe.index:
                    
                    rebalanced_series=rebalanced_portfolio(range_prices,allocation_dataframe.loc[key])
                    rebalanced_series_weights=rebalanced_series.apply(lambda x: x/rebalanced_series.sum(axis=1))
                    buy_and_hold_series=buy_and_hold(range_prices,allocation_dataframe.loc[key])
                    buy_and_hold_series_weights=buy_and_hold_series.apply(lambda x: x/buy_and_hold_series.sum(axis=1))
                    series_dict['Rebalanced '+key]=rebalanced_series_weights
                    series_dict['Buy and Hold '+key]=buy_and_hold_series_weights
                
                weights_ex_post=positions.copy()
                weights_ex_post=weights_ex_post.drop(columns=['USDTUSDT'])
                weights_ex_post=weights_ex_post.apply(lambda x: x/weights_ex_post['Total'])
                weights_ex_post=weights_ex_post.drop(columns=['Total'])
                weights_ex_post=weights_ex_post.fillna(0.0)
                
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
                    
                mask = (weights_ex_post.index >= selmind) & (weights_ex_post.index <= selmaxd)
                series_dict['Historical Portfolio']=weights_ex_post.loc[mask]
    
                tickers_combined=list(quantities.columns)+list(weights_ex_post.columns)
                tickers_combined=list(set(tickers_combined))
                options_vol=list(series_dict.keys())
                selected_fund_to_decompose=st.selectbox("Fund:",options=options_vol,index=len(options_vol)-1,key='selected_fund_risk_decomposition')
                window_risk=st.number_input("Window Vol:", min_value=7, value=252, step=1)
                ex_ante_vol_button=st.button("Get Risk History")
                ex_ante_vol_status=st.empty()
    
                if ex_ante_vol_button:
                    st.session_state.results_vol=None
                    st.session_state.current_underlying_returns=None
    
                    with st.spinner("Computing Ex Ante Vol...",show_time=True):
                                            
                        start_date=weights_ex_post.index[0].date()
                        
                        current_underlying_prices=Binance.get_price_threading(tickers_combined,start_date)
                        current_underlying_returns=current_underlying_prices.pct_change(fill_method=None)
                        
                        tasks=[(key,series_dict[key],range_returns,window_risk) for key in series_dict if key!='Historical Portfolio']
                        
                        mask = (weights_ex_post.index >= selmind) & (weights_ex_post.index <= selmaxd)
    
                        tasks.append(('Historical Portfolio',weights_ex_post.loc[mask],current_underlying_returns.loc[weights_ex_post.index].loc[mask],window_risk))
                                
                        results_dict = {}

                        for name,weight,returns,window in tasks:
                            results_dict[name]=get_ex_ante_vol(weight, returns, window_risk)
                                            
                        results_vol=pd.concat(results_dict.values(), axis=1)
                        results_vol.columns=results_dict.keys()
                        
                        st.session_state.results_vol= results_vol
                        st.session_state.current_underlying_returns=current_underlying_returns
                        
                        ex_ante_vol_status.success('Done!')
        
                if st.session_state.results_vol is not None:
                    series_weights=series_dict[selected_fund_to_decompose]
                    mask = (series_weights.index >= selmind) & (series_weights.index <= selmaxd)
                    results_vol=st.session_state.results_vol
                    current_underlying_returns=st.session_state.current_underlying_returns
    
                    if selected_fund_to_decompose!='Historical Portfolio':
                        
                        contribution_to_vol=get_ex_ante_vol_contribution(series_weights,range_returns.loc[series_weights.index],window_risk)
                        correlation_contrib=get_correlation_contribution(series_weights,range_returns.loc[series_weights.index],window_risk)
                        idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights,range_returns.loc[series_weights.index],window_risk)
        
                    else:
                        
                        contribution_to_vol=get_ex_ante_vol_contribution(series_weights.loc[mask],current_underlying_returns.loc[series_weights.index].loc[mask],window_risk)
                        correlation_contrib=get_correlation_contribution(series_weights.loc[mask],current_underlying_returns.loc[series_weights.index].loc[mask],window_risk)
                        idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights.loc[mask],current_underlying_returns.loc[series_weights.index].loc[mask],window_risk)
                    
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
                
            with risk_decomposition_tab[2]:
                
                dataframe = st.session_state.dataframe
                returns_to_use = st.session_state.returns_to_use
                res=st.session_state.results
                allocation_dataframe=res["alloc_df"]
                quantities=res['quantities']
                positions=st.session_state.positions
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
                    
                if "current_underlying_returns_te" not in st.session_state:
                    st.session_state.current_underlying_returns_te=None
                    
                series_dict={}
    
                for key in allocation_dataframe.index:
                    
                    rebalanced_series=rebalanced_portfolio(range_prices,allocation_dataframe.loc[key])
                    rebalanced_series_weights=rebalanced_series.apply(lambda x: x/rebalanced_series.sum(axis=1))
                    buy_and_hold_series=buy_and_hold(range_prices,allocation_dataframe.loc[key])
                    buy_and_hold_series_weights=buy_and_hold_series.apply(lambda x: x/buy_and_hold_series.sum(axis=1))
                    series_dict['Rebalanced '+key]=rebalanced_series_weights
                    series_dict['Buy and Hold '+key]=buy_and_hold_series_weights
                
                weights_ex_post=positions.copy()
                weights_ex_post=weights_ex_post.drop(columns=['USDTUSDT'])
                weights_ex_post=weights_ex_post.apply(lambda x: x/weights_ex_post['Total'])
                weights_ex_post=weights_ex_post.drop(columns=['Total'])
                weights_ex_post=weights_ex_post.fillna(0.0)
                
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
                    
                mask = (weights_ex_post.index >= selmind) & (weights_ex_post.index <= selmaxd)
                series_dict['Historical Portfolio']=weights_ex_post.loc[mask]
    
                tickers_combined=list(quantities.columns)+list(weights_ex_post.columns)
                tickers_combined=list(set(tickers_combined))
                options_te=list(series_dict.keys())
    
                selected_fund_to_decompose=st.selectbox("Fund:", options=options_te,index=len(options_te)-1,key='selected_fund_te_decomposition')
                select_benchmark_te=st.selectbox("Bench:", options=options_te,index=len(options_te)-2,key='selected_bench_risk_decomposition')
                window_te=st.number_input("Window Tracking Error:", min_value=7, value=252, step=1)
    
                selected_weights=series_dict[select_benchmark_te]
                not_in_bench=list(set(weights_ex_post.columns)-set(selected_weights.columns))
                not_in_fund=list(set(selected_weights.columns)-set(weights_ex_post.columns))
                
                selected_weights = selected_weights.copy()
                weights_ex_post = weights_ex_post.copy()
                
                weights_ex_post[not_in_fund] = 0
                selected_weights[not_in_bench] = 0
        
                spread_weights={}
        
                for key in series_dict:
                    spread_weights[key]=(series_dict[key]-selected_weights).fillna(0)
                
                mask = (weights_ex_post.index >= selmind) & (weights_ex_post.index <= selmaxd)
    
                spread_ex_post=(weights_ex_post-selected_weights).loc[weights_ex_post.index].loc[mask].fillna(0)
                spread_weights['Historical Portfolio']=spread_ex_post
                            
                ex_ante_te_button=st.button("Get Tracking Error History")
                ex_ante_te_status=st.empty()
    
                if ex_ante_te_button:
                    st.session_state.results_tracking_error=None
                    st.session_state.current_underlying_returns_te=None
    
                    with st.spinner("Computing Ex Ante TE...",show_time=True):
                                            
                        start_date=weights_ex_post.index[0].date()
                        
                        current_underlying_prices=Binance.get_price_threading(tickers_combined,start_date)
                        current_underlying_returns=current_underlying_prices.pct_change(fill_method=None)
                        
                        tasks=[(key,spread_weights[key],range_returns.loc[spread_weights[key].index],window_te) for key in series_dict if key!='Historical Portfolio']
                        
                        mask = (current_underlying_returns.index >= selmind) & (current_underlying_returns.index <= selmaxd)
    
                        tasks.append(('Historical Portfolio',spread_ex_post,current_underlying_returns.loc[spread_ex_post.index],window_te))
                                
                        results_dict = {}
                        
                        for name,weight,returns,window in tasks:
            
                            results_dict[name]=get_ex_ante_vol(weight, returns, window_te)
                                            
                        results_tracking_error=pd.concat(results_dict.values(), axis=1)
                        results_tracking_error.columns=results_dict.keys()
                        
                        st.session_state.results_tracking_error= results_tracking_error
                        st.session_state.current_underlying_returns_te=current_underlying_returns
                        
                        ex_ante_te_status.success('Done!')
    
                if st.session_state.results_tracking_error is not None:
                    series_weights=spread_weights[selected_fund_to_decompose]
                    mask = (series_weights.index >= selmind) & (series_weights.index <= selmaxd)
                    results_tracking_error=st.session_state.results_tracking_error
                    current_underlying_returns=st.session_state.current_underlying_returns_te
        
                    if selected_fund_to_decompose!='Historical Portfolio':
        
                        contribution_to_vol=get_ex_ante_vol_contribution(series_weights,range_returns.loc[series_weights.index],window_te)
                        correlation_contrib=get_correlation_contribution(series_weights,range_returns.loc[series_weights.index],window_te)
                        idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights,range_returns.loc[series_weights.index],window_te)
        
                    else:
                        
                        contribution_to_vol=get_ex_ante_vol_contribution(series_weights.loc[mask],current_underlying_returns.loc[series_weights.index].loc[mask],window_te)
                        correlation_contrib=get_correlation_contribution(series_weights.loc[mask],current_underlying_returns.loc[series_weights.index].loc[mask],window_te)
                        idiosyncratic_contrib=get_idiosyncratic_contribution(series_weights.loc[mask],current_underlying_returns.loc[series_weights.index].loc[mask],window_te)
                    
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
            
            var_decomposition_tab=st.tabs(['Value at Risk','Value at Risk Trajectory'])
            
            with var_decomposition_tab[0]:
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
            
                selmin, selmax = Model4
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
                        
                        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
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
                    
                    st.subheader('Value at Risk')
                    st.dataframe(var_dataframe,width='stretch')
                    st.subheader('Conditional Value at Risk')
                    st.dataframe(cvar_dataframe,width='stretch')
                    st.subheader('Results')
                    st.dataframe(fund_results_dataframe,width='stretch')
                    
            with var_decomposition_tab[1]:
                
                dataframe = st.session_state.dataframe
                returns_to_use = st.session_state.returns_to_use
                res=st.session_state.results
                allocation_dataframe=res["alloc_df"]
                    
                max_value = dataframe.index.max().strftime('%Y-%m-%d')
                min_value = dataframe.index.min().strftime('%Y-%m-%d')
                max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
                min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')  
                value=(min_value,max_value)
        
                Model_var_history = st.slider(
                'Date:',
                min_value=min_value,
                max_value=max_value,
                value=value,key='var_history_tab')
            
                selmin, selmax = Model_var_history
                selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
                selmaxd = selmax.strftime('%Y-%m-%d')
                
                mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
                
                range_prices=dataframe.loc[mask].copy()
                range_returns=returns_to_use.loc[mask].copy()
                
                if "results_var_history" not in st.session_state:
                    st.session_state.results_var_history=None
                    
                if "results_cvar_history" not in st.session_state:
                    st.session_state.results_cvar_history=None 
                series_dict={}
    
                for key in allocation_dataframe.index:
                    
                    rebalanced_series=rebalanced_portfolio(range_prices,allocation_dataframe.loc[key])
                    rebalanced_series_weights=rebalanced_series.apply(lambda x: x/rebalanced_series.sum(axis=1))
                    buy_and_hold_series=buy_and_hold(range_prices,allocation_dataframe.loc[key])
                    buy_and_hold_series_weights=buy_and_hold_series.apply(lambda x: x/buy_and_hold_series.sum(axis=1))
                    series_dict['Rebalanced '+key]=rebalanced_series_weights
                    series_dict['Buy and Hold '+key]=buy_and_hold_series_weights
                
                weights_ex_post=positions.copy()
                weights_ex_post=weights_ex_post.drop(columns=['USDTUSDT'])
                weights_ex_post=weights_ex_post.apply(lambda x: x/weights_ex_post['Total'])
                weights_ex_post=weights_ex_post.drop(columns=['Total'])
                weights_ex_post=weights_ex_post.fillna(0.0)
                
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
                    
                mask = (weights_ex_post.index >= selmind) & (weights_ex_post.index <= selmaxd)


                var_function_names={'Multivariate':'multivariate_distribution',
                   'Gaussian Copula':'gaussian_copula',
                   'Monte Carlo':'monte_carlo',
                   'Gumbel Copula':'gumbel_copula',
                   'T-Copula':'t_copula'}
                
                func_name=st.selectbox("Method:",options=list(var_function_names.keys()),index=0,key='functions')
                if func_name=='Gumbel Copula':
                    series_dict['Historical Portfolio']=weights_ex_post.loc[mask]
    
                tickers_combined=list(quantities.columns)+list(weights_ex_post.columns)
                tickers_combined=list(set(tickers_combined))
                options_vol=list(series_dict.keys())

                selected_fund_to_decompose_var_history=st.selectbox("Fund:",options=options_vol,index=len(options_vol)-1,key='selected_fund_to_decompose_var_history')

                stress_factor_history=st.number_input("Stress Factor:", min_value=1.0, value=1.0, step=0.1,key='stress_factor_history')
                mean_factor_history=st.number_input("Mean Shock Factor:", min_value=0.0, value=1.0, step=0.1,key='mean_factor_history')
                iterations_history=st.number_input("Iterations:", min_value=1, value=10000, step=1,key='iterations_history')
                var_centile_history=st.number_input("Centile:", min_value=0.00, value=0.05, step=0.01,key='var_centile_history')
                window_var_history=st.number_input("Window:", min_value=30, value=252, step=1,key='window_var_history')


                
                var_hist_button=st.button("Get Value At Risk History")
                var_hist_status=st.empty()
                spot=range_prices.iloc[-1]
                horizon=1/250
                theta=2
                
                distrib_functions = {
                'multivariate_distribution': (iterations_history, stress_factor_history,mean_factor_history),
                'gaussian_copula': (iterations_history, stress_factor_history,mean_factor_history),
                't_copula': (iterations_history, stress_factor_history,mean_factor_history),
                'gumbel_copula': (iterations_history, theta,stress_factor_history,mean_factor_history),
                'monte_carlo': (spot, horizon, iterations_history,stress_factor_history,mean_factor_history)}

                method=var_function_names[func_name]
                args=distrib_functions[method]
                
                if var_hist_button:
                    st.session_state.results_var_history=None
                    st.session_state.results_cvar_history=None
    
                    with st.spinner("Computing VaR History...",show_time=True):
                        
                        results_dict_var = {}
                        results_dict_cvar = {}
                        
                        start_date=weights_ex_post.index[0].date()
                        
                        current_underlying_prices=Binance.get_price_threading(tickers_combined,start_date)
                        current_underlying_returns=current_underlying_prices.pct_change(fill_method=None)
                        
                        tasks=[(key,method,args,range_returns,series_dict[key],window_var_history,var_centile_history) for key in series_dict]
                        
                        common=weights_ex_post.columns.intersection(current_underlying_returns.columns)
                        common_index=weights_ex_post.index.intersection(current_underlying_returns.index)
                        
                        mask = (common_index >= selmind) & (common_index<= selmaxd)
                        if method in ['gumbel_copula']:
                            tasks.append(
                             (
                              'Historical Portfolio',
                              method,
                              args,
                              current_underlying_returns.loc[common_index,common].loc[mask],
                              weights_ex_post.loc[common_index,common].loc[mask],
                              window_var_history,
                              var_centile_history
                             )
                            )
                        for name,func,arg,returns,weight,window,centile in tasks:
                
                            common_col=returns.columns.intersection(weight.columns)
                            common_index=returns.index.intersection(weight.index)
                            
                            var_data,cvar_data=get_var_contribution(func,arg,returns.loc[common_index,common_col],weight.loc[common_index,common_col],window_var_history,var_centile_history)
                            results_dict_var[name]=var_data['Portfolio']
                            results_dict_cvar[name] =cvar_data['Portfolio']
                                            
                        result_var = pd.concat(results_dict_var.values(),axis=1)
                        result_cvar = pd.concat(results_dict_cvar.values(), axis=1)
                        
                        result_var.columns=list(results_dict_var.keys())
                        result_cvar.columns=list(results_dict_var.keys())

                        st.session_state.results_var_history= result_var
                        st.session_state.results_cvar_history= result_cvar
                        
                        st.session_state.current_underlying_returns=current_underlying_returns
                        
                        var_hist_status.success('Done!')
            
            if st.session_state.results_var_history is not None:
                    
                series_weights=series_dict[selected_fund_to_decompose_var_history]
                
                mask = (series_weights.index >= selmind) & (series_weights.index <= selmaxd)
                
                results_var=st.session_state.results_var_history
                results_cvar=st.session_state.results_cvar_history
                current_underlying_returns=st.session_state.current_underlying_returns

                if selected_fund_to_decompose_var_history=='Historical Portfolio' and method=='gumbel_copula':
                    range_returns=current_underlying_returns
                else:
                    range_returns=range_returns
                common=series_weights.columns.intersection(range_returns.columns)
                common_index=series_weights.index.intersection(range_returns.index)
                
                var,cvar=get_var_contribution(method,args,
                                                range_returns.loc[common_index,common].loc[mask],
                                                series_weights.loc[common_index,common].loc[mask],
                                                window_var_history,
                                                var_centile_history
                                             )
                                    
                col1, col2 = st.columns([1, 1])
                with col1:
                    
                    mask = (results_var.index >= selmind) & (results_var.index <= selmaxd)
                    
                    fig = px.line(results_var.loc[mask], title='Portfolios Value At Risk', width=800, height=400, render_mode = 'svg')
                    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Historical Portfolio","Fund"])
                    fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    st.plotly_chart(fig,width='content')
    
                    fig4 = px.line(var, title='Value at Risk History', width=800, height=400, render_mode = 'svg')
                    fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig4.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Portfolio"])
                    st.plotly_chart(fig4,width='content')
                    
                with col2:
                    
                    fig2 = px.line(results_cvar, title='Portfolio Expected Shortfall', width=800, height=400, render_mode = 'svg')
                    fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Historical Portfolio","Fund"])
                    st.plotly_chart(fig2,width='content')
                
                    
                    fig3 = px.line(cvar, title='Expected Shortfall History', width=800, height=400, render_mode = 'svg')
                    fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
                    fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ["Portfolio"])
                    fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
                    st.plotly_chart(fig3,width='content')
            else:

                st.info("Load data first ⬅️")


                    
with main_tabs[4]:
    
    if "dataframe" not in st.session_state:
        st.info("Load data first ⬅️")
    else:

        sub_tabs_market=st.tabs(['Market Risk','Correlation','Market Factors'])

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
            distances=np.sqrt(np.sum(comparison.apply(lambda y:(y-historical_PCA['PCA'])**2),axis=0)).sort_values()
            
            pca_similarity=comparison[distances.index[:num_closest_to_pca]]
            pca_similarity.iloc[0]=0
            pca_similarity=(1+pca_similarity).cumprod()*100

    
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
            selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
            selmaxd = selmax.strftime('%Y-%m-%d')

            dropdown_asset1=st.selectbox("Asset 1:",options=range_returns.columns,index=0)

            dropdown_asset2=st.selectbox("Asset 2:",options=range_returns.columns,index=1)
                    
            window_corr=st.number_input("Window Correlation:",min_value=0,value=252)
    
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
            
                    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
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
                
                if 'ex_post_portfolios' in st.session_state and 'results' in st.session_state and st.session_state.results is not None:
                    
                    performance_ex_post=st.session_state.ex_post_portfolios.pct_change(fill_method=None)
                    res=st.session_state.results
                    global_returns=res['cumulative_results'].pct_change(fill_method=None)
                    perf_index_eigen=pd.concat([market_index,performance_ex_post,global_returns],axis=1)   
                    
                elif 'results' in st.session_state and st.session_state.results is not None:                        
                    
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

with main_tabs[2]:
    
    sub_tabs_ex_post=st.tabs(['Positioning','Historical Portfolio','Calendar Return'])
    
    with sub_tabs_ex_post[0]:

        if "dataframe" not in st.session_state:
            st.info("Load data first ⬅️")
            
        if "results" not in st.session_state:
            st.info("Compute Optimization first ⬅️")
            
        col1, col2, _ = st.columns([1, 1, 10])
        
        check_connection(position_url,quantities_url,trades_url)

        with col1:
            
            get_positions_button=st.button("Get Positions",key='position_button')
        
        with col2:
            pnl_button=st.button("Get P&L",key='pnl_button')

        
        st.session_state.current_portfolio=None

        if get_positions_button:
            st.session_state.current_positions=None
            st.session_state.current_weights=None
            
            with st.spinner("Loading Positions...",show_time=True):

                get_positions()
                    
                st.success("Done!")

        if pnl_button:
            
            st.session_state.realized_pnl=None
            st.session_state.book_cost=None
            
            with st.spinner("Loading P&L...",show_time=True):
                
                get_positions() 
                get_pnl(trades_url)
                st.success("Done!")
                
        if st.session_state.current_positions is not None:
            
            if 'results' not in st.session_state:
                current_positions=st.session_state.current_positions
            
            elif st.session_state.results is not None:
                
                current_positions=st.session_state.current_positions
                amount=st.session_state.amount
                condition=st.session_state.condition
                current_quantities=st.session_state.current_quantities  

                res=st.session_state.results
                quantities=res['quantities']
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
                
                st.session_state.current_portfolio=portfolio 

            if st.session_state.current_portfolio is not None:
                    
                to_display=st.session_state.current_portfolio
            else:
                to_display=st.session_state.current_positions            
            
            st.subheader("Current Portfolio")
            
            st.dataframe(to_display,width='stretch')

            if 'book_cost' not in st.session_state:
                st.info("Load P&L first ⬅️")
            
            else:
                
                current_positions=st.session_state.current_positions
                amount=st.session_state.amount
                condition=st.session_state.condition
                current_quantities=st.session_state.current_quantities  
                book_cost=st.session_state.book_cost
                realized_pnl=st.session_state.realized_pnl
                
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
                    
                pnl=pnl.sort_values(by='Weights', ascending=False).round(4)
                st.session_state.pnl=pnl
                
                st.subheader("P&L")
                pnl=st.session_state.pnl
                st.dataframe(pnl,width='stretch')
            
                st.subheader("Trade History")
                trades=st.session_state.trades
                st.dataframe(trades,width='stretch')
        
    with sub_tabs_ex_post[1]:
        
        quantities_holding=st.session_state.quantities_holding
        positions=st.session_state.positions
        st.success("Connected!")
 
        if 'quantities_holding' not in st.session_state:
            st.error("Error with URLs")

        else:
            
            col1, col2, _ = st.columns([1, 1, 7])
            
            with col1:
                historical_value=st.button("Get Historical Portfolio",key='historical_value')

            with col2:
                pnl_button_historical=st.button("Get P&L",key='historical_pnl')

            if pnl_button_historical:
                st.session_state.realized_pnl=None
                st.session_state.book_cost=None
                
                with st.spinner("Loading P&L...",show_time=True):
                    
                    get_positions() 
                    get_pnl(trades_url)
                    st.success("Done!")
            
            if historical_value:
                
                with st.spinner("Loading Portfolio Value...",show_time=True):

                    if 'book_cost' not in st.session_state:
                        get_positions() 
                        get_pnl(trades_url)
                        
                    st.session_state.daily_pnl=None
                    st.session_state.pnl_history=None
    
                    if 'book_cost' not in st.session_state:
                        get_pnl(trades_url)
                        
                    book_cost=st.session_state.book_cost  

                    historical_quantities_tickers=list(quantities_holding.columns)
                    daily_book_cost=book_cost.resample("D").last().dropna().sort_index()
                    book_cost_history=pd.DataFrame()
                    book_cost_history = pd.DataFrame(index=daily_book_cost.index.union(quantities_holding.index)).sort_index() 
                    
                    cols= quantities_holding.columns[quantities_holding.columns!='USDCUSDT']
                    
            
                    for col in cols:
                        book_cost_history[col]=daily_book_cost[col]
                            
                    book_cost_history=book_cost_history.ffill()
                    book_cost_history=book_cost_history.loc[quantities_holding.index] 

                    weights_ex_post=positions.copy()
                    weights_ex_post=weights_ex_post.drop(columns=['USDTUSDT'])
                    weights_ex_post=weights_ex_post.apply(lambda x: x/weights_ex_post['Total'])
                    
                    start_date=weights_ex_post.index[0].date()
                    
                    binance_data=Binance.get_price_threading(historical_quantities_tickers,start_date)
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
                    common_date = weights_ex_post.index.intersection(binance_data_return.index)
                    
                    binance_data2=binance_data_return.loc[list(common_date)].copy().sort_index()
                    weights_ex_post2=weights_ex_post.loc[list(common_date)].copy().sort_index()
                    historical_ptf=pd.DataFrame()
                    
                    common_cols = weights_ex_post2.columns.intersection(binance_data2.columns)
                    
                    historical_ptf = (
                        weights_ex_post2[cols] *
                        binance_data2[cols])
                                            
                    historical_ptf['Historical Portfolio']=historical_ptf.sum(axis=1)   
                    
                    performance_ex_post=historical_ptf['Historical Portfolio'].copy()
                    performance_ex_post=performance_ex_post.to_frame()
                    
                    cumulative_performance=performance_ex_post.copy()
                    cumulative_performance.iloc[0]=0
                    cumulative_results=(1+cumulative_performance).cumprod()*100
                    
                    st.session_state.ex_post_portfolios=cumulative_results
                    st.session_state.daily_pnl=daily_pnl
                    st.session_state.pnl_history=pnl_history
                    
                    st.success("Done!")
    
            if 'book_cost' not in st.session_state:
                st.info("Load P&L")
                
            elif 'daily_pnl' not in st.session_state:
                st.info("Load Historical Portfolio")

            elif st.session_state.daily_pnl is not None:
                    
                ex_post_portfolios=st.session_state.ex_post_portfolios
                
                if 'results' in st.session_state and  st.session_state.results is not None:
                    res=st.session_state.results
                    global_returns=res['cumulative_results']
                        
                else:
                    global_returns=pd.DataFrame()
            
                if not global_returns.empty:
                    ex_post_portfolios=pd.concat([ex_post_portfolios,global_returns],axis=1).sort_index()
                    
                daily_pnl=st.session_state.daily_pnl
                pnl_history=st.session_state.pnl_history
              
                max_value = ex_post_portfolios.index.max().strftime('%Y-%m-%d')
                min_value = ex_post_portfolios.index.min().strftime('%Y-%m-%d')
                max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
                min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')
                start_value=st.session_state.ex_post_portfolios.index.min().strftime('%Y-%m-%d')
                start_value=datetime.datetime.strptime(start_value, '%Y-%m-%d')

                value=(start_value,max_value)
        
                Model7 = st.slider(
                'Date:',
                min_value=min_value,
                max_value=max_value,
                value=value,key='ex_post_tab')
            
                selmin, selmax = Model7
                selmind = selmin.strftime('%Y-%m-%d')
                selmaxd = selmax.strftime('%Y-%m-%d')
                
                daily_pnl.index = pd.to_datetime(daily_pnl.index)
                ex_post_portfolios.index = pd.to_datetime(ex_post_portfolios.index)
                pnl_history.index = pd.to_datetime(pnl_history.index)
                
                mask = (daily_pnl.index >= selmind) & (daily_pnl.index <= selmaxd)
                selected_cumulative_pnl = daily_pnl.loc[mask, "Total"].copy()
    
                selected_cumulative_pnl=daily_pnl.loc[mask,'Total'].copy()
                selected_cumulative_pnl.iloc[0]=0
                
                selected_history=pd.concat([selected_cumulative_pnl.cumsum(),pnl_history['Total'].loc[mask]],axis=1)
                selected_history.columns=['Cumulative P&L','Total P&L']
                
                selected_daily_pnl=daily_pnl.loc[mask].copy()
                
                mask = (positions.index >= selmind) & (positions.index <= selmaxd)
    
                selected_positions=positions.loc[mask]
                
                mask = (ex_post_portfolios.index >= selmind) & (ex_post_portfolios.index <= selmaxd)
    
                cumulative_performance=ex_post_portfolios.loc[mask].pct_change(fill_method=None)
                cumulative_performance.iloc[0] = 0
                cumulative_performance_ex_post = (1 + cumulative_performance).cumprod() * 100
    
                mask = (pnl_history.index >= selmind) & (pnl_history.index <= selmaxd)
    
                pnl_contribution=(pnl_history-pnl_history.shift(1)).loc[mask]
    
                col1,col2=st.columns([1,1])
    
                with col1:
                        
                    fig=px.line(selected_positions,title='Portfolio Value', render_mode = 'svg')
                    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig.update_layout(xaxis_title=None, yaxis_title=None)
                    fig.update_traces(visible="legendonly", selector=lambda t:  not t.name in ['Total'])

                    st.plotly_chart(fig,width='content')
                    
                    fig2=px.line(selected_history,title='Cumulative P&L', render_mode = 'svg')
                    fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Cumulative P&L'])
                    fig2.update_layout(xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig2,width='content')

                    
                    fig3 = px.line(pnl_contribution.cumsum(),x=pnl_contribution.index,y=pnl_contribution.columns,
                     title="Cumulative P&L Contribution")
                    fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Total'])
                    fig3.update_layout(xaxis_title=None, yaxis_title=None,showlegend=True)
                    st.plotly_chart(fig3,width='content')

                    
                with col2:
                    fig4=px.line(cumulative_performance_ex_post,title='Cumulative Return', render_mode = 'svg')
                    fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig4.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Historical Portfolio','Fund','Bitcoin'])
                    fig4.update_layout(xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig4,width='content')
                    
                    drawdown = (cumulative_performance_ex_post - cumulative_performance_ex_post.cummax()) / cumulative_performance_ex_post.cummax()
                
                    fig5=px.line(drawdown,title='Drawdown', render_mode = 'svg')
                    fig5.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig5.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Historical Portfolio','Fund','Bitcoin'])
                    fig5.update_layout(xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig5,width='content')
    
                    fig6 = px.bar(pnl_contribution,x=pnl_contribution.index,y=pnl_contribution.columns,
                         title="Daily P&L",barmode="relative")
                    fig6.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig6.update_traces(visible="legendonly", selector=lambda t:  t.name in ['Total'])
                    fig6.update_layout(xaxis_title=None, yaxis_title=None,showlegend=True)
                    st.plotly_chart(fig6,width='content')
                    
                
                st.dataframe(pnl_contribution.round(2), width='stretch')

                push_button=st.button('Upload Files')

                if push_button:
                    quantities_holding.to_excel('Quantities.xlsx',index=True)
                    positions.to_excel('Positions.xlsx',index=True)
                    trades.to_excel('Trade History Reconstructed.xlsx',index=True)

                    git.push_or_update_file(positions,'Positions')
                    st.success('Positions Updated',icon="✅")   
                    
                    if 'trades' in st.session_state:
                        git.push_or_update_file(trades,'Trade History Reconstructed')
                        st.success('Trades Updated',icon="✅")

                    git.push_or_update_file(quantities_holding,'Quantities')
                    st.success('Quantities Updated',icon="✅")

                    
    with sub_tabs_ex_post[2]:
        
        if 'results' not in st.session_state or st.session_state.results is None:
            st.info("Run Optimization ⬅️")
            
        else:
            if 'ex_post_portfolios' in st.session_state:
            
                ex_post_portfolios=st.session_state.ex_post_portfolios
            
                if 'results' in st.session_state and  st.session_state.results is not None:
                    res=st.session_state.results
                    global_returns=res['cumulative_results']
                        
                else:
                    global_returns=pd.DataFrame()
            
                if not global_returns.empty:
                    ex_post_portfolios=pd.concat([ex_post_portfolios,global_returns],axis=1).sort_index()
                    
                rebalancing_frequency=['Month', 'Year']
        
                selmind,selmaxd=st.session_state['ex_post_tab']
                
                mask = (ex_post_portfolios.index >= selmind) & (ex_post_portfolios.index <= selmaxd)
                cumulative_performance=ex_post_portfolios.loc[mask].pct_change(fill_method=None)
                cumulative_performance.iloc[0] = 0
                cumulative_performance_ex_post = (1 + cumulative_performance).cumprod() * 100
                
                col1, col2, col3 = st.columns([1, 1, 1])
            
                with col1:
                    selected_frequency_calendar_historical = st.selectbox("Frequency:", rebalancing_frequency,index=1,key='selected_frequency_calendar_historical')
        
                with col2:
                    fund_calendar_historical=st.selectbox("Fund:", list(cumulative_performance_ex_post.columns),index=0,key='fund_calendar_historical')
                            
                with col3:
                    benchmark_calendar_historical=st.selectbox("Benchmark:", list(cumulative_performance_ex_post.columns),index=1,key='benchmark_calendar_historical')
        
                if fund_calendar_historical==benchmark_calendar_historical:
                    st.info("Benchmark and Fund must be different ⬅️")
                else:
                    graphs_historical=get_calendar_graph(cumulative_performance_ex_post, 
                                       freq=selected_frequency_calendar_historical, 
                                       benchmark=benchmark_calendar_historical, 
                                       fund=fund_calendar_historical)
        
                col1, col2 = st.columns([1, 1])
                keys=list(graphs_historical.keys())
                with col1:
                    st.plotly_chart(graphs_historical[keys[0]], width='content')
                    st.plotly_chart(graphs_historical[keys[2]], width='content')
                with col2:
                    st.plotly_chart(graphs_historical[keys[1]], width='content')
                    st.plotly_chart(graphs_historical[keys[3]], width='content')           
            else:
                
                st.error("Load Historical Portfolio")
