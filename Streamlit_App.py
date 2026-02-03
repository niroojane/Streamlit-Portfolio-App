# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


import streamlit as st
import pandas as pd
import random
import numpy as np
import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from RiskMetrics import *
from Stock_Data import get_close
from Binance_API import *
from PnL_Computation import *

from Git import *
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



def load_data(tickers,start=datetime.datetime(2023,1,1),today=None):

    if today is None:
        today = datetime.date.today()
    days_total = (today - start).days
    if days_total <= 0:
        print("Start date must be in the past.")
        return

    remaining = days_total % 500
    numbers_of_table = days_total // 500
    temp_end = datetime.datetime.combine(start, datetime.time())
    scope_prices = pd.DataFrame()

    for _ in range(numbers_of_table + 1):
        data = Binance.get_price(tickers, temp_end)
        temp_end += datetime.timedelta(days=500)
        scope_prices = scope_prices.combine_first(data)

    temp_end = datetime.datetime.combine(today - datetime.timedelta(days=remaining), datetime.time())
    data = Binance.get_price(tickers, temp_end)
    scope_prices = scope_prices.combine_first(data)

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
    
    st.session_state.book_cost=book_cost
    st.session_state.realized_pnl=realized_pnl
    st.session_state.profit_and_loss=profit_and_loss
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
    
    trade_history = read_excel_from_url(url_trades)
    
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
    positions=positions.loc[~positions.index.duplicated(keep='first'),:]
    positions['Total']=positions.loc[:,positions.columns!='Total'].sum(axis=1)
    
    # quantities_history=pd.read_excel('Quantities.xlsx',index_col=0)
    
    quantities_holding.index=pd.to_datetime(quantities_holding.index)
    quantities_holding=pd.concat([quantities_holding,quantities_history])
    quantities_holding=quantities_holding.loc[~quantities_holding.index.duplicated(),:]

    quantities_holding=quantities_holding.sort_index()

    st.session_state.quantities_holding=quantities_holding
    st.session_state.positions=positions
        
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
    
    if Binance is not None:
        get_positions()

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
    
    tickers_market_cap=Binance.get_market_cap()
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
        
        fig = px.line(dataframe.loc[mask], title='Price', width=800, height=400)
        fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
        fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
        fig.update_traces(visible="legendonly", selector=lambda t: not t.name in ["BTCUSDT"])

        cumulative_returns=returns_to_use.loc[mask].copy()
        cumulative_returns.iloc[0]=0
        cumulative_returns=(1+cumulative_returns).cumprod()*100
        
        fig2 = px.line(cumulative_returns, title='Cumulative Performance', width=800, height=400)
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

                cumulative_performance=res['cumulative_results'].loc[mask].pct_change()
                cumulative_performance.iloc[0] = 0
                cumulative_results = (1 + cumulative_performance).cumprod() * 100
                
                drawdown = (cumulative_results - cumulative_results.cummax()) / cumulative_results.cummax()
                rolling_vol_ptf = cumulative_results.pct_change().rolling(window_vol).std() * np.sqrt(260)
                
                frontier_indicators, fig4 = get_frontier(range_returns, res['alloc_df'])
        
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

        
with main_tabs[3]: 
    
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
            
            decomposition = pd.DataFrame(portfolio.var_contrib_pct(selected_weights))*100
            
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
    
            var_button=st.button("Get Value At Risk")
        
            selected_fund_var=st.selectbox("Fund:", list(allocation_dataframe.index),index=0,key='selected_fund_var')
    
        
            horizon = 1 / 250
            spot = dataframe.iloc[-1]
            theta = 2
        
            distrib_functions = {
                'multivariate_distribution': (iterations, stress_factor),
                'gaussian_copula': (iterations, stress_factor),
                't_copula': (iterations, stress_factor),
                'gumbel_copula': (iterations, theta),
                'monte_carlo': (spot, horizon, iterations, stress_factor)
            }
    
            
            var_scenarios, cvar_scenarios, fund_results = {}, {}, {}
            
            portfolio = RiskAnalysis(range_returns)
            
            if "fund_results" not in st.session_state:
                st.session_state.fund_results = None
                st.session_state.var_scenarios=None
                st.session_state.cvar_scenarios=None
                
            if var_button:
            
                st.session_state.fund_results=None
                st.session_state.var_scenarios=None
                st.session_state.cvar_scenarios=None
                
                for index in allocation_dataframe.index:
                    var_scenarios[index], cvar_scenarios[index] = {}, {}
                    for func_name, args in distrib_functions.items():
                        func = getattr(portfolio, func_name)
                        scenarios = {}
                
                        for i in range(num_scenarios):
                            if func_name == 'monte_carlo':
                                distrib = pd.DataFrame(func(*args)[1], columns=range_returns.columns)
                            else:
                                distrib = pd.DataFrame(func(*args), columns=range_returns.columns)
                    
                            distrib = distrib * allocation_dataframe.loc[index]
                            distrib = distrib[distrib.columns[allocation_dataframe.loc[index] > 0]]
                            distrib['Portfolio'] = distrib.sum(axis=1)
                    
                            results = distrib.sort_values(by='Portfolio').iloc[int(distrib.shape[0] * var_centile)]
                            scenarios[i] = results
                
                        scenario = pd.DataFrame(scenarios).T
                        mean_scenario = scenario.mean()
                        index_cvar = scenario['Portfolio'] < mean_scenario['Portfolio']
                        cvar = scenario.loc[index_cvar].mean()
                    
                        var_scenarios[index][func_name] = mean_scenario
                        cvar_scenarios[index][func_name] = cvar
                    
                    fund_results[index] = {'Value At Risk': mean_scenario.loc['Portfolio'],'CVaR': cvar.loc['Portfolio']}
        

        
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
with main_tabs[4]:
    
    
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
            
            fig3=px.line((1+historical_PCA).cumprod()*100,title='Eigen Index')
            fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
            fig3.update_traces(textfont=dict(family="Arial Narrow", size=15))
    
            fig4=px.line(pca_similarity,title='PCA Similarity')
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
            selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
            selmaxd = selmax.strftime('%Y-%m-%d')

            
            # with col1:
            dropdown_asset1=st.selectbox("Asset 1:",options=range_returns.columns,index=0)

        # with col2:
            dropdown_asset2=st.selectbox("Asset 2:",options=range_returns.columns,index=1)
                    
        # with col3:
            window_corr=st.number_input("Window Correlation",min_value=0,value=252)
    
            mask = (dataframe.index >= selmind) & (dataframe.index <= selmaxd)
            col1, col2, col3 = st.columns([1, 1, 1])

            
            range_prices=dataframe.loc[mask].copy()
            range_returns=returns_to_use.loc[mask].copy()
    
            pca_over_time=first_pca_over_time(returns=range_returns,window=window_corr)
    
            rolling_correlation = range_returns[dropdown_asset1].rolling(window_corr).corr(
                range_returns[dropdown_asset2]
            ).dropna()
            
            fig = px.line(rolling_correlation, title=f"{dropdown_asset1}/{dropdown_asset2} Correlation")
            fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white", width=800, height=400)
            fig.update_traces(textfont=dict(family="Arial Narrow", size=15))
    
    
            fig2 = px.imshow(range_returns.corr().round(2), title='Correlation Matrix',color_continuous_scale='blues', text_auto=True, aspect="auto")
            fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
            fig2.update_traces(xgap=2, ygap=2)
            fig2.update_traces(textfont=dict(family="Arial Narrow", size=15))
            
            fig3=px.line(pca_over_time,title='First principal component (Variance Explained in %)')
            fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
            fig3.update_layout(xaxis_title=None, yaxis_title=None)

            with col1:
                st.plotly_chart(fig,width='content')
            with col2:
                st.plotly_chart(fig3,width='content')
            with col3:
                st.plotly_chart(fig2,width='content')


with main_tabs[2]:
    
    sub_tabs_ex_post=st.tabs(['Positioning','Historical Portfolio','Calendar Return'])
    
    with sub_tabs_ex_post[0]:

        if "dataframe" not in st.session_state:
            st.info("Load data first ⬅️")
            
        if "results" not in st.session_state:
            st.info("Compute Optimization first ⬅️")
            
        col1, col2, _ = st.columns([1, 1, 10])
        
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

        check_connection(position_url,quantities_url,trades_url)
        
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
                    
                    days=(today-start_date).days
                    
                    remaining=days%500
                    numbers_of_table=days//500
                    remaining
                    temp_end=weights_ex_post.index[0]
                    prices=pd.DataFrame()
                    
                    for i in range(numbers_of_table+1):
                        temp_data=Binance.get_price(weights_ex_post.columns,temp_end)
                        temp_end=temp_end+datetime.timedelta(500)
                        prices=prices.combine_first(temp_data)
            
                    temp_end=temp_end+datetime.timedelta(500)
                    last_data=Binance.get_price(weights_ex_post.columns,temp_end)
                    binance_data=prices.combine_first(last_data)
                    binance_data=binance_data.sort_index()
                    binance_data = binance_data[~binance_data.index.duplicated(keep='first')]
                    binance_data.index=pd.to_datetime(binance_data.index)
            
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
                    
                    performance_ex_post=historical_ptf['Historical Portfolio'].copy()
                    performance_ex_post=performance_ex_post.to_frame()
                    performance_ex_post=historical_ptf['Historical Portfolio'].copy()
                    
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
    
                selected_positions=positions.loc[mask,"Total"]
                
                mask = (ex_post_portfolios.index >= selmind) & (ex_post_portfolios.index <= selmaxd)
    
                cumulative_performance=ex_post_portfolios.loc[mask].pct_change()
                cumulative_performance.iloc[0] = 0
                cumulative_performance_ex_post = (1 + cumulative_performance).cumprod() * 100
    
                mask = (pnl_history.index >= selmind) & (pnl_history.index <= selmaxd)
    
                pnl_contribution=(pnl_history-pnl_history.shift(1)).loc[mask]
    
                col1,col2=st.columns([1,1])
    
                with col1:
                        
                    fig=px.line(selected_positions,title='Portfolio Value')
                    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig.update_layout(xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig,width='content')
                    
                    fig2=px.line(selected_history,title='Cumulative P&L')
                    fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig2.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Cumulative P&L'])
                    fig2.update_layout(xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig2,width='content')
    
                with col2:
                    fig3=px.line(cumulative_performance_ex_post,title='Cumulative Return')
                    fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig3.update_traces(visible="legendonly", selector=lambda t: not t.name in ['Historical Portfolio'])
                    fig3.update_layout(xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig3,width='content')
    
                    fig4 = px.bar(selected_daily_pnl, color=selected_daily_pnl['color'],
                         color_discrete_map={'green': 'green', 'red': 'red'},
                         title="Daily P&L")
                    
                    fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",width=800, height=400)
                    fig4.update_layout(xaxis_title=None, yaxis_title=None,showlegend=False)
                    st.plotly_chart(fig4,width='content')
                
                st.dataframe(pnl_contribution.round(2), use_container_width=True)

                push_button=st.button('Upload Files')

                if push_button:
                    quantities_holding.to_excel('Quantities.xlsx',index=False)
                    positions.to_excel('Positions.xlsx')

                    git.push_or_update_file(positions,'Positions')
                    st.success('Positions Updated',icon="✅")                    

                    git.push_or_update_file(quantities_holding,'Quantities')
                    st.success('Quantities Updated',icon="✅")
    
    with sub_tabs_ex_post[2]:

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
            cumulative_performance=ex_post_portfolios.loc[mask].pct_change()
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
