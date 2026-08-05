# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2,gumbel_l
import datetime

from src.RiskMetrics import Portfolio, RiskAnalysis,create_constraint,diversification_constraint
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

st.title("Portfolio Optimization App")

# File Upload
uploaded_file = st.file_uploader("Upload an Excel file with time series", type="xlsx")

if uploaded_file:
    # Create tabs for Portfolio Analysis and Efficient Frontier
        # Load and prepare the data
    prices_original = pd.read_excel(uploaded_file, index_col=0)

    
    tab1, tab2 = st.tabs(["Portfolio Analysis", "Efficient Frontier"])


    with tab1:
        st.title("Asset View")
        
        prices_original.index = pd.to_datetime(prices_original.index)
        max_value = prices_original.index.max().strftime('%Y-%m-%d')
        min_value = prices_original.index.min().strftime('%Y-%m-%d')

        col1,col2=st.columns([1,1])
        with col1:
            
            start_date=st.date_input(label="Start Date",value=min_value,min_value=min_value,max_value=max_value)
        with col2:
            
            end_date=st.date_input(label="End Date",value=max_value,min_value=min_value,max_value=max_value)

        selmind = start_date.strftime('%Y-%m-%d') 
        selmaxd = end_date.strftime('%Y-%m-%d')
        
        # Filter data by selected date range
        mask = (prices_original.index >= selmind) & (prices_original.index <= selmaxd)
        prices=prices_original.loc[mask]
        returns = prices.pct_change().dropna()

        # Load Excel file and ensure datetime index
        
        st.subheader("Asset Returns")
        asset_returns=get_asset_returns(prices)

        st.dataframe(asset_returns,width='stretch')
        
        st.subheader("Asset Risk")
        
        asset_risk=get_asset_risk(prices)
        
        st.dataframe(asset_risk,width='stretch')
        
        st.title("Portfolio Construction")
        
        portfolio = RiskAnalysis(returns)
    
        st.subheader("Constraints")  
                
        data = pd.DataFrame({'Asset':[None],
        'Sign':[None],
        'Limit':[None]
        })
        drop_down_list=list(prices.columns)+['All']
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
        num_rows="dynamic")  # Allow rows to be added dynamically
   
    
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
                    position = np.where(prices.columns == ticker)[0][0]
                    constraint = create_constraint(sign, limit, position)
                    
                constraints.extend(constraint)
                
        
        except Exception as e:
            pass

        st.subheader("Optimized Weights")

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

        equal_weights = np.ones(returns.shape[1]) / returns.shape[1]

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
        allocation_dataframe=pd.DataFrame(allocation,index=returns.columns).T.round(6)
        
        
        allocation_dataframe = st.data_editor(allocation_dataframe, num_rows="dynamic")
    

        st.subheader("Allocation")

        initial_allocation={}
        initial_allocation['Allocation']=[0.0]*prices_original.shape[1]
        
        initial_allocation=pd.DataFrame(initial_allocation,index=prices_original.columns).T
        initial_allocation = st.data_editor(initial_allocation, num_rows="dynamic")
        
        allocation_dict={}
    
        for idx in allocation_dataframe.index:
            allocation_dict[idx]=allocation_dataframe.loc[idx].to_numpy()

        for idx in initial_allocation.index:
            allocation_dict[idx]=initial_allocation.loc[idx].to_numpy()

        allocation_final=pd.DataFrame(allocation_dict,index=returns.columns).T
   
        with st.sidebar:
            st.header("Settings")
            
            benchmark = st.selectbox("Benchmark :", list(allocation_dict.keys()))
            frequency = st.selectbox("Rebalancing Frequency:", ['Monthly','Quarterly','Yearly'])
            window_rolling=st.number_input("Sliding Window Size:",min_value=30,value=252,step=1)
            
        run_optimization=st.button(label='Run Optimization')
        if "portfolio_returns" not in st.session_state:
            st.session_state.portfolio_returns=None
        
        if "efficient_frontier" not in st.session_state:
            st.session_state.efficient_frontier=None    

        if run_optimization:
            st.session_state.portfolio_returns=None
            st.session_state.efficient_frontier=None
            with st.spinner("Optimizing..",show_time=True):
                
                portfolio_returns=pd.DataFrame()
                
                for key in allocation_dict:
                    portfolio_returns['Buy and Hold '+key]=buy_and_hold(prices, allocation_dict[key]).sum(axis=1)
                    portfolio_returns['Rebalanced '+key]=rebalanced_portfolio(prices, allocation_dict[key],frequency=frequency).sum(axis=1)
        
                portfolio_returns.index.name='Date'
                
                st.session_state.portfolio_returns=portfolio_returns
                indicators,fig=get_frontier(returns,allocation_final,constraints)
                st.session_state.efficient_frontier=(indicators,fig)
        
        if "portfolio_returns" not in st.session_state or st.session_state.portfolio_returns is None:
            st.info('Run Optimization')
        else:
            
            portfolio_returns=st.session_state.portfolio_returns
            indicators,fig_frontier=st.session_state.efficient_frontier
            
            risk_benchmark=st.selectbox("Risk Benchmark :", portfolio_returns.columns)

            st.subheader("Portfolio Metrics")
            
            st.dataframe(indicators)
            
            st.subheader("Performance")
            

            perfs=rebalanced_metrics(portfolio_returns)
            
            st.dataframe(perfs,width='stretch')
            
            st.subheader("Risk")
          
            ptf_drawdown=pd.DataFrame((((portfolio_returns-portfolio_returns.cummax()))/portfolio_returns.cummax()))
    
            rolling_vol=portfolio_returns.pct_change().rolling(window_rolling).std()*np.sqrt(260)
            rolling_corr=portfolio_returns.pct_change().rolling(window_rolling).corr(portfolio_returns.pct_change()[risk_benchmark])
            rolling_beta=(portfolio_returns.pct_change().rolling(window_rolling).cov(portfolio_returns.pct_change()[risk_benchmark]))/portfolio_returns.pct_change().rolling(window_rolling).var()

            risk=get_portfolio_risk(allocation_final, prices, portfolio_returns, benchmark)
        
            st.dataframe(risk,width='stretch')
            
            st.subheader("Portfolio Value Evolution")
            
            col1,col2=st.columns([1,1])
            
    
            with col1:
                fig = px.line(portfolio_returns, title="Portfolio Value Evolution", render_mode = 'svg').update_traces(visible="legendonly", selector=lambda t: not t.name in ["Rebalanced Optimal Portfolio","Buy and Hold Optimal Portfolio"])
                st.plotly_chart(fig,width='stretch')
                fig4 = px.line(rolling_corr, title=f"Correlation to {risk_benchmark}", render_mode = 'svg').update_traces(visible="legendonly", selector=lambda t:  not t.name in [rolling_corr.columns[1]])
                st.plotly_chart(fig4,width='stretch')
    

                fig_frontier.update_layout(hoverlabel_namelength=-1,yaxis_tickformat=".2%",xaxis_tickformat=".2%")
                st.plotly_chart(fig_frontier,width='content')
                
                
            with col2:
                
                fig2 = px.line(ptf_drawdown, title="Portfolio Drawdown", render_mode = 'svg').update_traces(visible="legendonly", selector=lambda t: not t.name in ["Rebalanced Optimal Portfolio","Buy and Hold Optimal Portfolio"])
                fig2.update_layout(yaxis_tickformat=".2%")

                st.plotly_chart(fig2,width='stretch')
            
                fig3 = px.line(rolling_vol, title="Portfolio Rolling Volatility", render_mode = 'svg').update_traces(visible="legendonly", selector=lambda t: not t.name in ["Rebalanced Optimal Portfolio","Buy and Hold Optimal Portfolio"])
                fig3.update_layout(yaxis_tickformat=".2%")
        
                st.plotly_chart(fig3,width='stretch')
                
                fig5= px.line(rolling_beta, title=f"Rolling Beta vs {risk_benchmark}", render_mode = 'svg').update_traces(visible="legendonly", selector=lambda t: not t.name in ["Rebalanced Optimal Portfolio","Buy and Hold Optimal Portfolio"])
        
                st.plotly_chart(fig5,width='stretch')
                
            st.write(portfolio_returns)


    with tab2:
        
        st.title("Efficient Frontier")
        
        prices_original.index = pd.to_datetime(prices_original.index)
        max_value = prices_original.index.max().strftime('%Y-%m-%d')
        min_value = prices_original.index.min().strftime('%Y-%m-%d')

        col1,col2=st.columns([1,1])
        with col1:
            
            start_date=st.date_input(label="Start Date",value=min_value,min_value=min_value,max_value=max_value,key='start_date2')
        with col2:
            
            end_date=st.date_input(label="End Date",value=max_value,min_value=min_value,max_value=max_value,key='end_date2')
                
    
        selmind = start_date.strftime('%Y-%m-%d')  # datetime to str
        selmaxd = end_date.strftime('%Y-%m-%d')
        
        # Filter data by selected date range
        mask = (prices_original.index >= selmind) & (prices_original.index <= selmaxd)
        prices=prices_original.loc[mask]
        returns=prices.pct_change()
        
        portfolio = RiskAnalysis(returns)
        
        data = pd.DataFrame({'Assets':[None],
        'Sign':[None],
        'Limit':[None]
        })
        
        drop_down_list=list(prices.columns)+['All']
        # Define dropdown options for the 'Risk Level' column
        column_config = {'Assets':st.column_config.SelectboxColumn(
            options=drop_down_list),
        'Sign': st.column_config.SelectboxColumn(
            options=["=", "≥", "≤"],  # Dropdown options
            help="Select the risk level for each asset."  # Tooltip for the column
        )
        }

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
                    position = np.where(prices.columns == ticker)[0][0]
                    constraint = create_constraint(sign, limit, position)
                    
                constraints.extend(constraint)
        except Exception as e:
            pass

        optimized_weights_constraint = portfolio.optimize(objective="sharpe_ratio",constraints=constraints)
        minvar_weights_constraint = portfolio.optimize(objective="minimum_variance",constraints=constraints)
        risk_parity_weights_constraint = portfolio.optimize(objective="risk_parity",constraints=constraints)
        max_diversification_weights_constraint=portfolio.optimize("maximum_diversification",constraints=constraints)

        optimized_weights = portfolio.optimize(objective="sharpe_ratio")
        minvar_weights = portfolio.optimize(objective="minimum_variance")
        risk_parity_weights = portfolio.optimize(objective="risk_parity")
        max_diversification=portfolio.optimize(objective="maximum_diversification")
        
        optimal_results={}  

        optimal_results['Optimal Portfolio']=optimized_weights.tolist()
        optimal_results['Optimal Constrained Portfolio']=optimized_weights_constraint.tolist()
        optimal_results['Minimum Variance Portfolio']=minvar_weights.tolist()
        optimal_results['Minimum Variance Constrained Portfolio']=minvar_weights_constraint
        optimal_results['Maximum Diversification Portfolio']=max_diversification.tolist()
        optimal_results['Maximum Diversification Portfolio Constrained']=max_diversification_weights_constraint.tolist()
        optimal_results['Risk Parity Portfolio']=risk_parity_weights.tolist()
        optimal_results['Risk Parity Constrained Portfolio']=risk_parity_weights_constraint.tolist()
        optimal_results['Risk Parity Constrained Portfolio']=risk_parity_weights_constraint.tolist()
        optimal_results['Equal Weights']=equal_weights.tolist()
        
        former_results={}
        
        for idx in allocation_dataframe.index:
            former_results[idx]=allocation_dict[idx].tolist()

        for idx in initial_allocation.index:
            former_results[idx]=allocation_dict[idx].tolist()


        former_results=pd.DataFrame(former_results,index=prices.columns).T.round(6)

        st.subheader("Results since Inception")

        editable_weights = st.data_editor(former_results, num_rows="dynamic",width='stretch')

        st.subheader("Results with current timeframe")

        current_results={}
        
        for key in optimal_results:
            current_results[key]=optimal_results[key]
            
        for idx in initial_allocation.index:
            current_results[idx]=allocation_dict[idx].tolist()    

    
        current_results_dataframe=pd.DataFrame(current_results,index=prices.columns).T.round(6)
        current_results=st.data_editor(current_results_dataframe, num_rows="dynamic")
        weight_matrix={}
        variance_contrib_summary=pd.DataFrame()
        
        for idx in current_results.index:
            weight_matrix[idx]=current_results.loc[idx].to_numpy()

        metrics={}
        metrics['Returns']={}
        metrics['Volatility']={}
        metrics['Sharpe Ratio']={}

        for key in weight_matrix:
    
            metrics['Returns'][key]=(np.round(portfolio.performance(weight_matrix[key]), 4))
            metrics['Volatility'][key]=(np.round(portfolio.variance(weight_matrix[key]), 4))
            metrics['Sharpe Ratio'][key]=np.round(metrics['Returns'][key]/metrics['Volatility'][key],4)
            temp=pd.DataFrame(portfolio.var_contrib(weight_matrix[key])[0]['Vol Contribution'])*100
            temp.columns=[key]
            
            variance_contrib_summary=pd.concat([variance_contrib_summary,temp],axis=1)
        variance_contrib_summary.loc['Total']=variance_contrib_summary.sum(axis=0)
        variance_contrib_summary=variance_contrib_summary.sort_values(by=variance_contrib_summary.columns[0],ascending=False)


        
        if "efficient_frontier_2" not in st.session_state:
            
            st.session_state.efficient_frontier_2=None    
            
        run_frontier=st.button(label='Run Efficient Frontier')

        if run_frontier:
            st.session_state.efficient_frontier_2=None
            
            with st.spinner("Optimizing..",show_time=True):
                indicators,fig=get_frontier(returns,current_results_dataframe,constraints)
                st.session_state.efficient_frontier_2=indicators,fig
                
        if "efficient_frontier_2" in st.session_state and st.session_state.efficient_frontier_2 is not None:
            
            indicators,fig=st.session_state.efficient_frontier_2
            
            col1,col2=st.columns([1,1])
            
            with col1:
                st.subheader('Efficient Frontier')
    
                fig.update_layout(hoverlabel_namelength=-1,yaxis_tickformat=".2%",xaxis_tickformat=".2%")
                st.plotly_chart(fig,width='content')
    
            with col2:
                st.subheader('Correlation Matrix')
                fig = px.imshow(returns.corr().round(2),color_continuous_scale='blues',text_auto=True, aspect="auto")
                fig.update_traces(xgap=2, ygap=2)
                fig.update_traces(textfont=dict(family="Arial Narrow", size=12))
        
                st.plotly_chart(fig,width='content')
            
    
            st.subheader("Expected Return")
                
            st.dataframe(indicators.T,width='stretch')
        else:
            st.info('Run Efficient Frontier')
            
            
        st.subheader("Risk Reward Decomposition")
        
        # st.dataframe(variance_contrib.fillna(0.0000))
    
        funds_options=list(weight_matrix.keys())
        selected_fund= st.selectbox("Fund:", funds_options,index=1)
        selected_weights=weight_matrix[selected_fund]
        
        decomposition = pd.DataFrame(portfolio.var_contrib(selected_weights)[0])*100

        quantities_rebalanced=rebalanced_portfolio(prices,selected_weights,frequency=frequency)/prices
        quantities_buy_hold=buy_and_hold(prices,selected_weights)/prices
        
        cost_rebalanced=rebalanced_book_cost(prices,quantities_rebalanced)
        cost_buy_and_hold=rebalanced_book_cost(prices,quantities_buy_hold)
        
        mtm_rebalanced=quantities_rebalanced*prices
        mtm_buy_and_hold=quantities_buy_hold*prices

        pnl_buy_and_hold=pd.DataFrame((mtm_buy_and_hold-cost_buy_and_hold).iloc[-1])
        pnl_buy_and_hold.columns=['Profit and Loss (Buy and Hold)']
        
        pnl_rebalanced=pd.DataFrame((mtm_rebalanced-cost_rebalanced).iloc[-1])
        pnl_rebalanced.columns=['Profit and Loss (Rebalanced)']


        pnl=pd.concat([pnl_buy_and_hold,pnl_rebalanced,decomposition],axis=1)
        pnl.loc['Total']=pnl.sum(axis=0)
        
        st.dataframe(pnl.fillna(0).sort_values(by='Profit and Loss (Rebalanced)',ascending=False),width='stretch')
        st.dataframe(variance_contrib_summary,width='stretch')
        
