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

from RiskMetrics import Portfolio, RiskAnalysis,create_constraint,diversification_constraint
from Rebalancing import *

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
        max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
        min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')
        value=(min_value,max_value)
        
        Model = st.slider(
            'Date:',
            min_value=min_value,
            max_value=max_value,
            value=value)
    
        selmin, selmax = Model
        selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
        selmaxd = selmax.strftime('%Y-%m-%d')
        
        # Filter data by selected date range
        mask = (prices_original.index >= selmind) & (prices_original.index <= selmaxd)
        prices=prices_original.loc[mask]
        returns = prices.pct_change().dropna()

        # Load Excel file and ensure datetime index
        
        st.subheader("Asset Returns")
    
        ret=prices.iloc[-1]/prices.iloc[0]-1
        ytd=(1+ret)**(365/(prices.index[-1]-prices.index[0]).days)-1
        ret_ytd=prices.loc[datetime.datetime(max(prices.index.year), 1, 1):].iloc[-1]/prices.loc[datetime.datetime(max(prices.index.year),1,1):].iloc[0]-1
        
        perfs=pd.concat([ret,ret_ytd,ytd],axis=1)
        perfs.columns=['Returns since '+ pd.to_datetime(prices.index[0], format='%Y-%d-%m').strftime("%Y-%m-%d"),
                  'Returns since '+datetime.datetime(max(prices.index.year), 1, 1).strftime("%Y-%m-%d"),
                  'Annualized Returns']
    
        st.dataframe(perfs.T,width='stretch')
        
        st.subheader("Asset Risk")
        
        dates_drawdown=((prices-prices.cummax())/prices.cummax()).idxmin()
        monthly_vol=prices.resample('ME').last().iloc[-50:].pct_change().std()*np.sqrt(12)
    
        drawdown=pd.DataFrame((((prices-prices.cummax()))/prices.cummax()).min())
        Q=0.05
        intervals=np.arange(Q, 1, 0.0005, dtype=float)
        cvar=monthly_vol*norm(loc =0 , scale = 1).ppf(1-intervals).mean()/0.05
        vol=prices.pct_change().iloc[-360:].std()*np.sqrt(260)
    
        risk=pd.concat([vol,monthly_vol,cvar,drawdown,dates_drawdown],axis=1).round(4)
        risk.columns=['Annualized Volatility (daily)','Annualized Volatility (Monthly)','CVar Parametric '+str(int((1-Q)*100))+'%','Max Drawdown','Date of Max Drawdown']
    
        st.dataframe(risk.T,width='stretch')
        
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

        optimized_weights_constraint = portfolio.optimize(objective="sharpe_ratio",constraints=constraints)
        minvar_weights_constraint = portfolio.optimize(objective="minimum_variance",constraints=constraints)
        risk_parity_weights_constraint = portfolio.optimize(objective="risk_parity",constraints=constraints)
        max_diversification_weights_constraint=portfolio.optimize("maximum_diversification",constraints=constraints)
        equal_weights = np.ones(returns.shape[1]) / returns.shape[1]

        st.subheader("Optimized Weights")

        allocation={}

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

    
        metrics={}
        metrics['Expected Returns']={}
        metrics['Expected Volatility']={}
        metrics['Sharpe Ratio']={}
    
        for key in allocation_dict:
    
            metrics['Expected Returns'][key]=(np.round(portfolio.performance(allocation_dict[key]), 4))
            metrics['Expected Volatility'][key]=(np.round(portfolio.variance(allocation_dict[key]), 4))
            sharpe_ratio=np.round(portfolio.performance(allocation_dict[key])/portfolio.variance(allocation_dict[key]),2)
            metrics['Sharpe Ratio'][key]=sharpe_ratio
        
        indicators = pd.DataFrame(metrics,index=allocation_dict.keys())
        
        st.subheader("Portfolio Metrics")
        
        st.dataframe(indicators.T,width='stretch')
   
    # Convert the index to datetime and clean the data

        with st.sidebar:
            st.header("⚙️ Settings")
            
            benchmark = st.selectbox("Benchmark :", list(allocation_dict.keys()))
            frequency = st.selectbox("Rebalancing Frequency:", ['Monthly','Quarterly','Yearly'])
            window_rolling=st.number_input("Sliding Window Size:",min_value=0,value=30,step=1)
        
        portfolio_returns=pd.DataFrame()
    
        
        for key in allocation_dict:
            portfolio_returns['Buy and Hold '+key]=buy_and_hold(prices, allocation_dict[key]).sum(axis=1)
            portfolio_returns['Rebalanced '+key]=rebalanced_portfolio(prices, allocation_dict[key],frequency=frequency).sum(axis=1)
            
        portfolio_returns.index.name='Date'
    
        ret=portfolio_returns.iloc[-1]/portfolio_returns.iloc[0]-1
        ytd=(1+ret)**(365/(portfolio_returns.index[-1]-portfolio_returns.index[0]).days)-1
        ret_ytd=portfolio_returns.loc[datetime.datetime(max(portfolio_returns.index.year),1,1):].iloc[-1]/portfolio_returns.loc[datetime.datetime(max(portfolio_returns.index.year),1,1):].iloc[0]-1
        
        perfs=pd.concat([ret,ret_ytd,ytd],axis=1)
        perfs.columns=['Returns since '+ pd.to_datetime(portfolio_returns.index[0], format='%Y-%d-%m').strftime("%Y-%m-%d"),
                  'Returns since '+datetime.datetime(max(portfolio_returns.index.year), 1, 1).strftime("%Y-%m-%d"),
                  'Annualized Returns']

        st.subheader("Performance")
    
        st.dataframe(perfs.T,width='stretch')
        
        st.subheader("Risk")
            
        tracking_error_daily={}
        tracking_error_monthly={}
        monthly_returns=prices.resample('ME').last().iloc[-180:].pct_change()
    
    
        for key in allocation_dict:
            tracking_error_daily['Buy and Hold '+key]=RiskAnalysis(returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(260)
            tracking_error_daily['Rebalanced '+key]=RiskAnalysis(returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(260)
            tracking_error_monthly['Buy and Hold '+key]=RiskAnalysis(monthly_returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(12)
            tracking_error_monthly['Rebalanced '+key]=RiskAnalysis(monthly_returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(12)
    
        tracking_error_daily=pd.DataFrame(tracking_error_daily.values(),index=tracking_error_daily.keys(),columns=['Tracking Error (daily)'])
        tracking_error_monthly=pd.DataFrame(tracking_error_monthly.values(),index=tracking_error_monthly.keys(),columns=['Tracking Error (Monthly)'])
        
        ptf_drawdown=pd.DataFrame((((portfolio_returns-portfolio_returns.cummax()))/portfolio_returns.cummax()))
        dates_drawdown=ptf_drawdown.idxmin().dt.date
        
        vol=portfolio_returns.pct_change().iloc[:].std()*np.sqrt(260)
        monthly_vol=portfolio_returns.resample('ME').last().iloc[:].pct_change().std()*np.sqrt(12)
        
        rolling_vol=portfolio_returns.pct_change().rolling(window_rolling).std()*np.sqrt(260)
        
        
        drawdown=pd.DataFrame((((portfolio_returns-portfolio_returns.cummax()))/portfolio_returns.cummax()).min())
        Q=0.05
        intervals=np.arange(Q, 1, 0.0005, dtype=float)
        cvar=monthly_vol*norm(loc =0 , scale = 1).ppf(1-intervals).mean()/0.05
    
        risk=pd.concat([vol,tracking_error_daily,monthly_vol,tracking_error_monthly,cvar,drawdown,dates_drawdown],axis=1).round(4)
        risk.columns=['Annualized Volatility (daily)','TEV (daily)',
                      'Annualized Volatility (Monthly)','TEV (Monthly)',
                      'CVar Parametric '+str(int((1-Q)*100))+'%',
                      'Max Drawdown','Date of Max Drawdown']
    
        st.dataframe(risk.T,width='stretch')
        
        st.subheader("Portfolio Value Evolution")
        
            
        fig = px.line(portfolio_returns, title="Portfolio Value Evolution", render_mode = 'svg').update_traces(visible="legendonly", selector=lambda t: not t.name in ["Rebalanced Optimal Portfolio","Buy and Hold Optimal Portfolio"])
        st.plotly_chart(fig,width='stretch')
        
        fig2 = px.line(ptf_drawdown, title="Portfolio Drawdown", render_mode = 'svg').update_traces(visible="legendonly", selector=lambda t: not t.name in ["Rebalanced Optimal Portfolio","Buy and Hold Optimal Portfolio"])
        st.plotly_chart(fig2,width='stretch')
        
        fig3 = px.line(rolling_vol, title="Portfolio Rolling Volatility", render_mode = 'svg').update_traces(visible="legendonly", selector=lambda t: not t.name in ["Rebalanced Optimal Portfolio","Buy and Hold Optimal Portfolio"])

        st.plotly_chart(fig3,width='stretch')
        
        st.write(portfolio_returns)


    with tab2:
        
        st.title("Efficient Frontier")

        max_value = prices_original.index.max().strftime('%Y-%m-%d')
        min_value = prices_original.index.min().strftime('%Y-%m-%d')
        max_value=datetime.datetime.strptime(max_value, '%Y-%m-%d')
        min_value=datetime.datetime.strptime(min_value, '%Y-%m-%d')
        value=(min_value,max_value)
        
        Model = st.slider(
            'Date Efficient Frontier:',
            min_value=min_value,
            max_value=max_value,
            value=value)
    
        selmin, selmax = Model
        selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
        selmaxd = selmax.strftime('%Y-%m-%d')
        
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

        optimal_results['Current Optimal Portfolio']=optimized_weights.tolist()
        optimal_results['Current Optimal Constrained Portfolio']=optimized_weights_constraint.tolist()
        optimal_results['Current Minimum Variance Portfolio']=minvar_weights.tolist()
        optimal_results['Current Minimum Variance Constrained Portfolio']=minvar_weights_constraint
        optimal_results['Current Maximum Diversification Portfolio']=max_diversification.tolist()
        optimal_results['Current Maximum Diversification Portfolio Constrained']=max_diversification_weights_constraint.tolist()
        optimal_results['Current Risk Parity Portfolio']=risk_parity_weights.tolist()
        optimal_results['Current Risk Parity Constrained Portfolio']=risk_parity_weights_constraint.tolist()
        optimal_results['Current Risk Parity Constrained Portfolio']=risk_parity_weights_constraint.tolist()
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
        variance_contrib=pd.DataFrame()
        
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
            temp=pd.DataFrame(portfolio.var_contrib_pct(weight_matrix[key])['Vol Contribution in %'])
            temp.columns=[key]
            variance_contrib=pd.concat([variance_contrib,temp],axis=1)

        @st.cache_data


        
        def get_frontier(returns):
            portfolio_class=RiskAnalysis(returns)
            return portfolio_class.efficient_frontier()
        
        # frontier_weights, frontier_returns, frontier_risks, frontier_sharpe_ratio = portfolio.efficient_frontier()
        frontier_weights, frontier_returns, frontier_risks, frontier_sharpe_ratio = get_frontier(returns)
        frontier = pd.DataFrame(
            {
                "Returns": frontier_returns,
                "Volatility": frontier_risks,
                "Sharpe Ratio": frontier_sharpe_ratio,
            }
        )
    
        fig = px.scatter(
            frontier,
            y="Returns",
            x="Volatility",
            color="Sharpe Ratio",
            color_continuous_scale='blues',
        )

        
        for key in weight_matrix:
            
            fig.add_scatter(
                x=[metrics["Volatility"][key]],
                y=[metrics["Returns"][key]],
                mode="markers",
                marker=dict(color="orange", size=8, symbol="x"),
                name=key)
            
        col1,col2=st.columns([1,1])
        with col1:
            st.subheader('Efficient Frontier')

            fig.update_layout(showlegend=False)
            fig.update_layout(hoverlabel_namelength=-1)
            st.plotly_chart(fig,width='content')

        with col2:
            st.subheader('Correlation Matrix')
            fig = px.imshow(returns.corr().round(2),color_continuous_scale='blues',text_auto=True, aspect="auto")
            fig.update_traces(xgap=2, ygap=2)
            fig.update_traces(textfont=dict(family="Arial Narrow", size=12))
    
            st.plotly_chart(fig,width='content')
        
        indicators = pd.DataFrame(metrics,index=weight_matrix.keys())


        st.subheader("Expected Return")
        
        st.dataframe(indicators.T,width='stretch')
        
        st.subheader("Risk Reward Decomposition")
        
        # st.dataframe(variance_contrib.fillna(0.0000))
    
        funds_options=list(weight_matrix.keys())
        selected_fund= st.selectbox("Fund:", funds_options,index=1)
        selected_weights=weight_matrix[selected_fund]
        
        decomposition = pd.DataFrame(portfolio.var_contrib_pct(selected_weights))*100
        decomposition_vol = pd.DataFrame(portfolio.var_contrib(selected_weights)[0])*100
        decomposition_vol.loc['Total'] = decomposition_vol.sum(axis=0)

        
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

        st.dataframe(decomposition_vol.sort_values(by='Vol Contribution',ascending=False),width='stretch')
