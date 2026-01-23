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

import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, chi2,gumbel_l

from IPython.display import HTML
from io import BytesIO
import requests
import base64

from RiskMetrics import *
from Rebalancing import *

def display_scrollable_df(df, max_height="50vh", max_width="90vw"):
    style = f"""
    <div style="
        display: flex;
        justify-content: center;
        padding: 20px;
    ">
        <div style="
            overflow: auto;
            max-height: {max_height};
            max-width: {max_width};
            width: 100%;
            border: 1px solid #444;
            padding: 10px;
            background-color: #000;
            color: #eee;
            font-family: 'Arial Narrow', Arial, sans-serif;
            box-sizing: border-box;
        ">
            {df.to_html(classes='table', border=0, index=True)}
        </div>
    </div>
    """
    return HTML(style)

def build_constraint(prices, constraint_matrix):
    constraints = []
    dico_map = {'=': 'eq', '≥': 'ineq', '≤': 'ineq'}

    drop_down_list_asset = list(prices.columns) + ['All']
    drop_down_list = drop_down_list_asset + [None]

    try:
        for row in range(constraint_matrix.shape[0]):
            temp = constraint_matrix[row, :]
            ticker = temp[0]

            if ticker not in drop_down_list:
                continue

            sign = temp[1]
            limit = float(temp[2])

            if ticker == 'All':
                constraint = diversification_constraint(sign, limit)

            elif ticker in drop_down_list_asset:
                position = np.where(prices.columns == ticker)[0][0]
                constraint = create_constraint(sign, limit, position)

            constraints.extend(constraint)

    except Exception as e:
        print(f"Error in build_constraint: {e}")

    return constraints

    
def get_expected_metrics(returns,dataframe):
    portfolio=RiskAnalysis(returns)
    allocation_dict={}

    for idx in dataframe.index:
        allocation_dict[idx]=dataframe.loc[idx].to_numpy()


    
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

    return indicators.T.round(4)
def rebalanced_time_series(prices,dataframe,frequency='Monthly'):

    portfolio_returns=pd.DataFrame()

    for key in dataframe.index:
        portfolio_returns['Buy and Hold '+key]=buy_and_hold(prices, dataframe.loc[key]).sum(axis=1)
        portfolio_returns['Rebalanced '+key]=rebalanced_portfolio(prices, dataframe.loc[key],frequency=frequency).sum(axis=1)

    portfolio_returns.index.name='Date'
    
    return portfolio_returns

def rebalanced_metrics(portfolio_returns):

    ret=portfolio_returns.iloc[-1]/portfolio_returns.iloc[0]-1
    ytd=(1+ret)**(365/(portfolio_returns.index[-1]-portfolio_returns.index[0]).days)-1
    ret_ytd=portfolio_returns.loc[datetime.datetime(max(portfolio_returns.index.year),1,1):].iloc[-1]/portfolio_returns.loc[datetime.datetime(max(portfolio_returns.index.year),1,1):].iloc[0]-1

    perfs=pd.concat([ret,ret_ytd,ytd],axis=1)
    perfs.columns=['Returns since '+ pd.to_datetime(portfolio_returns.index[0], format='%Y-%d-%m').strftime("%Y-%m-%d"),
              'Returns since '+datetime.datetime(max(portfolio_returns.index.year), 1, 1).strftime("%Y-%m-%d"),
              'Annualized Returns']
    
    return perfs.T.round(4)

    
def get_portfolio_risk(dataframe,prices,portfolio_returns,benchmark):

    allocation_dict={}
    
    returns=prices.pct_change()
    
    for idx in dataframe.index:
        allocation_dict[idx]=dataframe.loc[idx].to_numpy()


    tracking_error_daily={}
    tracking_error_monthly={}
    monthly_returns=prices.resample('ME').last().iloc[-180:].pct_change()


    for key in allocation_dict:
        if key not in allocation_dict or benchmark not in allocation_dict:
            continue

        tracking_error_daily['Buy and Hold '+key]=RiskAnalysis(returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(260)
        tracking_error_daily['Rebalanced '+key]=RiskAnalysis(returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(260)
        tracking_error_monthly['Buy and Hold '+key]=RiskAnalysis(monthly_returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(12)
        tracking_error_monthly['Rebalanced '+key]=RiskAnalysis(monthly_returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(12)

 
            
    tracking_error_daily=pd.DataFrame(tracking_error_daily.values(),index=tracking_error_daily.keys(),columns=['Tracking Error (daily)'])
    tracking_error_monthly=pd.DataFrame(tracking_error_monthly.values(),index=tracking_error_monthly.keys(),columns=['Tracking Error (Monthly)'])

    dates_drawdown=((portfolio_returns-portfolio_returns.cummax())/portfolio_returns.cummax()).idxmin().dt.date
    
    vol=portfolio_returns.pct_change().iloc[:].std()*np.sqrt(260)
    monthly_vol=portfolio_returns.resample('ME').last().iloc[:].pct_change().std()*np.sqrt(12)

    drawdown=pd.DataFrame((((portfolio_returns-portfolio_returns.cummax()))/portfolio_returns.cummax()).min())
    Q=0.05
    intervals=np.arange(Q, 1, 0.0005, dtype=float)
    cvar=monthly_vol*norm(loc =0 , scale = 1).ppf(1-intervals).mean()/0.05

    risk=pd.concat([vol,tracking_error_daily,monthly_vol,tracking_error_monthly,cvar,drawdown,dates_drawdown],axis=1).round(4)
    risk.columns=['Annualized Volatility (daily)','TEV (daily)',
                  'Annualized Volatility (Monthly)','TEV (Monthly)',
                  'CVar Parametric '+str(int((1-Q)*100))+'%',
                  'Max Drawdown','Date of Max Drawdown']
    
    return risk.T.round(4)
    
def get_asset_returns(prices):
    
    ret=prices.iloc[-1]/prices.iloc[0]-1
    ytd=(1+ret)**(365/(prices.index[-1]-prices.index[0]).days)-1
    ret_ytd=prices.loc[datetime.datetime(max(prices.index.year), 1, 1):].iloc[-1]/prices.loc[datetime.datetime(max(prices.index.year),1,1):].iloc[0]-1

    perfs=pd.concat([ret,ret_ytd,ytd],axis=1)
    perfs.columns=['Returns since '+ pd.to_datetime(prices.index[0], format='%Y-%d-%m').strftime("%Y-%m-%d"),
              'Returns since '+datetime.datetime(max(prices.index.year), 1, 1).strftime("%Y-%m-%d"),
              'Annualized Returns']
    
    return perfs.T.round(4)

def get_asset_risk(prices):

    dates_drawdown=((prices-prices.cummax())/prices.cummax()).idxmin().dt.date
    
    vol=prices.pct_change().iloc[-260:].std()*np.sqrt(260)
    weekly_vol=prices.resample('W').last().iloc[-153:].pct_change().std()*np.sqrt(52)
    monthly_vol_1Y=prices.resample('ME').last().iloc[-50:].pct_change().std()*np.sqrt(12)
    monthly_vol_5Y=prices.resample('ME').last().iloc[-181:].pct_change().std()*np.sqrt(12)

    drawdown=pd.DataFrame((((prices-prices.cummax()))/prices.cummax()).min())
    Q=0.05
    intervals=np.arange(Q, 1, 0.0005, dtype=float)
    cvar=monthly_vol_5Y*norm(loc =0 , scale = 1).ppf(1-intervals).mean()/0.05

    risk=pd.concat([vol,weekly_vol,monthly_vol_1Y,monthly_vol_5Y,cvar,drawdown,dates_drawdown],axis=1).round(4)
    risk.columns=['Annualized Volatility (daily)',
    'Annualized Volatility 3Y (Weekly)',
    'Annualized Volatility 5Y (Monthly)','Annualized Volatility since '+str(prices.index[0].year) +' (Monthly)',
    'CVar Parametric '+str(int((1-Q)*100))+'%','Max Drawdown','Date of Max Drawdown']
    
    
    return risk.T.round(4)

def get_yearly_metrics(portfolio_returns,fund='Fund',bench='Bitcoin'):

    portfolio_returns = portfolio_returns[[fund, bench]].ffill()
    portfolio_returns_pct = portfolio_returns.pct_change(fill_method=None)

    year_returns = {}
    year_vol = {}
    year_tracking_error = {}
    year_sharpe_ratio = {}

    years = sorted(portfolio_returns.index.strftime('%Y').unique())

    for i, year in enumerate(years):
        mask_curr = portfolio_returns.index.strftime('%Y') == year
        temp = portfolio_returns_pct.loc[mask_curr]
        temp_prices = portfolio_returns.loc[mask_curr]

        if len(temp_prices) < 2:
            continue  

        if i > 0:
            prev_mask = portfolio_returns.index.strftime('%Y') == years[i - 1]
            prev_prices = portfolio_returns.loc[prev_mask]
            if not prev_prices.empty:
                first_price = prev_prices.iloc[-1]
            else:
                first_price = temp_prices.iloc[0]
        else:
            first_price = temp_prices.iloc[0]

        last_price = temp_prices.iloc[-1]

        perf_year = last_price / first_price - 1
        year_returns[year] = perf_year

        vol = temp.std() * np.sqrt(252)
        year_vol[year] = vol

        tracking_error = (temp[fund] - temp[bench]).std() * np.sqrt(252)
        year_tracking_error[year] = tracking_error

        sharpe = perf_year / vol
        year_sharpe_ratio[year] = sharpe

    year_returns_df = pd.DataFrame(year_returns).round(4)
    year_vol_df = pd.DataFrame(year_vol).round(4)
    year_tracking_error_df = pd.DataFrame.from_dict(
        year_tracking_error, orient='index', columns=['Tracking Error']
    ).round(4)
    year_sharpe_ratio_df = pd.DataFrame(year_sharpe_ratio).round(4)

    return year_returns_df, year_vol_df, year_tracking_error_df, year_sharpe_ratio_df

def get_monthly_metrics(portfolio_returns, fund='Fund', bench='Bitcoin'):
    portfolio_returns = portfolio_returns[[fund, bench]].ffill()
    portfolio_returns_pct = portfolio_returns.pct_change(fill_method=None)

    month_returns = {}
    month_vol = {}
    month_tracking_error = {}
    month_sharpe_ratio = {}

    month_years = sorted(portfolio_returns.index.strftime('%Y-%m').unique())

    for i, month in enumerate(month_years):

        mask_curr = portfolio_returns.index.strftime('%Y-%m') == month
        temp = portfolio_returns_pct.loc[mask_curr]
        temp_prices = portfolio_returns.loc[mask_curr]

        if len(temp_prices) < 2:
            continue  
        if i > 0:
            prev_mask = portfolio_returns.index.strftime('%Y-%m') == month_years[i - 1]
            prev_prices = portfolio_returns.loc[prev_mask]
            if not prev_prices.empty:
                first_price = prev_prices.iloc[-1]  
            else:
                first_price = temp_prices.iloc[0]
        else:
            first_price = temp_prices.iloc[0]

        last_price = temp_prices.iloc[-1]

        perf_month = last_price / first_price - 1
        month_returns[month] = perf_month

        vol = temp.std() * np.sqrt(252)
        month_vol[month] = vol

        tracking_error = (temp[fund] - temp[bench]).std() * np.sqrt(252)
        month_tracking_error[month] = tracking_error

        sharpe = perf_month / vol
        month_sharpe_ratio[month] = sharpe

    month_returns_df = pd.DataFrame(month_returns).round(4)
    month_vol_df = pd.DataFrame(month_vol).round(4)
    month_tracking_error_df = pd.DataFrame.from_dict(
        month_tracking_error, orient='index', columns=['Tracking Error']
    ).round(4)
    month_sharpe_ratio_df = pd.DataFrame(month_sharpe_ratio).round(4)

    return month_returns_df, month_vol_df, month_tracking_error_df, month_sharpe_ratio_df
    
def get_calendar_graph(performance_fund,fund='Fund',benchmark='Bitcoin',freq='Year'):

    dico_fig={}
    if freq=='Year':
        metrics = get_yearly_metrics(performance_fund,fund=fund,bench=benchmark)
    else:
        metrics = get_monthly_metrics(performance_fund,fund=fund,bench=benchmark)
        
    titles = [
        f"Returns by {freq}",
        f"Volatility by {freq}",
        f"Tracking Error by {freq}",
        f"Sharpe Ratio by {freq}"
    ]
    
    for i, title in enumerate(titles):
        if title!=f"Tracking Error by {freq}":
            df = metrics[i].T  # <-- no transpose
            fig = px.bar(
                df,
                x=df.index,  # years on x-axis
                y=df.columns,
                barmode="group",
                title=title,
                labels={"value": "Value", "variable": "Asset", "x": "Year"}
            )
    
            fig.update_layout(
                plot_bgcolor="black",
                paper_bgcolor="black",
                font_color="white",
                title_font=dict(size=20),
                legend=dict(title="Legend")
            )
    
            fig.update_traces(
                textfont=dict(family="Arial Narrow", size=15),
                hovertemplate="Year: %{x}<br>%{y:.3f}<extra></extra>")

            
        else:
            
            df = metrics[i]
            fig = px.bar(df, x=df.index, y=df.columns,
                         barmode="group",  # use "overlay" to overlap instead
                         title=f"Tracking Error by {freq}",
                         labels={"value": "Value", "Year": "Year", "variable": "Asset"})
            
            # fig.update_layout(xaxis=dict(dtick=1))
            fig.update_layout(
                plot_bgcolor="black",
                paper_bgcolor="black",
                font_color="white",
                title_font=dict(size=20),
                legend=dict(title="Legend")
            )            
            
            fig.update_traces(
                textfont=dict(family="Arial Narrow", size=15),
                hovertemplate="Year: %{x}<br>%{y:.3f}<extra></extra>")
            
        dico_fig[title]=fig
        
    return dico_fig
        
def get_frontier(returns,dataframe):
    portfolio=RiskAnalysis(returns)
    frontier_weights, frontier_returns, frontier_risks, frontier_sharpe_ratio = portfolio.efficient_frontier()
    
    weight_matrix={}

    for idx in dataframe.index:
        
        weight_matrix[idx]=dataframe.loc[idx].to_numpy()
    
    metrics = {
        'Returns': {},
        'Volatility': {},
        'Sharpe Ratio': {}
    }
    for key in weight_matrix:
        
        metrics['Returns'][key]=(np.round(portfolio.performance(weight_matrix[key]), 4))
        metrics['Volatility'][key]=(np.round(portfolio.variance(weight_matrix[key]), 4))
        metrics['Sharpe Ratio'][key]=np.round(metrics['Returns'][key]/metrics['Volatility'][key],4)
    
    
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
            name=key,
        )
        
        
    fig.add_scatter(
        x=[metrics["Volatility"]['Optimal Portfolio']],
        y=[metrics["Returns"]['Optimal Portfolio']],
        mode="markers",
        marker=dict(color="red", size=8, symbol="x"),
        name='Optimal Portfolio',
    )
    
    fig.update_layout(
        showlegend=False, 
        hoverlabel_namelength=-1,
        font=dict(
            family="Arial Narrow",
            size=14,
            color="white" 
        ),
        plot_bgcolor="black", 
        paper_bgcolor="black"  
    )
    
    fig.update_layout(showlegend=False)
    fig.update_layout(hoverlabel_namelength=-1)
    indicators = pd.DataFrame(metrics,index=weight_matrix.keys()).T

    return indicators,fig

def read_excel_from_url(url,index_col=None):
        try:
            response = requests.get(url)
            response.raise_for_status()  # raises HTTPError for 4xx / 5xx
    
            return pd.read_excel(BytesIO(response.content), index_col=index_col)
    
        except requests.exceptions.HTTPError as e:
            # File not found (404) or server error
            print(f"HTTP error while downloading {url}: {e}")
    
        except ValueError as e:
            # File exists but is not a valid Excel file
            print(f"Invalid Excel file at {url}: {e}")
    
        except RequestException as e:
            # Network issues, timeout, DNS, etc.
            print(f"Request failed for {url}: {e}")
    
        return None      