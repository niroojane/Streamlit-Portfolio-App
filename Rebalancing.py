#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:

def get_rebalancing_dates(data,frequency='Quarterly'):
    
    if frequency=='Quarterly':
        month=list(sorted(set(data.index + pd.offsets.BQuarterEnd(0))))
    
    elif frequency=='Monthly':

        month=list(sorted(set(data.index + pd.offsets.BMonthEnd(0))))

    elif frequency=='Yearly':
        month=list(sorted(set(data.index + pd.offsets.BYearEnd(0))))
        
    
    month = pd.to_datetime(month)

    idx1 = pd.Index(data.iloc[:-1].index)
    idx2 = pd.Index(month)
    
    closest_dates = idx1[idx1.get_indexer(idx2, method='nearest')]
    closest_dates=list(closest_dates)
    closest_dates=sorted(closest_dates)
    
    rebalancing_dates=sorted(list(closest_dates))
    
    return rebalancing_dates

def rebalanced_portfolio_quantities(data,weights,investment_amount=100,frequency='Quarterly'):

    perf=data.pct_change()
    prices_dict=data.T.to_dict()
    #perf_dict=perf.T.to_dict()


    dates=sorted(list(prices_dict.keys()))    
    
    rebalancing_dates=get_rebalancing_dates(data,frequency=frequency)
    
    weights=dict(zip((data.columns),weights))
    
    
    shares={}
    portfolio={}

    for key in weights:
        
        shares[key]=weights[key]*investment_amount/prices_dict[dates[0]][key]
    
    portfolio[dates[0]]=shares
    
    i=0
    for j in range(len(dates)-1):
    
    
        if dates[j+1]>rebalancing_dates[i]:
            
            shares={}
            
            prices=prices_dict[dates[j]]
            investment_amount=0
            perf_w=0
            
            for key in weights:
                
                investment_amount+=portfolio[dates[j]][key]*prices[key]
                #perf_w+=weights[key]*perf_dict[dates[j+1]][key]
                
            for key in weights:
                shares[key]=investment_amount*weights[key]/prices[key]
    
            i=i+1
        else:
    
            shares=portfolio[dates[j]]
    
    
        portfolio[dates[j+1]]=shares
        
    quantities=pd.DataFrame(portfolio).T

    return quantities

def rebalanced_portfolio(data,weights,investment_amount=100,frequency='Quarterly'):
    
    quantities=rebalanced_portfolio_quantities(data,weights,investment_amount=investment_amount,frequency=frequency)
    
    return quantities*data

def buy_and_hold(data,weights,investment_amount=100):

    shares=weights*investment_amount/data.iloc[0]
    
    return data*shares

def buy_and_hold_pnl(data,weights,investment_amount=100):
    portfolio_value=buy_and_hold(data,weights,investment_amount=investment_amount)
    pnl=pd.DataFrame((portfolio_value-portfolio_value.iloc[0]).iloc[-1])
    pnl.columns=['PnL']

    return pnl


def rebalanced_pnl(data,weights,investment_amount=100,frequency='Quarterly'):
    portfolio_value=rebalanced_portfolio(data,weights,investment_amount=investment_amount,frequency=frequency)
    pnl=pd.DataFrame((portfolio_value-portfolio_value.iloc[0]).iloc[-1])
    pnl.columns=['PnL']

    return pnl


def rebalanced_book_cost(data,weights,investment_amount=100,frequency='Quarterly'):
    
    rebalancing_dates=get_rebalancing_dates(data,frequency=frequency)
    
    quantities=rebalanced_portfolio_quantities(data,weights,investment_amount=investment_amount,frequency=frequency)
    
    quantities_when_rebalanced=quantities.loc[rebalancing_dates]
    trading_prices=data.loc[rebalancing_dates]
    trading_prices.loc[data.index[0]]=data.loc[data.index[0]]
    trading_prices=trading_prices.sort_index()
    
    variation=quantities_when_rebalanced-quantities_when_rebalanced.shift(1)
    variation.iloc[0]=quantities.iloc[0]
    buy_variation=variation.copy()
    
    buy_variation[buy_variation< 0] = 0
    buy_variation.loc[data.index[0]]=quantities.loc[data.index[0]]
    buy_variation=buy_variation.sort_index()
    quantities_over_time=buy_variation.cumsum()
    amount_traded=quantities_over_time*trading_prices
    book_cost=(amount_traded.shift(-1)+amount_traded)/(quantities_over_time.shift(-1)+quantities_over_time).ffill()

    book_cost_history=pd.DataFrame()
    book_cost_history.index=data.index
    
    for asset in data.columns:
        
        book_cost_history[asset]=amount_traded[asset]
    
    book_cost.iloc[-1]=amount_traded.iloc[-1]/quantities_over_time.iloc[-1]
    book_cost_history=book_cost_history.ffill()
    return book_cost_history
    

    
