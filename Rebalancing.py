#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def rebalanced_portfolio(data,weights,investment_amount=100,frequency='quarterly'):

    perf=data.pct_change()

    prices_dict=data.T.to_dict()
    #perf_dict=perf.T.to_dict()

    if frequency=='quarterly':
        month=list(sorted(set(data.index + pd.offsets.BQuarterEnd(0))))
    
    elif frequency=='monthly':

        month=list(sorted(set(data.index + pd.offsets.BMonthEnd(0))))

    elif frequency=='year_end':
        month=list(sorted(set(data.index + pd.offsets.BYearEnd(0))))
        
    
    month = pd.to_datetime(month)

    idx1 = pd.Index(data.iloc[:-1].index)
    idx2 = pd.Index(month)
    
    closest_dates = idx1[idx1.get_indexer(idx2, method='nearest')]
    closest_dates=list(closest_dates)
    closest_dates=sorted(closest_dates)
    
    rebalancing_dates=sorted(list(closest_dates))
    dates=sorted(list(prices_dict.keys()))    


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

    return pd.DataFrame(portfolio).T*data
   


# In[3]:


def buy_and_hold(data,weights,investment_amount=100):

    shares=weights*investment_amount/data.iloc[0]
    
    return data*shares

def compute_buy_and_hold_pnl(data,weights,investment_amount=100):
    portfolio_value=buy_and_hold(data,weights,investment_amount=investment_amount)
    pnl=pd.DataFrame((portfolio_value-portfolio_value.iloc[0]).iloc[-1])
    pnl.columns=['PnL']

    return pnl


def compute_rebalance_pnl(data,weights,investment_amount=100,frequency='quarterly'):
    portfolio_value=buy_and_hold(data,weights,investment_amount=investment_amount,frequency=frequency)
    pnl=pd.DataFrame((portfolio_value-portfolio_value.iloc[0]).iloc[-1])
    pnl.columns=['PnL']

    return pnl
    
