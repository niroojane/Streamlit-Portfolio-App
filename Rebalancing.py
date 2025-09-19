# Copyright (c) 2025 [Niroojane Selvam]
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


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
    
    dates_end=list(closest_dates)
    dates_end.insert(0,data.index[0])
    # dates_end.append(data.index[-1])
    dates_end=sorted(list(set(dates_end)))
    
    rebalancing_dates=sorted(list(dates_end))
    
    return rebalancing_dates

def rebalanced_portfolio_quantities(data,weights,investment_amount=100,frequency='Quarterly'):

    perf=data.pct_change()


    dates=sorted(list(prices_dict.keys()))    
    
    rebalancing_dates=get_rebalancing_dates(data,frequency=frequency)
    
    weights=dict(zip((data.columns),weights))
    
    
    shares={}
    portfolio={}

    for key in weights:
        
        shares[key]=weights[key]*investment_amount/data.loc[dates[0]][key]
    
    portfolio[dates[0]]=shares
    
    i=0
    for j in range(len(dates)-1):
    
    
        if dates[j+1]>rebalancing_dates[i]:
            
            shares={}
            
            prices=data.loc[dates[j]]
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

def buy_and_hold_contribution(data,weights,investment_amount=100):
    portfolio_value=buy_and_hold(data,weights,investment_amount=investment_amount)
    pnl=pd.DataFrame((portfolio_value-portfolio_value.iloc[0]).iloc[-1])
    pnl.columns=['PnL']

    return pnl


def rebalanced_contribution(data,weights,investment_amount=100,frequency='Quarterly'):
    portfolio_value=rebalanced_portfolio(data,weights,investment_amount=investment_amount,frequency=frequency)
    pnl=pd.DataFrame((portfolio_value-portfolio_value.iloc[0]).iloc[-1])
    pnl.columns=['PnL']

    return pnl


def rebalanced_book_cost(data,quantities,investment_amount=100):

    prices_array = data.to_numpy()
    quantities_array = quantities.to_numpy()
    
    row, col = quantities_array.shape
    cumulative_quantities = np.zeros(quantities_array.shape)
    trading_prices = np.zeros(prices_array.shape)
    amount_traded = np.zeros(prices_array.shape)
    
    for i in range(row):
        if i > 0:
            previous_quantities = quantities_array[i - 1]
            current_quantities = quantities_array[i]
    
            for j in range(col):
                delta = current_quantities[j] - previous_quantities[j]
    
                if delta != 0:
                    if delta > 0:
                        cumulative_quantities[i, j] = cumulative_quantities[i - 1, j] + delta
                        trading_prices[i, j] = prices_array[i - 1, j]
                    else:
                        cumulative_quantities[i, j] = cumulative_quantities[i - 1, j] + delta
                        trading_prices[i, j] = trading_prices[i - 1, j]
    
                    amount_traded[i, j] = delta * prices_array[i - 1, j] + amount_traded[i - 1, j]
                else:
                    cumulative_quantities[i, j] = cumulative_quantities[i - 1, j]
                    trading_prices[i, j] = trading_prices[i - 1, j]
                    amount_traded[i, j] = amount_traded[i - 1, j]
    
        else: 
            cumulative_quantities[0] = quantities_array[0]
            trading_prices[0] = prices_array[0]
            amount_traded[0] = trading_prices[0] * quantities_array[0]
    
    cost = pd.DataFrame(amount_traded, index=data.index, columns=data.columns)

    return cost

def rebalanced_dynamic_quantities(data,matrix,investment_amount=100,transaction_fee=0.0):
    
    transaction_fee=0.0
    investment_amount=100
    weights=matrix.iloc[0].values
    perf=data.pct_change()
    
    dates=sorted(list(data.index))    
    
    rebalancing_dates=matrix.index
    
    weights=dict(zip((data.columns),weights))
    
    shares={}
    portfolio={}
    
    for key in weights:
    
        shares[key]=weights[key]*investment_amount*(1-transaction_fee)/data.loc[dates[0]][key]
    
    portfolio[dates[0]]=shares
    
    i = 0
    for j in range(len(dates) - 1):
    
        if i < len(rebalancing_dates) and dates[j + 1] > rebalancing_dates[i]:
    
            updated_allocation = matrix.loc[rebalancing_dates[i]].values
            weights = dict(zip(data.columns, updated_allocation))
    
            shares = {}
    
            prices = data.loc[dates[j]]
            investment_amount = 0
    
            for key in weights:
                investment_amount += portfolio[dates[j]][key] * prices[key]
    
            for key in weights:
                shares[key] = investment_amount * weights[key] * (1 - transaction_fee) / prices[key]
    
            i += 1
    
        else:
            shares = portfolio[dates[j]]
    
        portfolio[dates[j + 1]] = shares
        
    quantities = pd.DataFrame(portfolio).T

    
    return quantities


    
