# Copyright (c) 2025 [Niroojane Selvam]
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:

REBALANCING_RULES = {'Weekly': 'W-FRI', 'Monthly': 'ME', 'Quarterly': 'QE', 'Yearly': 'YE'}


def get_rebalancing_dates(data,frequency='Quarterly'):
    """First date + the last available date of each calendar period.

    The last date of the data is never a rebalancing date (nothing left to
    hold), so a partial final period adds no rebalance -- the old 'nearest
    business period end' mapping always added one on the second-to-last day
    and could land on the first day of the next period when the period end
    was a holiday. Works for 7-day (crypto) calendars too.
    """
    if frequency not in REBALANCING_RULES:
        raise ValueError(f"Unknown rebalancing frequency {frequency!r} (expected one of {list(REBALANCING_RULES)})")

    index = pd.DatetimeIndex(sorted(data.index))
    period_ends = pd.Series(index, index=index).groupby(pd.Grouper(freq=REBALANCING_RULES[frequency])).last().dropna()

    dates_end = set(pd.DatetimeIndex(period_ends.values)) - {index[-1]}
    dates_end.add(index[0])

    return sorted(dates_end)


def _weights_by_asset(data, weights):
    """{asset: weight} -- a Series is matched by ticker (missing -> 0), an
    array / list by position in data.columns."""
    if isinstance(weights, pd.Series):
        weights = weights.reindex(data.columns).fillna(0.0)
    return dict(zip(data.columns, np.asarray(weights, dtype=float)))


def rebalanced_portfolio_quantities(data,weights,investment_amount=100,frequency='Quarterly'):

    dates=sorted(list(data.index))

    rebalancing_dates=get_rebalancing_dates(data,frequency=frequency)

    weights=_weights_by_asset(data, weights)
    
    
    shares={}
    portfolio={}

    for key in weights:
        
        shares[key]=weights[key]*investment_amount/data.loc[dates[0]][key]
    
    portfolio[dates[0]]=shares
    
    i=0
    for j in range(len(dates)-1):

        # i < len: after the last rebalancing date the shares are just held
        if i < len(rebalancing_dates) and dates[j+1]>rebalancing_dates[i]:
            
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
    """Quantities held each day when the weights change at every date of
    `matrix` (index = rebalancing dates, columns = assets, matched by name).

    transaction_fee is a rate on the amount traded (0.001 = 10 bps): the
    first purchase pays it on the whole investment, a rebalance only on the
    turnover, sum |target value - current value|.
    """
    matrix=matrix.reindex(columns=data.columns).fillna(0.0)

    dates=sorted(list(data.index))

    rebalancing_dates=matrix.index

    weights=dict(zip(data.columns, matrix.iloc[0].values))

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

            # fee on the traded amount only (first-order: turnover measured at the pre-fee value)
            traded = sum(abs(investment_amount * weights[key] - portfolio[dates[j]][key] * prices[key])
                         for key in weights)
            investment_amount -= transaction_fee * traded

            for key in weights:
                shares[key] = investment_amount * weights[key] / prices[key]

            i += 1
    
        else:
            shares = portfolio[dates[j]]
    
        portfolio[dates[j + 1]] = shares
        
    quantities = pd.DataFrame(portfolio).T

    
    return quantities


    
