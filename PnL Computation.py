#!/usr/bin/env python
# coding: utf-8

# In[1]:


from binance.spot import Spot
import pandas as pd
import requests
import datetime
import numpy as np


# In[2]:


get_ipython().run_line_magic('run', '"Binance API.ipynb".ipynb')


# In[3]:


binance_api_key='E90xR7UnO2oeP3lrgC50zkqUuAqyyXs4uSlAGcqIVO9n90ELrFI4gfVI2HGGxxCj'
binance_api_secret='Inls2xYl9FqPi0zlIrucfxG81drx7w6Pv6SzYFxMNoAcwDjlCKug7zUM9zB3lF9h'
Binance=BinanceAPI(binance_api_key,binance_api_secret)


# In[4]:


def get_trade_in_usdt(trade_history):

    trade_history['Date(UTC)']=pd.to_datetime(trade_history['Date(UTC)'])
    trade_history=trade_history.set_index('Date(UTC)')

    trade_info=zip(trade_history['Market'],trade_history.index)
    trade_info=dict(enumerate(trade_info))

    trade_price={}
    for index in trade_info:

        if trade_info[index][0][-4:]=='USDT':
            ticker=trade_info[index][0]
        else:
            ticker=trade_info[index][0][-3:]+'USDT'

        price=Binance.binance_api.klines(ticker,interval='1m',startTime=int(trade_info[index][1].round(freq='min').timestamp()-60)*1000,limit=1)

        trade_price[index]=(trade_info[index][1],trade_info[index][0],price[0][4])

    price=pd.DataFrame(trade_price.values(),columns=['Time','Market','Pair Price'])
    price=pd.concat([trade_history.reset_index(),price['Pair Price']],axis=1)
    price['Price in USDT']=np.where(price['Market'].str[-4:]=='USDT',price['Price'],price['Price'].astype(float)*price['Pair Price'].astype(float))
    price['Total in USDT']=(price['Price in USDT'].astype(float))*(price['Amount'].astype(float))

    return price


# In[5]:


def get_crypto_traded(price):

    buy=price[price['Type']=='BUY'][['Market','Amount','Price in USDT']]
    traded_crypto=set(buy['Market'])
    
    crypto_list=set()
    for key in traded_crypto:

        if key[-4:]=='USDT':
            crypto=key[:-4]
        else:
            crypto=key[:-3]

        crypto_list.add(crypto)
        
    return crypto_list


# In[6]:


def get_book_cost(price):
    
    crypto_list=get_crypto_traded(price)
    
    dynamic_average_total={}
    dynamic_average_amount={}
    
    dataframe_amount={}
    dataframe_total={}


    for crypto in crypto_list:

        dataset=price[price['Market'].str[:len(crypto)]==crypto]
        index=dataset[dataset['Type']=='BUY'].index

        results_amount=list(zip(price.iloc[index]['Date(UTC)'],price.iloc[index]['Amount']))
        results_total=list(zip(price.iloc[index]['Date(UTC)'],price.iloc[index]['Total in USDT']))
        dynamic_average_total[crypto]=results_total
        dynamic_average_amount[crypto]=results_amount
            
        temp=pd.DataFrame(dynamic_average_total[crypto],columns=['Date','Total']).groupby(by='Date').sum()
        temp_amount=pd.DataFrame(dynamic_average_amount[crypto],columns=['Date','Quantities']).groupby(by='Date').sum()
        dataframe_total[crypto+'USDT']=dict(zip(temp.index,temp['Total']))
        dataframe_amount[crypto+'USDT']=dict(zip(temp_amount.index,temp_amount['Quantities']))
        
    #quantities=pd.DataFrame(dataframe_amount).sort_index().cumsum().fillna(method='ffill').fillna(0)
    #total=pd.DataFrame(dataframe_total).sort_index().cumsum().fillna(method='ffill').fillna(0)
    quantities=pd.DataFrame(dataframe_amount).sort_index().cumsum().ffill().fillna(0)
    total=pd.DataFrame(dataframe_total).sort_index().cumsum().ffill().fillna(0)
    
    book_cost=(total.shift(-1)+total)/(quantities.shift(-1)+quantities)
    book_cost=book_cost.fillna(0)
    book_cost.iloc[-1]=total.iloc[-1]/quantities.iloc[-1]
    
    return book_cost


# In[7]:


def get_pnl(book_cost,price):

    positions_history={}
    transaction_type={}
    results={}
    profit_and_loss={}
    pnl_per_crypto={}

    crypto_list=get_crypto_traded(price)
    for crypto in crypto_list:

        dataset=price[price['Market'].str[:len(crypto)]==crypto]

        grouped=dataset.groupby(by='Date(UTC)').sum()
        positions_history[crypto]=dict(zip(grouped.index,grouped['Amount'].astype(float)))
        transaction_type[crypto]=list(zip(dataset['Date(UTC)'],dataset['Type']))

        temp=price[price['Market'].str[:len(crypto)]==crypto].copy()
        temp['Flows']=np.where(temp['Type']=='SELL',-temp['Amount'],temp['Amount'])
        temp['Flows'].sum()
        temp=temp.set_index('Date(UTC)').sort_index()
        temp['Cost']=book_cost[crypto+'USDT']
        #temp['Cost']=temp['Cost'].fillna(method='ffill')
        temp['Cost']=temp['Cost'].ffill()
        temp[crypto]=np.where(temp['Type']=='SELL',(temp['Cost']-temp['Price in USDT'])*temp['Flows'],0)
        
        profit_and_loss[crypto]=temp[crypto]

        pnl_per_crypto[crypto+'USDT']=profit_and_loss[crypto].sum()
        
    realized_pnl=pd.DataFrame(pnl_per_crypto.values(),index=pnl_per_crypto.keys(),columns=['Realized PnL'])
    
    return realized_pnl,profit_and_loss


# In[22]:


def get_historical_positions(price):

    crypto_list=get_crypto_traded(price)
    quantities={}
    dataframe_total={}


    for crypto in crypto_list:

        dataset=price[price['Market'].str[:len(crypto)]==crypto].copy()
        dataset['Quantities']=np.where(dataset['Type']=="SELL",-dataset['Amount'].astype(float),dataset['Amount'].astype(float))

        quantities[crypto]=list(zip(dataset['Date(UTC)'],dataset['Quantities']))
        temp=pd.DataFrame(quantities[crypto],columns=['Date','Quantities']).groupby(by='Date').sum()
        dataframe_total[crypto]=dict(zip(temp.index,temp['Quantities']))

    historical_positions=pd.DataFrame(dataframe_total).sort_index()
    historical_positions=historical_positions.groupby(historical_positions.index).sum()
    historical_positions.index=historical_positions.index.round('d')
    historical_positions=historical_positions.groupby(historical_positions.index).sum().cumsum()

    return historical_positions


# In[ ]:




