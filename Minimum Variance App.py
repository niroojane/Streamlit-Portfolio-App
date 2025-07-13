import streamlit as st
import pandas as pd
import random
import numpy as np
import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from RiskMetrics import RiskAnalysis,create_constraint,diversification_constraint
from Price_Endpoint import *
from Stock_Data import get_close



selected_number = st.slider(
    "Number of Crypto:",
    min_value=1,
    max_value=40,
    value=20,     
    step=1           
)
tickers=get_market_cap()['Ticker'].iloc[:selected_number].to_list()
market_cap_table=get_market_cap()[['Long name','Ticker','Market Cap','Supply']].set_index('Ticker').iloc[:selected_number]
selected = st.multiselect("Select Crypto:", tickers,default=tickers)

st.dataframe(market_cap_table)

starting_date= st.date_input("Starting Date", datetime.datetime(2020, 1, 1))
dt = datetime.datetime.combine(starting_date, datetime.datetime.min.time())

@st.cache_data
def load_data(tickers,start_date=datetime.datetime(2023,1,1),today=datetime.datetime.today()):

    days=(today-start_date).days

    
    remaining=days%500
    numbers_of_table=days//500

    temp_end=start_date
    scope_prices=pd.DataFrame()
    for i in range(numbers_of_table+1):
        data=get_price(tickers,temp_end)
        temp_end=temp_end+datetime.timedelta(500)
        scope_prices=scope_prices.combine_first(data)
        
    temp_end=(today-datetime.timedelta(remaining))
    data=get_price(tickers,temp_end)
    scope_prices=scope_prices.combine_first(data)
    scope_prices=scope_prices.sort_index()
    scope_prices = scope_prices[~scope_prices.index.duplicated(keep='first')]
    scope_prices.index=pd.to_datetime(scope_prices.index)

    trx=get_close(['TRX-USD'],start=start_date.strftime("%Y-%m-%d"),end=today.strftime("%Y-%m-%d"))
    trx.index=pd.to_datetime(trx.index)
    trx = trx[~trx.index.duplicated(keep='first')]
    trx=trx.sort_index().dropna()
    trx_returns=trx.pct_change().sort_index()
    scope_prices=pd.concat([trx,scope_prices],axis=1)

    returns=np.log(1+scope_prices.pct_change(fill_method=None))
    returns.index=pd.to_datetime(returns.index)
    with_no_na=returns.columns[np.where((returns.isna().sum()<30))]
    returns_to_use=returns[with_no_na].sort_index()


    dataframe=scope_prices[with_no_na].sort_index().ffill()
    dataframe.index=pd.to_datetime(dataframe.index)
    
    returns_to_use.index=pd.to_datetime(returns_to_use.index)
    returns_to_use = returns_to_use[~returns_to_use.index.duplicated(keep='first')]
    
    return dataframe., returns_to_use


st.subheader("Constraints")

dataframe,returns_to_use=load_data(tickers=selected,start_date=dt)
data = pd.DataFrame({'Asset':[None],
'Sign':[None],
'Limit':[None]
})
drop_down_list=list(dataframe.columns)+['All']
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
            position = np.where(dataframe.columns == ticker)[0][0]
            constraint = create_constraint(sign, limit, position)
            
        constraints.extend(constraint)
            
    
except Exception as e:
    pass


if 'USDCUSDT' in returns_to_use.columns:
    
    cash=np.where(returns_to_use.columns=='USDCUSDT')[0][0]

else:

    cash=[]

constraints.extend([{'type': 'eq', 'fun': lambda weights: weights[cash]-0.00}])


month=list(sorted(set(returns_to_use.index + pd.offsets.BMonthEnd(0))))
#month_end=pd.to_datetime(mrat_wo_na.index)
month = pd.to_datetime(month)

idx1 = pd.Index(returns_to_use.iloc[:-1].index)
idx2 = pd.Index(month)
closest_dates = idx1[idx1.get_indexer(idx2, method='nearest')]

dates_end=list(closest_dates)
dates_end.insert(0,returns_to_use.index[1])
dates_end.append(returns_to_use.index[-1])
dates_end=sorted(list(set(dates_end)))
dates_end.pop(0)
dates_end=sorted(list(set(dates_end)))

results={}


for i in range(len(dates_end)-1):
    dataset=returns_to_use.loc[dates_end[i]:dates_end[i+1]]
    risk=RiskAnalysis(dataset)
    date=dataset.index[-1]
    
    optimal=risk.optimize(objective='minimum_variance',constraints=constraints)
    results[date]=np.round(optimal,6)

rolling_optimization=pd.DataFrame(results,index=dataframe.columns).T
rolling_optimization.loc[dates_end[0]]=1/len(dataframe.columns)
rolling_optimization=rolling_optimization.sort_index()

tracking={}
portfolio={}
investment_amount=1
initial_amount=investment_amount
perf=dataframe.pct_change(fill_method=None)
transaction_fee=0.005
gold_limit=0.05
weight_dict={col: 1/returns_to_use.shape[1] for col in returns_to_use.columns}


for i in range(len(dates_end)-1):
    
    print(dates_end[i],investment_amount,investment_amount/initial_amount)

    
    temp=dataframe.loc[dates_end[i]:dates_end[i+1]].copy()
    initial_price=temp.iloc[0].to_dict()

    if dates_end[i]>dates_end[0]:

        top50=rolling_optimization.loc[dates_end[i]]
        top50_dict=rolling_optimization.loc[dates_end[i]].to_dict()
        weight_dict={}

        for key in temp.columns:

            if key in top50_dict: 
                weight_dict[key]=top50_dict[key]
            else:
                weight_dict[key]=0

                
    weight_vec=np.array(list(weight_dict.values()))
    
    inital_investment_per_stock={}
    shares={}

    for col in temp.columns:
        
        weighted_perf=weight_vec*perf.loc[dates_end[i]]
        
        inital_investment_per_stock[col]=weight_dict[col]*investment_amount*(1+weighted_perf.sum())
        shares[col]=inital_investment_per_stock[col]*(1-transaction_fee)/initial_price[col]

    tracking[dates_end[i]]=(weight_dict,shares,investment_amount,initial_price)

    temp=temp*shares    
    portfolio[dates_end[i]]=temp
    investment_amount=temp.iloc[-1].sum()
    

temp=dataframe.loc[dates_end[-2]:]*shares
portfolio[dates_end[-1]]=temp

st.subheader("Portfolio Composition")


dates_options=sorted(dates_end[:-1],reverse=True)

selected_date = st.selectbox("Weights history:", dates_options,index=1)

last_weights=tracking[selected_date][0]
weights=pd.DataFrame(last_weights.values(),index=last_weights.keys(),columns=[selected_date])
st.dataframe(weights)

# last_weights=tracking[dates_end[-3]][0]
# weights=pd.DataFrame(last_weights.values(),index=last_weights.keys(),columns=['Weights Model'])

# current_positions=Binance.get_inventory().round(4)
# current_positions.columns=['Current Portfolio in USDT','Current Weights']
# amount=current_positions.loc['Total']['Current Portfolio in USDT']

# last_prices=Binance.get_price(list(last_weights.keys()))

# quantities={}

# for key in last_weights:
#     quantities[key]=amount*last_weights[key]#/last_prices[key].values[0]

# positions=pd.DataFrame(quantities.values(),index=quantities.keys(),columns=['Mark To Market Model'])
# positions=pd.concat([positions,weights],axis=1)

# condition=current_positions.index!='Total'
# portfolio_composition=pd.concat([positions,current_positions.loc[condition]],axis=1).fillna(0)
# portfolio_composition.loc['Total']=portfolio_composition.sum(axis=0)
# portfolio_composition=portfolio_composition.sort_values(by='Weights Model',ascending=False).round(4)

# st.dataframe(portfolio_composition)


st.subheader("Returns")

historical_portfolio=pd.DataFrame()
performance=pd.DataFrame()
for key in portfolio.keys():
    historical_portfolio=historical_portfolio.combine_first(portfolio[key])


performance['Fund']=historical_portfolio.sum(axis=1)
performance['Bitcoin']=dataframe['BTCUSDT']
#performance['Mantra']=dataframe['OMUSDT']
performance=performance.dropna()
performance_pct=performance.copy()
performance_pct=performance_pct.pct_change()



years=sorted(list(set(performance.index.year)))
month_year=performance.index.strftime('%Y-%m')
month_year=sorted(list(set(month_year)))

year_returns={}
for year in years:

    perf_year=performance.loc[str(year)].iloc[-1]/performance.loc[str(year)].iloc[0]-1
    year_returns[year]=perf_year

year_returns[years[-1]]=performance.loc[str(years[-1])].iloc[-2]/performance.loc[str(years[-1])].iloc[0]-1
year_returns_dataframe=pd.DataFrame(year_returns)
st.dataframe(year_returns_dataframe)

month_returns={}
for month in month_year:

    perf_year=performance.loc[str(month)].iloc[-1]/performance.loc[str(month)].iloc[0]-1
    month_returns[month]=perf_year

month_returns_dataframe=pd.DataFrame(month_returns)
st.dataframe(month_returns_dataframe)

st.subheader("Indicators")


metrics={}
metrics['Tracking Error']=(performance_pct['Fund']-performance_pct['Bitcoin']).std()*np.sqrt(252)
metrics['Fund Vol']=performance_pct['Fund'].std()*np.sqrt(252)
metrics['Bench Vol']=performance_pct['Bitcoin'].std()*np.sqrt(252)
metrics['Fund Return']=performance['Fund'].iloc[-2]/performance['Fund'].iloc[0]
metrics['Bench Return']=performance['Bitcoin'].iloc[-2]/performance['Bitcoin'].iloc[0]
metrics['Sharpe Ratio']=(1+metrics['Fund Return'])**(1/len(set(returns_to_use.index.year)))/metrics['Fund Vol']

indicators=pd.DataFrame(metrics.values(),index=metrics.keys(),columns=['Indicators'])
st.dataframe(indicators)


st.subheader("Portfolio Value Evolution")

max_value = performance_pct.index.max().strftime('%Y-%m-%d')
min_value = performance_pct.index.min().strftime('%Y-%m-%d')
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

selmin, selmax = Model

selmind = selmin.strftime('%Y-%m-%d')  # datetime to str
selmaxd = selmax.strftime('%Y-%m-%d')

mask = (performance_pct.index >= selmind) & (performance_pct.index <= selmaxd)

portfolio_returns=(1+performance_pct.loc[mask]).cumprod()*100

fig = px.line(portfolio_returns, title="Portfolio Value Evolution")
st.plotly_chart(fig)
st.dataframe((1+performance_pct.loc[mask]).cumprod()*100)




