import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, chi2,gumbel_l
from RiskMetrics import RiskAnalysis,diversification_constraint, create_constraint
from Rebalancing import rebalanced_portfolio , buy_and_hold
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

allocation_df = pd.DataFrame()
constraint_data=[]
constraints=[]
constraint_table=pd.DataFrame(columns=["Asset", "Sign", "Limit"])

drop_down_list = []
constraints_options = []

def load_excel(file):
    global allocation_df,prices,returns,portfolio,drop_down_list

    try:
        prices = pd.read_excel(file.name, index_col=0, parse_dates=True)

        if prices.empty:
            return "Error: The file contains no data."

        returns = prices.pct_change()

        portfolio = RiskAnalysis(returns)
        optimized_weights = portfolio.optimize(objective="sharpe_ratio")
        
        allocation_df = pd.DataFrame(
            {"Asset": returns.columns, "Optimal Portfolio": optimized_weights}
        ).set_index("Asset").T 
        
        drop_down_list = list(prices.columns) + ['All']+[None]
        constraints_options = ["=", "≥", "≤"]
        
        return "File uploaded successfully!",gr.update(choices=drop_down_list)


    except Exception as e:
        return f"Error: {str(e)}"

def add_allocation(new_allocation):
    
    global allocation_df, benchmarks,optimzed_weights_constraints
    
    optimzed_weights_constraints=portfolio.optimize(objective="sharpe_ratio",constraints=constraints)
    allocation_df.loc['Constrained Optimal Portfolio'] = optimzed_weights_constraints
    
    if not new_allocation or len(new_allocation.strip()) == 0:
        new_allocation = '1.0'+',0.0' * (len(returns.columns) - 1) 

    try:
        new_allocation = [float(x) for x in new_allocation.split(',')]

        if len(new_allocation) != len(allocation_df.columns):
            return "Error: Number of allocation values does not match the number of assets.", allocation_df

        total = sum(new_allocation)
        if total != 0:
            new_allocation = [x / total for x in new_allocation]

        allocation_df.loc["Allocation " + str(len(allocation_df)-1)] = new_allocation

        benchmarks = list(allocation_df.index)

        return allocation_df.reset_index().rename(columns={'index': 'Allocation'}).round(4), gr.update(choices=benchmarks, value=None)
    
    except Exception as e:
        return allocation_df.reset_index().rename(columns={'index': 'Allocation'}).round(4), gr.update(choices=benchmarks, value=None)

def submit(value1, value2, value3):
    global constraint_table, constraints

    new_row = {"Asset": value1, "Sign": value2, "Limit": value3}
    constraint_table = pd.concat([constraint_table, pd.DataFrame([new_row])], ignore_index=True)

    constraint_matrix = pd.DataFrame(constraint_table).to_numpy()
    constraints = []
    dico_map = {'=': 'eq', '≥': 'ineq', '≤': 'ineq'}

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
            elif ticker not in prices.columns:
                continue  
            else:
                position = np.where(prices.columns == ticker)[0][0]
                constraint = create_constraint(sign, limit, position)
        
            constraints.extend(constraint)

    except Exception as e:
        pass

    return constraint_table
        
def reset_constraints():
    
    global constraint_table, constraints
    constraint_table=pd.DataFrame(columns=["Asset", "Sign", "Limit"])
    constraints = []
    return constraint_table
    
def clear_allocation():
    
    global allocation_df
    
    optimized_weights = portfolio.optimize(objective="sharpe_ratio")
    optimzed_weights_constraints=portfolio.optimize(objective="sharpe_ratio",constraints=constraints)

    allocation_df = pd.DataFrame(
        {"Asset": returns.columns, "Optimal Portfolio": optimized_weights,
         "Constrained Optimal Portfolio":optimzed_weights_constraints}
    ).set_index("Asset").T 
    
    
    return allocation_df.reset_index().rename(columns={'index': 'Allocation'}).round(4)
    
def load_choices():
    return benchmarks
    
def get_asset_returns():

    if prices is None:
        return "No file uploaded yet."
    
    ret = prices.iloc[-1] / prices.iloc[0] - 1
    ytd = (1 + ret) ** (365 / (prices.index[-1] - prices.index[0]).days) - 1

    year_start = datetime(prices.index[-1].year, 1, 1)
    if year_start not in prices.index:
        year_start = prices.index[prices.index >= year_start][0]

    ret_ytd = prices.loc[year_start:].iloc[-1] / prices.loc[year_start:].iloc[0] - 1

    perfs = pd.concat([ret, ret_ytd, ytd], axis=1)
    perfs.columns = [
        'Returns since ' + prices.index[0].strftime("%Y-%m-%d"),
        'Returns since ' + year_start.strftime("%Y-%m-%d"),
        'Annualized Returns'
    ]
    
    return perfs.T.reset_index().rename(columns={'index': 'Returns'}).round(4)  # Transpose for better readability

def get_asset_risk():

    dates_drawdown=((prices-prices.cummax())/prices.cummax()).idxmin().dt.date
    monthly_vol=prices.resample('ME').last().iloc[-50:].pct_change().std()*np.sqrt(12)

    drawdown=pd.DataFrame((((prices-prices.cummax()))/prices.cummax()).min())
    Q=0.05
    intervals=np.arange(Q, 1, 0.0005, dtype=float)
    cvar=monthly_vol*norm(loc =0 , scale = 1).ppf(1-intervals).mean()/0.05
    vol=prices.pct_change().iloc[-360:].std()*np.sqrt(260)

    risk=pd.concat([vol,monthly_vol,cvar,drawdown,dates_drawdown],axis=1).round(4)
    risk.columns=['Annualized Volatility (daily)','Annualized Volatility (Monthly)','CVar Parametric '+str(int((1-Q)*100))+'%','Max Drawdown','Date of Max Drawdown']
    
    return risk.T.reset_index().rename(columns={'index': 'Risks'}).round(4)

def asset_metrics():

    return get_asset_returns(),get_asset_risk()

def get_expected_metrics():
    
    allocation_dict={}
    for idx in allocation_df.index:
        allocation_dict[idx]=allocation_df.loc[idx].to_numpy()
    
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

    return indicators.T.reset_index().rename(columns={'index': 'Indicators'}).round(4)


def rebalanced_time_series(frequency='Monthly'):
    
    global portfolio_returns
    portfolio_returns=pd.DataFrame()

    for key in allocation_df.index:
        portfolio_returns['Buy and Hold '+key]=buy_and_hold(prices, allocation_df.loc[key]).sum(axis=1)
        portfolio_returns['Rebalanced '+key]=rebalanced_portfolio(prices, allocation_df.loc[key],frequency=frequency).sum(axis=1)

    portfolio_returns.index.name='Date'

    return portfolio_returns.rename(columns={'index': 'Date'}).round(4)

def rebalanced_metrics():

    global perfs
    
    ret=portfolio_returns.iloc[-1]/portfolio_returns.iloc[0]-1
    ytd=(1+ret)**(365/(portfolio_returns.index[-1]-portfolio_returns.index[0]).days)-1
    ret_ytd=portfolio_returns.loc[datetime(max(portfolio_returns.index.year),1,1):].iloc[-1]/portfolio_returns.loc[datetime(max(portfolio_returns.index.year),1,1):].iloc[0]-1

    perfs=pd.concat([ret,ret_ytd,ytd],axis=1)
    perfs.columns=['Returns since '+ pd.to_datetime(portfolio_returns.index[0], format='%Y-%d-%m').strftime("%Y-%m-%d"),
              'Returns since '+datetime(max(portfolio_returns.index.year), 1, 1).strftime("%Y-%m-%d"),
              'Annualized Returns']
    
    return perfs.T.reset_index().rename(columns={'index': 'Portfolio Returns'}).round(4)

def get_portfolio_risk():

    allocation_dict={}
        
    for idx in allocation_df.index:
        allocation_dict[idx]=allocation_df.loc[idx].to_numpy()


    tracking_error_daily={}
    tracking_error_monthly={}
    monthly_returns=prices.resample('ME').last().pct_change()


    for key in allocation_dict:
        tracking_error_daily['Buy and Hold '+key]=RiskAnalysis(returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(260)
        tracking_error_daily['Rebalanced '+key]=RiskAnalysis(returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(260)
        tracking_error_monthly['Buy and Hold '+key]=RiskAnalysis(monthly_returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(12)
        tracking_error_monthly['Rebalanced '+key]=RiskAnalysis(monthly_returns).variance(allocation_dict[key]-allocation_dict[benchmark])/np.sqrt(252)*np.sqrt(12)

    tracking_error_daily=pd.DataFrame(tracking_error_daily.values(),index=tracking_error_daily.keys(),columns=['Tracking Error (daily)'])
    tracking_error_monthly=pd.DataFrame(tracking_error_monthly.values(),index=tracking_error_monthly.keys(),columns=['Tracking Error (Monthly)'])

    dates_drawdown=((portfolio_returns-portfolio_returns.cummax())/portfolio_returns.cummax()).idxmin().dt.date
    monthly_vol=portfolio_returns.resample('ME').last().iloc[-50:].pct_change().std()*np.sqrt(12)

    drawdown=pd.DataFrame((((portfolio_returns-portfolio_returns.cummax()))/portfolio_returns.cummax()).min())
    Q=0.05
    intervals=np.arange(Q, 1, 0.0005, dtype=float)
    cvar=monthly_vol*norm(loc =0 , scale = 1).ppf(1-intervals).mean()/0.05
    vol=portfolio_returns.pct_change().iloc[-360:].std()*np.sqrt(260)

    risk=pd.concat([vol,tracking_error_daily,monthly_vol,tracking_error_monthly,cvar,drawdown,dates_drawdown],axis=1).round(4)
    risk.columns=['Annualized Volatility (daily)','TEV (daily)',
                  'Annualized Volatility (Monthly)','TEV (Monthly)',
                  'CVar Parametric '+str(int((1-Q)*100))+'%',
                  'Max Drawdown','Date of Max Drawdown']
    
    return risk.T.reset_index().rename(columns={'index': 'Indicators'}).round(4)

def get_portfolio_evolution(frequency):
    
    rebalanced_time_series(frequency=frequency)
    fig = px.line(portfolio_returns, title="Portfolio Value Evolution",color_discrete_sequence = px.colors.sequential.Sunsetdark)
    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white") 
    fig.update_traces(textfont=dict(family="Arial Narrow"))

    return get_expected_metrics(),rebalanced_metrics(),get_portfolio_risk(),fig,portfolio_returns.reset_index().rename(columns={'index': 'Portfolio Returns'}).round(4)



def current_allocation():
    global optimal_results,optimal_results_dataframe

    optimal_results={}

    # optimized_weights_constraint = portfolio.optimize(objective="sharpe_ratio",constraints=constraints)
    # optimized_weights = portfolio.optimize(objective="sharpe_ratio")
    
    # optimal_results['Current Optimal Portfolio']=optimized_weights.tolist()
    # optimal_results['Current Optimal Constrained Portfolio']=optimized_weights_constraint.tolist()

    for idx in allocation_df.index:
        optimal_results[idx]=allocation_df.loc[idx].tolist()

    optimal_results_dataframe=pd.DataFrame(optimal_results,index=prices.columns).T


def add_allocation_optimized(new_allocation):

    # global optimal_results,optimal_results_dataframe
        
    # # optimized_weights_constraint = portfolio.optimize(objective="sharpe_ratio",constraints=constraints)
    # # optimized_weights = portfolio.optimize(objective="sharpe_ratio")
        
    # # optimal_results['Current Optimal Portfolio']=optimized_weights.tolist()
    # # optimal_results['Current Optimal Constrained Portfolio']=optimized_weights_constraint.tolist()

    # for idx in optimal_results_dataframe.index:
    #     optimal_results[idx]=optimal_results_dataframe.loc[idx].tolist()

    # optimal_results_dataframe=pd.DataFrame(optimal_results,index=prices.columns).T
    
    if not new_allocation or len(new_allocation.strip()) == 0:
        new_allocation = '1.0'+',0.0' * (len(returns.columns) - 1)
        
    try:
        
        new_allocation = [float(x) for x in new_allocation.split(',')]

        if len(new_allocation) != len(optimal_results_dataframe.columns):
            return "Error: Number of allocation values does not match the number of assets.", optimal_results_dataframe

        total = sum(new_allocation)
        if total != 0:
            new_allocation = [x / total for x in new_allocation]

        optimal_results_dataframe.loc["Allocation " + str(len(optimal_results_dataframe)-1)] = new_allocation

        return optimal_results_dataframe.reset_index().rename(columns={'index': 'Allocation'}).round(4)
    
    except Exception as e:
        
        return optimal_results_dataframe.reset_index().rename(columns={'index': 'Allocation'}).round(4)

def current_metrics():
    
    global weight_matrix,metrics
    
    weight_matrix={}
    
    for idx in optimal_results_dataframe.index:
        weight_matrix[idx]=optimal_results_dataframe.loc[idx].to_numpy()
    
        metrics={}
        metrics['Returns']={}
        metrics['Volatility']={}
        metrics['Sharpe Ratio']={}
    
    for key in weight_matrix:
    
        metrics['Returns'][key]=(np.round(portfolio.performance(weight_matrix[key]), 4))
        metrics['Volatility'][key]=(np.round(portfolio.variance(weight_matrix[key]), 4))
        metrics['Sharpe Ratio'][key]=np.round(metrics['Returns'][key]/metrics['Volatility'][key],4)

    indicators = pd.DataFrame(metrics,index=weight_matrix.keys()).T
    
    return indicators.reset_index().rename(columns={'index': 'Metrics'}).round(4)

def efficient_frontier_fig():
    
    frontier_weights, frontier_returns, frontier_risks, frontier_sharpe_ratio = portfolio.efficient_frontier()
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
        


    return fig

def plot_corr_heatmap():
    
    if returns is None:
        return "No file uploaded yet."
    
    fig = px.imshow(returns.corr().round(2), color_continuous_scale='blues', text_auto=True, aspect="auto")
    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")  
    fig.update_traces(xgap=2, ygap=2)
    fig.update_traces(textfont=dict(family="Arial Narrow", size=15))

    return fig

def get_frontier_metrics():

    ptf_metrics=current_metrics()
    fig=efficient_frontier_fig()
    
    return ptf_metrics,fig
    

with gr.Blocks(css="* { font-family: 'Arial Narrow', sans-serif; }") as app:
        
    with gr.Tab("Portfolio Construction"):

        gr.Markdown("# Upload Data")  
        file_upload = gr.File(label="Upload Excel File")
        upload_status = gr.Textbox(label="Upload Status")
        
        
        gr.Markdown("# Asset Metrics")
        
        asset_metrics_view = gr.Interface(fn=asset_metrics,
                                          inputs=None,
                                          outputs=[gr.DataFrame(label="Asset Returns"),gr.DataFrame(label="Asset Risk")]
                    )
        

        
        gr.Markdown("# Portfolio Allocation")

        
        with gr.Column():
            
            constraints_table = gr.Dataframe(headers=["Asset", "Sign", "Limit"], interactive=False)
            
            asset_dropdown = gr.Dropdown(choices=drop_down_list, label="Asset")
            sign_dropdown = gr.Dropdown(choices=["=", "≥", "≤"], label="Sign")
            limit_input = gr.Number(label="Limit")
            
            file_upload.change(load_excel, inputs=file_upload, outputs=[upload_status, asset_dropdown])
            submit_button = gr.Button("Add Constraint")
            reset_button = gr.Button("Reset Constraints")
            

            submit_button.click(
                submit,
                inputs=[asset_dropdown, sign_dropdown, limit_input],
                outputs=[constraints_table]
            )
            
            reset_button.click(
                reset_constraints,
                outputs=[constraints_table]
            )
            
            allocation_table_view = gr.Dataframe(label="Portfolio Allocation", interactive=True)
            new_allocation_input = gr.Textbox(label="Enter Allocation (comma separated)")
        
            add_button = gr.Button("Add Allocation Row")
            
            clear_allocation_button = gr.Button("Reset Allocation")
            clear_allocation_button.click(fn=clear_allocation, inputs=[], outputs=[allocation_table_view])

            
            dropdown = gr.Dropdown(choices=[], label="Select Benchmark")
            

            add_button.click(fn=add_allocation, inputs=new_allocation_input, outputs=[allocation_table_view,dropdown])
            
            new_allocation_input.submit(fn=add_allocation, inputs=new_allocation_input, outputs=[allocation_table_view,dropdown])
            
            output = gr.Textbox(visible=False)
            load_button = gr.Button("Load List",visible=False)
        
            load_button.click(fn=load_choices, outputs=dropdown)
        
            def show_selection(choice):
                global benchmark
                benchmark=choice
                return f"You selected: {benchmark}"
            
            dropdown.change(fn=show_selection, inputs=dropdown, outputs=output)

        

            gr.Markdown("# Portfolio Metrics")
            
            rebalancing_frequency = gr.Dropdown(choices=['Monthly','Quarterly','Yearly'], label="Select Rebalancing Frequency",value='Quarterly')

            get_metrics_button = gr.Button("Get Portfolio Metrics")
            metrics_table = gr.DataFrame(label="Portfolio Expected Metrics")
            returns_table = gr.DataFrame(label="Portfolio Returns")
            risk_table = gr.DataFrame(label="Portfolio Risk")
            time_series_plot = gr.Plot(label="Portfolio Evolution")
            time_series_data = gr.DataFrame(label="Portfolio Time Series")
    
            get_metrics_button.click(
                fn=get_portfolio_evolution,
                inputs=[rebalancing_frequency],
                outputs=[metrics_table, returns_table, risk_table, time_series_plot, time_series_data])

    with gr.Tab("Efficient Frontier"):
        with gr.Column():

            gr.Markdown("# Portfolio Allocation")
            


            allocation_table_current = gr.Dataframe(label="Portfolio Allocation", interactive=True)
            
            new_allocation_input_current = gr.Textbox(label="Enter Allocation (comma separated)")
            
            init_opt_btn = gr.Button("Get Previous Allocation")
            init_opt_btn.click(fn=current_allocation, inputs=[], outputs=[])

            add_button_optimized = gr.Button("Add New Allocation")
            add_button_optimized.click(fn=add_allocation_optimized, inputs=new_allocation_input_current, outputs=[allocation_table_current])
    
            new_allocation_input_current.submit(fn=add_allocation_optimized, inputs=new_allocation_input_current, outputs=[allocation_table_current])

            gr.Markdown("# Efficient Frontier")
            get_portfolio_evolution_graph=gr.Interface(fn=get_frontier_metrics,
                                                       inputs=None,
                                                       outputs=[gr.DataFrame(label="Portfolio Expected Metrics")
                                                                ,gr.Plot(label='Efficient Friontier')])

            gr.Markdown("# Correlation Matrix")
            
            get_corr_matrix=gr.Interface(fn=plot_corr_heatmap,inputs=None,outputs=gr.Plot(label='Correlation Matrix'))
         

    
app.launch()