#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import random
import numpy as np
from scipy.stats import norm, chi2,gumbel_l
import scipy.optimize as sco
from scipy.optimize import minimize
import datetime
from statsmodels.stats.correlation_tools import cov_nearest


# # General Functions

# In[2]:


def halton_sequences(number,base=2):
    
    #Generate Halton sequences
    
    inv_base=1/base
    
    i=number
    halton=0
    
    while i>0:
        
        digit = i%base
        halton=halton + digit*inv_base
        i=(i-digit)/base
        inv_base=inv_base/base
        
    return halton

def generate_halton(iterations,dimensions=1,base=2):
    
    #Generate a Halton Sequences at basis k , then shuffles it
    
    rng = np.random.default_rng()
    matrix=[]
    haltons=[]
    
    for i in range(iterations):
        halton=halton_sequences(i,base=base)
        haltons.append(halton)
    
    for dim in range(dimensions):
        
        matrix.append(haltons)
    
    matrix = rng.permuted(matrix, axis=1)
    return matrix

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def near_psd(x, epsilon=0):
    
    #Calculate the nearest positive semi definite matrix

    if min(np.linalg.eigvals(x))> epsilon:
        return x

    n = x.shape[0]
    var_list = np.array(np.sqrt(np.diag(x)))
    y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])

    eigval, eigvec = np.linalg.eig(y)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    near_corr = B*B.T    

    near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])
    return near_cov


# In[3]:


def performance(perf,weights):
    
    #Calculate the performance of a portfolio on a daily basis
    
    return np.dot(perf,weights)


# In[ ]:


def rolling_var(returns,weights,window=30,Q=1):

    #This function will return the rolling VaR on a x-days window following historical,parametric and multivariate model

    value_at_risk=pd.DataFrame()

    mean=returns.rolling(window).mean().dropna()
    cov=returns.rolling(window).cov().dropna()
    corr=returns.rolling(window).corr().dropna()
    std=returns.rolling(window).std().dropna()

    index=sorted(tuple(set(cov.index.get_level_values(0))))


    var={}

    for date in index:
            
            multivariate_var=performance(np.random.multivariate_normal(mean.loc[date],cov.loc[date],10000),weights) 
            var[date]=np.percentile(multivariate_var,Q)


    var=pd.DataFrame(var.values(),index=var.keys())

    portfolio=Portfolio(returns).portfolio(weights)

    value_at_risk['Historical']=portfolio.rolling(window=window).apply(lambda x:np.percentile(x,Q))
    value_at_risk['Parametric']=portfolio.rolling(window=window).std()*norm(loc =0 , scale = 1).ppf(Q/100)
    value_at_risk['Multivariate']=var
    value_at_risk['Portfolio']=portfolio

    return value_at_risk.dropna()

def kupiec_test(rolling_var,Q=5):

    number_obs=rolling_var.shape[0]
    confidence=Q/100

    ret=(1+rolling_var['Portfolio']).cumprod()
    return_mean=(ret.iloc[-1])**(1/number_obs)-1

    stats={}

    stats['Proportion of failure']=[]
    stats['Kupiec Stat']=[]
    stats['P-value']=[]
    stats['Model']=[]

    for col in rolling_var.columns:

        if col=='Portfolio':

            continue

        else:

            number_violation=np.sum(np.where(rolling_var[col]>rolling_var['Portfolio'],1,0))
            number_non_violation=number_obs-number_violation
            proportion_violation=number_violation/number_obs
            proportion_non_violation=1-proportion_violation

            kupiec=2*np.log((proportion_non_violation/(1-confidence))**(number_non_violation)*
                                (proportion_violation/confidence)**number_violation)

            p_value=1-chi2.cdf(kupiec,1)

        stats['Kupiec Stat'].append(kupiec)
        stats['P-value'].append(p_value)
        stats['Proportion of failure'].append(proportion_violation)
        stats['Model'].append(col)

    stats=pd.DataFrame(stats.values(),index=stats.keys(),columns=stats['Model'])
    stats=stats.drop(stats.index[3])

    return stats


# In[ ]:


def create_constraint(sign,limit,position):
    
    dico_map = {'=': 'eq', '≥': 'ineq', '≤': 'ineq'}

    if sign=='≤' :
        constraints=[{'type': dico_map[sign], 'fun': lambda weights: limit-weights[position]}]
    elif sign=='≥' :
    
        constraints=[{'type': dico_map[sign], 'fun': lambda weights: weights[position]-limit}]
    else:
        constraints=[{'type': dico_map[sign], 'fun': lambda weights: weights[position]-limit}]

    return constraints

def diversification_constraint(sign,limit):
    
    dico_map = {'=': 'eq', '≥': 'ineq', '≤': 'ineq'}

    if sign=='≤' :
        constraints=[{'type': dico_map[sign], 'fun': lambda weights: limit-weights}]
    elif sign=='≥' :
    
        constraints=[{'type': dico_map[sign], 'fun': lambda weights: weights-limit}]
    else:
        constraints=[{'type': dico_map[sign], 'fun': lambda weights: weights-limit}]

    return constraints


# ## Portfolio Construction

# In[2]:


class Portfolio:
    
    #This class allows the user to calculate various metrics of a portfolio
    #and also allows to optmize the portfolio with various constraints
    
    def __init__(self,returns):
        
        self.returns=returns
        
    def inventory(self,weights):

        dico_ptf=dict(zip(self.returns.columns,weights))

        inventory=pd.DataFrame(dico_ptf.values(),index=dico_ptf.keys(),columns=['Weights'])
        inventory=inventory.loc[(inventory!=0).any(axis=1)].sort_values(by='Weights',ascending=False)
        
        return inventory

    def portfolio(self,weights):
            
        portfolio=pd.DataFrame()
        portfolio['Portfolio']=np.sum(weights*self.returns,axis=1)
        
        return portfolio
    
    def evolution(self,weights):
        
        portfolio=self.portfolio(weights)
        evolution=(1+portfolio).cumprod()*100
        
        return evolution
    
    def performance(self,weights):
        performance=np.sum(self.returns*weights,axis=1).mean()*252
        #performance=(1+np.sum(returns_to_use*weights,axis=1).mean())**252-1
        return performance
    
    def variance(self,weights):
        variance=np.sqrt(np.dot(weights.T,np.dot(self.returns.cov(),weights)))*np.sqrt(252)
        return variance
    
    def sharpe_ratio(weights):
            return self.performance(weights)/self.variance(weights)

    def optimize(self,objective='minimum_variance',constraints=False):
        
            
        def sum_equal_one(weight):
            return np.sum(weight) - 1   
        
        def sharpe_ratio(weights):
            return - self.performance(weights)/self.variance(weights)
        
        def variance(weights):
            variance=np.sqrt(np.dot(weights.T,np.dot(self.returns.cov(),weights)))*np.sqrt(252)
            return variance
        
        n_assets = len(self.returns.columns)
        weight = np.array([1 / n_assets] * n_assets)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        if not constraints:
            
            constraints = [{'type': 'eq', 'fun': sum_equal_one}]
        
        else:
            
            constraints=[{'type': 'eq', 'fun': sum_equal_one}]+constraints
        
        if objective=='minimum_variance':

            optimum_weights = sco.minimize(variance, weight, method='SLSQP', bounds=bounds, constraints=constraints)
        
        elif objective=='sharpe_ratio':
            
            optimum_weights = sco.minimize(sharpe_ratio, weight, method='SLSQP', bounds=bounds, constraints=constraints)
            
        else:
            
            print("Objective function undefined")
            
            
        return optimum_weights.x
    
    def efficient_frontier(self,constraints=False,points=100):

        num_assets = self.returns.shape[1]

        def portfolio_return(weights):
            return weights @ self.returns.mean()*252

        def sum_equal_one(weight):
            return np.sum(weight) - 1 

        def portfolio_risk(weights):
            return np.sqrt(weights @ self.returns.cov() @ weights) * np.sqrt(252)

        def objective(weights):
            return portfolio_risk(weights)


        if not constraints:

            constraints = [{'type': 'eq', 'fun': sum_equal_one}]

        else:

            constraints=[{'type': 'eq', 'fun': sum_equal_one}]+constraints


        bounds = [(0, 1) for _ in range(num_assets)]
        frontier_weights = []
        frontier_returns = []
        frontier_risks = []
        frontier_sharpe_ratio=[]
        mus = np.linspace(min(self.returns.mean()*252), max(self.returns.mean()*252), points)

        for mu in mus:
            target_return_constraint = {'type': 'eq', 'fun': lambda weights, mu=mu: portfolio_return(weights) - mu}
            result = minimize(
                objective,
                x0=np.ones(num_assets) / num_assets,
                method='SLSQP',
                bounds=bounds,

                constraints=constraints + [target_return_constraint]
            )
            if result.success:
                weights = result.x
                frontier_weights.append(weights)
                frontier_returns.append(portfolio_return(weights))
                frontier_risks.append(portfolio_risk(weights))
                frontier_sharpe_ratio.append(portfolio_return(weights)/portfolio_risk(weights))

        return frontier_weights, frontier_returns, frontier_risks,frontier_sharpe_ratio
    
    def black_Litterman(self,P,Q,weights,risk_aversion,tau=0.025):
        
        implied_returns=risk_aversion*self.returns.cov().dot(weights).squeeze()
        omega=np.diag(np.diag(P.dot(tau*self.returns.cov()).dot(P.T)))
        sigma_scaled=self.returns.cov()*tau
        BL_returns= implied_returns + sigma_scaled.dot(P.T).dot(np.linalg.inv(P.dot(sigma_scaled).dot(P.T))+omega).dot(Q-P.dot(implied_returns))
        inv_cov=np.linalg.inv(self.returns.cov())
        BL_weights=inv_cov.dot(BL_returns)
        BL_weights=BL_weights/BL_weights.sum()
        
        return BL_weights,BL_returns


# ## Risk Analysis

# In[6]:


class RiskAnalysis(Portfolio):
    
    
    #This class is used to assess various risk of a portfolio such as Market Risk, VaR
    # and to know which asset could possibly contribute to it
    
    def __init__(self,returns):
        
        self.returns=returns
        super().__init__(returns=returns)
        

    def historical_var(self,weights,last_days=False,Q=5):
        
        #Return Historical VaR on the Past x days at Q confidence interval

        performance=super().portfolio(weights)
        
        if last_days:
            performance=performance[-last_days:]
            
        var=np.percentile(performance,Q)
        cvar=performance[performance<var].mean().values[0]
        
        return var,cvar
    
    
    def parametric_var(self,weights,Q=0.95,stress_factor=1):
        
        #Return parametric VaR, where assets follows a Normal Distribution
        
        intervals=np.arange(Q, 1, 0.0005, dtype=float)
        
        variance=super().variance(weights)*stress_factor
        VaR=variance/np.sqrt(252)*norm(loc =0 , scale = 1).ppf(1-Q)
        CVaR=variance/np.sqrt(252)*norm(loc =0 , scale = 1).ppf(1-intervals).mean()
        
        return VaR,CVaR
        
        
    def multivariate_distribution(self,
                    stress_factor=1.0,
                    iterations=10000):
        
        #Return Multivariate Distribution of a portfolio taking into account potential correlation
        
        num_asset=len(self.returns.columns)
        
        if type(stress_factor)==float:
            
            stress_vec=np.linspace(stress_factor,stress_factor,num_asset)
            
        else:       
            stress_vec=stress_factor
            
        stress_matrix=np.diag(stress_vec)
        stress_matrix=pd.DataFrame(stress_matrix,columns=self.returns.columns,index=self.returns.columns)
        
        stressed_cov=self.returns.cov().dot(stress_matrix)
        mean=self.returns.mean()
        
        multivariate=np.random.multivariate_normal(mean,stressed_cov,iterations)
        
        return multivariate
    
    def gaussian_copula(self,iterations=10000,stress_factor=1.0):
        
       
        randoms=np.random.normal(size=(10000,self.returns.shape[1])).T
        corr_matrix=self.returns.corr()
        
        if type(stress_factor)==float:
            stress_vec=np.linspace(stress_factor,stress_factor,self.returns.shape[1])
            
        else:
            
            stress_vec=stress_factor
        
        if not is_pos_def(corr_matrix):
            corr_matrix=cov_nearest(corr_matrix)
        
        cholesky=np.linalg.cholesky(corr_matrix)
        simulation=np.matmul(cholesky,randoms).T
        simulation=pd.DataFrame(simulation)
        simulation.columns=self.returns.columns

        copula_sample=simulation*self.returns.std()*stress_vec+self.returns.mean()
        
        return copula_sample
    
    def t_copula(self,iterations=10000,stress_factor=1.0):
        

        df=self.returns.shape[1]*self.returns.shape[1]//2+self.returns.shape[1]
        ChiSquared = np.random.chisquare(df=df, size=iterations)

        randoms=np.random.normal(size=(10000,self.returns.shape[1])).T
        corr_matrix=self.returns.corr()
        
        if type(stress_factor)==float:
            stress_vec=np.linspace(stress_factor,stress_factor,self.returns.shape[1])
            
        else:    
            stress_vec=stress_factor
        
        if not is_pos_def(corr_matrix):
            
            corr_matrix=cov_nearest(corr_matrix)
        
        cholesky=np.linalg.cholesky(corr_matrix)
            

        simulation=np.matmul(cholesky,randoms)/np.sqrt(ChiSquared/df)
        simulation=pd.DataFrame(simulation.T)
        simulation.columns=self.returns.columns

        copula_sample=simulation*self.returns.std()*stress_vec+self.returns.mean()
        
        return copula_sample
    
    def gumbel_copula(self,iterations=10000,theta=2):
        
        uniform_sample=np.random.uniform(size=(iterations,self.returns.shape[1]))
        gumbel=np.exp(-(-np.log(uniform_sample))**(theta))
        scaled_gumbel=norm.ppf(gumbel,loc=self.returns.mean(),scale=self.returns.std())

        return scaled_gumbel

    def monte_carlo(self,spot,horizon=20/250,iterations=10000,stress_factor=1.0):
        
        
        num_asset=len(self.returns.columns)
        #haltons=generate_halton(iterations,num_asset,base=2)
        randoms=np.random.normal(size=(10000,num_asset)).T
        
        # Create a stress matrix to stress the covariance matrix
        
        if type(stress_factor)==float:
            
            stress_vec=np.linspace(stress_factor,stress_factor,num_asset)
            
        else: 
            
            stress_vec=stress_factor
        
        
        #Stress the volatilities of the assets
        
        vol=self.returns.std()*np.sqrt(250)*stress_vec
        
        #Create a diagonal matrix of the stress factors
        
        stress_matrix=np.diag(stress_vec)
        stress_matrix=pd.DataFrame(stress_matrix,columns=self.returns.columns,index=self.returns.columns)
        
        #Find nearest PSD matrix and apply cholesky decomposition to create correaltion effect in Monte Carlo
        
        stressed_cov=self.returns.cov().dot(stress_matrix)
        stressed_std=np.sqrt(np.diag(stressed_cov))
        corr_matrix=stressed_cov/np.outer(stressed_std,stressed_std)
        
        if not is_pos_def(corr_matrix):
            corr_matrix=cov_nearest(corr_matrix)
        
        cholesky=np.linalg.cholesky(corr_matrix)
            
            
        drift=np.exp(-0.5*horizon*vol**2)
        factors=spot*drift
        factors_vec=factors.to_numpy().reshape(num_asset,-1)
                
        simulation=np.matmul(cholesky,randoms).T
        simulation=pd.DataFrame(simulation)
        simulation.columns=self.returns.columns
 
        
        monte_carlo=factors_vec.T*np.exp(simulation.dot(np.diag(vol))*np.sqrt(horizon))
        monte_carlo=pd.DataFrame(monte_carlo)
        monte_carlo.columns=self.returns.columns
        perf_monte_carlo=np.log(monte_carlo/spot)
        
        return monte_carlo,perf_monte_carlo
      

    
    def pca(self,num_components=2):
        
        #Returns the eigen vectors of the covariance matrix
        
        cov_matrix=self.returns.cov()
                
        eig_val, eig_vec=np.linalg.eig(cov_matrix)
        sorted_eig_val=eig_val.argsort()[::-1]
        eig_val=eig_val[sorted_eig_val]
        eig_vec=eig_vec[:,sorted_eig_val]
        eig_val=eig_val[:num_components]
        eig_vec=eig_vec[:,0:num_components]
        
        PC={}
        
        for i in range(eig_vec.shape[1]):
            
            PC["PC" +str(i+1)]=eig_vec[:,i]/eig_vec[:,i].sum()
        
        
        portfolio_components=pd.DataFrame(PC.values(),index=PC.keys(),columns=self.returns.columns).T
        
        return eig_val,eig_vec,portfolio_components
    

    def var_contrib(self,weights):
        
        weights_matrix=np.diag(weights)
        variance_contrib=np.dot(weights_matrix,np.dot(self.returns.cov(),weights_matrix.T))
        
        asset_contrib=variance_contrib.sum(axis=0)    
        diag=np.diag(variance_contrib.diagonal())
        variance_decomposition=np.column_stack([asset_contrib,variance_contrib.diagonal(),(variance_contrib-diag).sum(axis=0)])
        contrib=pd.DataFrame(variance_decomposition,index=self.returns.columns,columns=['Variance Contribution','Idiosyncratic Risk','Correlation'])
        
        weighted_covar=pd.DataFrame(variance_contrib,columns=self.returns.columns,index=self.returns.columns)
        
        return contrib,weighted_covar
    
    def var_contrib_pct(self,weights):
        
        var_contrib=self.var_contrib(weights)[0]
        var_contrib=var_contrib/var_contrib['Variance Contribution'].sum()
        var_contrib.columns=['Variance Contribution in %','Idiosyncratic Risk in %','Correlation in %']
        var_contrib=var_contrib.loc[(var_contrib!=0).any(axis=1)]
        var_contrib=var_contrib.sort_values('Variance Contribution in %', ascending=False)
    
        return var_contrib
    
    def perf_contrib(self,weights,amount=100):

        fictive_prices=(1+self.returns).cumprod()
        fictive_prices.iloc[0]=1
        shares=amount*weights/fictive_prices.iloc[0]
        fictive_portfolio=shares*fictive_prices
        book_cost=fictive_portfolio.iloc[0]
        last_value=fictive_portfolio.iloc[-1]
        pnl=last_value-book_cost
        pnl_dataframe=pd.DataFrame(pnl.sort_values(ascending=False),columns=['Performance Contribution'])
        pnl_dataframe=pnl_dataframe.loc[(pnl_dataframe!=0).any(axis=1)]
        return pnl_dataframe

    def perf_contrib_pct(self,weights,amount=100):

        pnl_dataframe=self.perf_contrib(weights,amount)
        pnl_dataframe_pct=pnl_dataframe/pnl_dataframe.sum()
        
        return pnl_dataframe_pct
    
    def summary(self,weights):
        
        inventory=self.inventory(weights)
        perf_report=self.perf_contrib_pct(weights)
        var_contrib_pct=self.var_contrib_pct(weights)

        report=pd.concat([inventory,perf_report,var_contrib_pct],axis=1)
        
        return report.dropna()
    
    def tracking_error(self,ptf,bench):
        
        excess_return=ptf-bench
        tracking_error=excess_return.std()*np.sqrt(252)
        
        return tracking_error


# In[ ]:





# In[ ]:




