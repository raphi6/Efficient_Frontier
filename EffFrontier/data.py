import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.optimize as sc
import datetime as dt

from pandas_datareader import data as pandasdr
import yfinance as yf
yf.pdr_override()




def getData(stocks, start, end):

    yahoo_data = pandasdr.get_data_yahoo(stocks, end=end, start=start)['Close']

    daily_returns = yahoo_data.pct_change()
    covariance_matrix = daily_returns.cov()
    std_daily_returns = daily_returns.std()

    daily_returns_mean = daily_returns.mean()

    return daily_returns_mean, covariance_matrix


def portfolioPerformance(weights, meanReturns, covMatrix):
    "Calculations for Portfolio"

    " Returns: Summing mean returns with weights, and for yearly weight: 252, simply"
    returns = np.sum(meanReturns*weights)*252


    " std: Standard Deviation, using Portfolio Variance from Modern Portfolio Theory, and sqrt(252) for yearly"

    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)

    return returns, std






""" 
    
    Now for the Optimization Problem from Modern Portfolio Theory 

        We will first maximise the Sharpe ratio for given weights. Lets do this with a minimization function, but just
        compute the negative Sharpe. (we have good minimization function from scipy)

"""

def negativeSharpe(weights, meanReturns, covMatrix, riskFreeRate=0):

    portReturns, portStd = portfolioPerformance(weights, meanReturns, covMatrix)


    return -(portReturns - riskFreeRate)/portStd

""" negative Sharpe done, now for optimising the max Sharpe ratio (by min -Sharpe)  """

def maxSharpe(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """
    Minimize the negative Sharpe Ratio, by altering weights of portfolio.

    :param constraintSet:  Constraint for each stocks final weight in portfolio (Sharpe might want 0 in TSLA. but
                           we want TSLA)
    """

    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # sum of all weights must be 1
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))

    result = sc.minimize(negativeSharpe, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds,
                         constraints=constraints)  # first argument: func we want to minimise

    return result






""" Now to optimize by minimize variance  """

def portfolioVariance(weights, meanReturns, covMatrix):

    print(portfolioPerformance(weights, meanReturns, covMatrix)[1])
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def minVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    """ Minimize portfolio variance by altering weights of portfolio. """

    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # sum of all weights must be 1

    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))

    result = sc.minimize(portfolioVariance, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                         constraints=constraints)  # first argument: func we want to minimise

    return result



""" Printing results from max Sharpe portfolio and min Variance portfolio.  """


def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """ Read in mean, cov, and other financial info

        To Output: Max SR, Min Volatility, Efficient Frontier"""

    # Max Sharpe Ratio Portfolio
    maxSR_portfolio = maxSharpe(meanReturns, covMatrix)

    maxSR_weights = maxSR_portfolio['x']
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_weights, meanReturns, covMatrix)
    maxSR_returns, maxSR_std = round(maxSR_returns*100, 2), round(maxSR_std*100, 2)
    maxSR_allocation = pd.DataFrame(maxSR_weights, index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100, 0 ) for i in maxSR_allocation.allocation]

    # Min Volatility Portfolio
    minVol_portfolio = minVariance(meanReturns, covMatrix)

    minVol_weights = minVol_portfolio['x']
    minVol_returns, minVol_std = portfolioPerformance(minVol_weights, meanReturns, covMatrix)
    minVol_returns, minVol_std = round(minVol_returns*100, 2), round(minVol_std*100, 2)
    minVol_allocation = pd.DataFrame(minVol_weights, index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i * 100, 0) for i in minVol_allocation.allocation]

    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation


"""
    
    Now for the Final optimization from MPT;
    
                - min Portfolio Variance
                
                s. t.    Portfolio Return > mu_b   
                         sum(weights) = 1
"""


def portfolioReturn(weights, meanReturns, covMatrix):

    return portfolioPerformance(weights, meanReturns, covMatrix)[0]

def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0., 1)):

    """ For each return target (mu_b), we want to optimize the portfolio for min variance. """


    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = [{'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    #constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    #constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget})
    print(covMatrix)
    print(meanReturns)
    print(returnTarget)

    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = sc.minimize(portfolioVariance, numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds,
                         constraints=constraints)
    return effOpt




end_date = dt.datetime.now()  # - datet.timedelta(days=1600)
start_date = end_date - dt.timedelta(days=30)
stock_list = ['GOOGL', 'TSLA', 'ADBE']
#stock_list = ['GOOGL', 'TSLA', 'MSFT']
#weights = np.array([0.3, 0.3, 0.4])


#meanReturns, covMatrix = getData(stocks=stock_list, start=start_date, end=end_date)


#print(calculatedResults(meanReturns, covMatrix))


yahoo_data = pandasdr.get_data_yahoo(stock_list, end=end_date, start=start_date)['Close']

dr = yahoo_data.pct_change()
covariance_matrix = dr.cov()
std_dr = dr.std()

dr_mean = dr.mean()



print(efficientOpt(dr_mean, covariance_matrix, 0.05))



