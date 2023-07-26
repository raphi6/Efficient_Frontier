import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.optimize as sc
import datetime as dt

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(
            np.dot(weights.T,np.dot(covMatrix, weights))
           )*np.sqrt(252)
    return returns, std

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return - (pReturns - riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


end_date = dt.datetime.now()  # - datet.timedelta(days=1600)
start_date = end_date - dt.timedelta(days=365)
stock_list = ['ALK', 'ALLE', 'ADBE']
stock_list = ['GOOGL', 'TSLA', 'MSFT']
weights = np.array([0.3, 0.3, 0.4])

meanReturns, covMatrix = getData(stocks=stock_list, start=start_date, end=end_date)

result = maxSR(meanReturns, covMatrix)
print(result)