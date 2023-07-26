import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()

# Define the tickers for the stocks in your portfolio
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Define the time period for which you want to fetch the stock data
start_date = '2018-01-01'
end_date = '2021-09-30'

# Fetch the stock data using pandas-datareader
stock_data = web.get_data_yahoo(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns of the stocks
returns = stock_data.pct_change().dropna()

# Calculate the mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Number of portfolios to simulate
num_portfolios = 10000

# Initialize arrays to store portfolio weights, returns, and volatility
weights_array = np.zeros((num_portfolios, len(tickers)))
returns_array = np.zeros(num_portfolios)
volatility_array = np.zeros(num_portfolios)

# Simulate random portfolios
for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    weights_array[i, :] = weights

    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    returns_array[i] = portfolio_return
    volatility_array[i] = portfolio_std_dev

# Create a DataFrame to store the simulated data
portfolios_df = pd.DataFrame({'Return': returns_array, 'Volatility': volatility_array})

# Plot the efficient frontier
plt.figure(figsize=(12, 8))
plt.scatter(portfolios_df['Volatility'], portfolios_df['Return'], marker='o', s=10, alpha=0.3)
plt.xlabel('Volatility (Annualized)')
plt.ylabel('Expected Return (Annualized)')
plt.title('Efficient Frontier')
plt.grid(True)

# Find the portfolio with the highest Sharpe ratio (risk-adjusted return)
risk_free_rate = 0.03  # You can update this value with the risk-free rate in your region
portfolios_df['Sharpe Ratio'] = (portfolios_df['Return'] - risk_free_rate) / portfolios_df['Volatility']
max_sharpe_idx = portfolios_df['Sharpe Ratio'].idxmax()
max_sharpe_return = portfolios_df.loc[max_sharpe_idx, 'Return']
max_sharpe_volatility = portfolios_df.loc[max_sharpe_idx, 'Volatility']

# Mark the portfolio with the highest Sharpe ratio
plt.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='r', s=100, label='Max Sharpe Ratio')

# Find the portfolio with the minimum volatility
min_volatility_idx = portfolios_df['Volatility'].idxmin()
min_volatility_return = portfolios_df.loc[min_volatility_idx, 'Return']
min_volatility_volatility = portfolios_df.loc[min_volatility_idx, 'Volatility']

# Mark the portfolio with the minimum volatility
plt.scatter(min_volatility_volatility, min_volatility_return, marker='*', color='g', s=100, label='Min Volatility')

# Add legend
plt.legend()

# Show the plot
plt.show()