import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Download historical stock price data for Apple Inc.
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Extract the closing prices
stock_prices = data['Close']

# Generate a time index
time_index = np.arange(len(stock_prices))

# Trend Stationary Process (TSP)
# Fit a linear trend and subtract it from the original series
trend = np.polyval(np.polyfit(time_index, stock_prices, 1), time_index)
detrended_prices = stock_prices - trend

# Difference Stationary Process (DSP)
# Apply differencing to the stock prices
diff_prices = stock_prices.diff().dropna()

# Generate a time index for differenced data
diff_time_index = np.arange(len(diff_prices))

# Plot original and TSP-detrended stock prices side by side
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(time_index, stock_prices, label='Original Stock Prices')
plt.plot(time_index, detrended_prices, label='TSP-Detrended Stock Prices', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Original and TSP-Detrended Stock Prices')
plt.legend()

# Plot original and DSP-differenced stock prices side by side
plt.subplot(1, 2, 2)
plt.plot(time_index, stock_prices, label='Original Stock Prices')
plt.plot(diff_time_index, diff_prices, label='DSP-Differenced Stock Prices', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Original and DSP-Differenced Stock Prices')
plt.legend()

plt.tight_layout()
plt.show()

# Trend Stationary Process (TSP)
# Fit a linear trend and subtract it from the original series
trend = np.polyval(np.polyfit(time_index, stock_prices, 1), time_index)
detrended_prices = stock_prices - trend

# Difference Stationary Process (DSP)
# Apply differencing to the stock prices
diff_prices = stock_prices.diff().dropna()

# Perform Augmented Dickey-Fuller (ADF) test
adf_result = adfuller(stock_prices)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

# Plot original, TSP-detrended, and DSP-differenced stock prices
plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
plt.plot(time_index, stock_prices, label='Original Stock Prices')
plt.plot(time_index, detrended_prices, label='TSP-Detrended Stock Prices', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Original and TSP-Detrended Stock Prices')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_index[1:], diff_prices, label='DSP-Differenced Stock Prices', linestyle='--')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Differenced Stock Price')
plt.title(f'{ticker} DSP-Differenced Stock Prices')
plt.legend()

plt.tight_layout()
plt.show()

# Apply log transformation
log_transformed_prices = np.log(stock_prices)

# Plot the original and log-transformed stock prices side by side
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(time_index, stock_prices, label='Original Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Original Stock Prices')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time_index, log_transformed_prices, label='Log-Transformed Stock Prices', color='orange')
plt.xlabel('Time')
plt.ylabel('Log Stock Price')
plt.title(f'{ticker} Log-Transformed Stock Prices')
plt.legend()

plt.tight_layout()
plt.show()
