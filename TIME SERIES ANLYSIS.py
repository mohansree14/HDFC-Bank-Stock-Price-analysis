# %%
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns



# %%
data = pd.read_csv('HDFCBANK.csv', parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))

# Calculate the difference between the closing price of the first and last trading days for each year
data['Year'] = data['Date'].dt.year
yearly_changes = data.groupby('Year').agg({'Close': lambda x: x.iloc[-1] - x.iloc[0]})

# %%
data.describe()

# %%
plt.figure(figsize=(10, 6))
plt.bar(yearly_changes.index, yearly_changes['Close'], color=['red' if x < 0 else 'green' for x in yearly_changes['Close']])
plt.title('Yearly Losses and Gains of HDFC Bank Stock')
plt.xlabel('Year')
plt.ylabel('Price Difference (INR)')
plt.axhline(y=0, color='black', linestyle='--')  # Adding a horizontal line at y=0
plt.show()

# %%
plt.figure(figsize=(12, 6))
for year in data['Year'].unique():
    plt.bar(data[data['Year'] == year]['Date'], data[data['Year'] == year]['Close'], label=year)

plt.title('HDFC Bank Stock Closing Prices Over Various Years')
plt.xlabel('Date')
plt.ylabel('Closing Price (INR)')
plt.legend(title='Year', loc='upper left')
plt.grid(True)
plt.show()

# %%
dates = pd.to_datetime(data['Date'])
closing_prices = data['Close']

# Plotting the data
plt.figure(figsize=(14, 7))
plt.plot(dates, closing_prices, color='b', label='Closing Prices')
plt.title('HDFC Bank Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# %%
result = seasonal_decompose(data['Close'], model='additive', period=1)
result.plot()
plt.show()

# %%
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'{key}: {value}')

# %%
adf_test(data['Close'])

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Define and fit SARIMA model
model = SARIMAX(train['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fit_model = model.fit()

# Forecast future prices
forecast = fit_model.forecast(steps=len(test))

# Evaluate model performance
mae = mean_absolute_error(test['Close'], forecast)
mse = mean_squared_error(test['Close'], forecast)
rmse = np.sqrt(mse)

print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)


# %%
# Visualize actual vs. predicted prices
plt.figure(figsize=(14, 7))
plt.plot(test['Date'], test['Close'], color='blue', label='Actual Prices')
plt.plot(test['Date'], forecast, color='red', label='Predicted Prices')
plt.title('Actual vs. Predicted Prices of HDFC Bank Stock')
plt.xlabel('Date')
plt.ylabel('Closing Price (INR)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()



# %%
from statsmodels.tsa.arima.model import ARIMA

# Define and fit ARIMA model
arima_model = ARIMA(train['Close'], order=(1, 1, 1))
arima_fit = arima_model.fit()

# Forecast future prices using ARIMA
arima_forecast = arima_fit.forecast(steps=len(test))

# Evaluate model performance
arima_mae = mean_absolute_error(test['Close'], arima_forecast)
arima_mse = mean_squared_error(test['Close'], arima_forecast)
arima_rmse = np.sqrt(arima_mse)

print('ARIMA Mean Absolute Error:', arima_mae)
print('ARIMA Mean Squared Error:', arima_mse)
print('ARIMA Root Mean Squared Error:', arima_rmse)



# %%
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Close'], label='Actual')
plt.plot(test.index, forecast, color='red', label='SARIMA Forecast')
plt.plot(test.index, arima_forecast, color='green', label='ARIMA Forecast')
plt.title('SARIMA vs ARIMA Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()


