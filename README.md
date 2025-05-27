# Exp.no: 10 IMPLEMENTATION OF SARIMA MODEL
Date:
## AIM:
To implement SARIMA model using python.

## ALGORITHM:
Explore the dataset
Check for stationarity of time series
Determine SARIMA models parameters p, q
Fit the SARIMA model
Make time series predictions and Auto-fit the SARIMA model
Evaluate model predictions
## PROGRAM:
```
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# Load the Gold Price data
file_path = 'Gold Price Prediction.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
time_series = data['Price Today']

# Plot the Gold price time series
plt.figure(figsize=(10, 6))
plt.plot(time_series, label="Gold Price Close")
plt.title('Gold Price Close Time Series')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.savefig('gold_close_price_series.png')
plt.show()

# Function to test stationarity
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary.")

# Test stationarity of the original time series
test_stationarity(time_series)

# Differencing if non-stationary
time_series_diff = time_series.diff().dropna()
print("\nAfter Differencing:")
test_stationarity(time_series_diff)

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(time_series_diff, lags=30, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")

plt.subplot(1, 2, 2)
plot_pacf(time_series_diff, lags=30, ax=plt.gca())
plt.title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.savefig('gold_acf_pacf.png')
plt.show()

# SARIMA model parameters for Gold Price
p, d, q = 1, 1, 1
P, D, Q, m = 1, 1, 1, 12  # m=12 assumes monthly seasonality

# Fit the SARIMA model
model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(P, D, Q, m), enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = model.fit(disp=False)
print(sarima_fit.summary())

# Forecasting
forecast_steps = 12  # Forecast 12 future periods
forecast = sarima_fit.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Generate date index for forecast
forecast_index = pd.date_range(start=time_series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

# Plot historical data and forecast, saving the plot as an image
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Historical Gold Price')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast of Gold Price Close')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.savefig('gold_sarima_forecast.png')
plt.show()

# Calculate Mean Absolute Error (MAE) on the forecast (if comparing with existing data)
test_data = time_series[-forecast_steps:]
pred_data = forecast.predicted_mean[:len(test_data)]
mae = mean_absolute_error(test_data, pred_data)
print('Mean Absolute Error:', mae)
```
## OUTPUT:
Screenshot 2024-11-15 000229 image

![image](https://github.com/user-attachments/assets/a94cacce-d5df-415b-a89a-b764da47f674)


![image](https://github.com/user-attachments/assets/3f5b9bff-0892-4e91-ae30-fd41b04b1e48)


![image](https://github.com/user-attachments/assets/5a731665-0a90-40ab-85f8-6af39b50a97b)

![image](https://github.com/user-attachments/assets/1b0033ec-38e9-4ce9-837c-824087933348)

![image](https://github.com/user-attachments/assets/395dbc0d-e387-48c9-a99a-2e09ccc4abdf)

![image](https://github.com/user-attachments/assets/8e8879e7-a82d-47e1-9f55-37e3dafa0bb7)



## RESULT:
Thus the program run successfully based on the SARIMA model.
