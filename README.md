
# Inventory Management and Demand Forecasting Project

## Overview

This project focuses on optimizing inventory management using various forecasting techniques to predict demand accurately. The aim is to minimize costs, avoid stockouts, and enhance decision-making through data-driven insights.

## Objectives

1. **Minimize Cost**: Reduce the overall inventory holding and ordering costs.
2. **Avoid Stockouts**: Ensure that products are available when needed by maintaining optimal stock levels.
3. **Reduce Lead Time**: Streamline processes to ensure timely fulfillment of orders.
4. **Demand Forecasting**: Predict future sales based on historical data to make informed inventory decisions.

## Algorithm Implemented


### 3. ARIMA (AutoRegressive Integrated Moving Average)

**Objective**: Combines autoregressive and moving average models to capture different aspects of a time series. It is suitable for datasets that exhibit trends and seasonality.

**Why ARIMA?**: ARIMA is beneficial for univariate time series forecasting where:
- The data shows evidence of non-stationarity (the mean and variance change over time).
- There are underlying patterns that can be captured through past values and errors.
- It can handle seasonality with an extension called Seasonal ARIMA (SARIMA).

**Implementation**:
```python
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(sales_data, order=(1, 1, 1)):
    model = ARIMA(sales_data['QuantitySold'], order=order)
    model_fit = model.fit()
    return model_fit.forecast(steps=12)
```


## Data Preparation

Data was prepared by loading historical sales data, transforming it into a time series format, and resampling as needed for analysis.

```python
import pandas as pd

# Load and preprocess sales data
sales_data = pd.read_csv('sales_data.csv')
sales_data['Date'] = pd.to_datetime(sales_data['Date'])
sales_data.set_index('Date', inplace=True)
sales_data = sales_data.resample('M').sum()
```

## Visualization

The results of each forecasting method were visualized to compare their effectiveness in predicting demand. 

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(sales_data.index, sales_data['QuantitySold'], label='Actual Sales', color='blue')
# Add other forecasts as needed
plt.title('Sales Forecasting Comparison')
plt.xlabel('Date')
plt.ylabel('Sales Quantity')
plt.legend()
plt.show()
```

## Conclusion

This project demonstrates the effectiveness of various demand forecasting techniques in optimizing inventory management. By selecting the appropriate forecasting method, businesses can significantly improve their inventory strategies, reduce costs, and enhance customer satisfaction.

## Requirements

To run this project, ensure you have the following Python packages installed:

- pandas
- statsmodels
- prophet
- matplotlib
- keras

You can install the required packages using:

```bash
pip install pandas statsmodels prophet matplotlib keras
```

## Future Work

1. **Integration with a Frontend**: Develop a user interface to allow users to visualize forecasts and interact with the data more effectively.
2. **Real-time Data Processing**: Implement mechanisms to update forecasts based on real-time sales data.
3. **Advanced Machine Learning Techniques**: Explore other advanced algorithms such as XGBoost or other neural network architectures for improved forecasting accuracy.
