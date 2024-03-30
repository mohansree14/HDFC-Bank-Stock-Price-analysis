# HDFC-Bank-Stock-Price-analysis
 This project utilizes SARIMA and ARIMA models to analyze HDFC Bank stock prices, forecasting future movements. It identifies trends, detects seasonal patterns, and assesses volatility. Performance is evaluated using MAE, MSE, and RMSE. Insights from models and external factors offer a holistic view of potential price trends.
# HDFC Bank Stock Price Analysis and Forecasting

## Overview
This project aims to analyze and forecast the stock prices of HDFC Bank using time series analysis techniques, specifically SARIMA (Seasonal Autoregressive Integrated Moving Average) and ARIMA (Autoregressive Integrated Moving Average) models. The analysis includes data visualization, trend identification, seasonal pattern detection, and model evaluation to provide valuable insights into potential future price movements of HDFC Bank stock.

## Data Description
The dataset used in this analysis contains historical daily closing prices of HDFC Bank stock. The data is loaded from a CSV file and includes information such as date, closing price, and yearly changes in stock prices.

## Exploratory Data Analysis (EDA)
### Yearly Changes
- Calculated the difference between the closing price of the first and last trading days for each year.
- Visualized yearly losses and gains using a bar chart.
- ![image](https://github.com/mohansree14/HDFC-Bank-Stock-Price-analysis/assets/113782905/695b2e84-4021-4582-ae0f-040a1bf543e4)
- ![image](https://github.com/mohansree14/HDFC-Bank-Stock-Price-analysis/assets/113782905/1bbd3d1d-1bcb-4c82-aa65-a23e1021d87d)



### Stock Prices Over Time
- Plotted the closing prices of HDFC Bank stock over various years to visualize the trends and fluctuations.

## Time Series Decomposition
- Conducted seasonal decomposition to identify underlying seasonal patterns or trends in the stock prices.

## Stationarity Testing
- Applied the Augmented Dickey-Fuller (ADF) test to check for the stationarity of the stock prices.

## Modeling and Forecasting
### SARIMA Model
- Split the data into training and testing sets.
- Defined and fitted the SARIMA model to the training data.
- Forecasted future prices and evaluated model performance using MAE, MSE, and RMSE.
- ![image](https://github.com/mohansree14/HDFC-Bank-Stock-Price-analysis/assets/113782905/fd8a34cf-e0c4-4be4-9b30-c84ae93959c4)


### ARIMA Model
- Defined and fitted the ARIMA model to the training data.
- Forecasted future prices using ARIMA and evaluated its performance.
- 

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Results and Insights
- The SARIMA and ARIMA models were able to forecast future prices with reasonable accuracy.
- The models provided insights into potential future price trends, but it is essential to consider external factors that may influence stock prices.

- ![image](https://github.com/mohansree14/HDFC-Bank-Stock-Price-analysis/assets/113782905/e903ef0f-bfa4-40d3-b9c2-c091e0e9ec67)


## Conclusion
While the analysis and models provide valuable insights into HDFC Bank stock prices, it is crucial to exercise caution and maintain a prudent approach to decision-making. The forecasts should be viewed as guidance rather than absolute predictions of future prices. Continuous monitoring and refinement of the models are necessary to adapt to changing market conditions and dynamics.

## Future Work
- Explore additional factors and external variables that may influence HDFC Bank stock prices.
- Implement more advanced modeling techniques and strategies to improve forecasting accuracy.
- Conduct a comprehensive risk assessment and develop robust risk management strategies.

## Acknowledgments
- The dataset used in this analysis was sourced from HDFCBANK.csv.[https://drive.google.com/file/d/1I80KP_PM5ODOBZl1awCM7knZF2prkhYm/view?usp=drive_link]

