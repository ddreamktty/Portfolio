# Machine Learning - Avocado Price Prediction Analysis

**Date**: March 2025

## Overview
This project involves building and evaluating a machine learning model to predict avocado prices using historical data. The aim is to forecast future prices and improve recommendations based on data analysis, supporting better business decisions and user insights.

## Dataset
- Source: [Kaggle - Avocado Prices](https://www.kaggle.com/datasets/neuromusic/avocado-prices)

## Objective
- Develop a regression model to predict avocado prices.
- Apply data analysis to identify pricing trends and improve forecasts.
- Forecast prices for the next 2 months.

## ML Pipeline
- Data preprocessing
- Feature selection
- Model training and testing
- Evaluation
- Forecasting

## Model Details
- **Model Used**: Decision Tree Regressor
- **Target Variable**: `AveragePrice`
- **Evaluation Metrics**: MAE, MSE, R² Score

## Results

| Metric    | Training Set | Testing Set |
|-----------|--------------|-------------|
| MSE       | 0.01772      | 0.01665     |
| MAE       | 0.09098      | 0.08972     |
| R² Score  | 0.89153      | 0.89276     |

## Tools & Libraries
- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Conclusion
This project successfully demonstrates the use of machine learning for price forecasting in the agricultural sector. The Decision Tree model provided high accuracy in both training and testing sets, showing strong potential for real-world applications in market trend analysis and decision support.
