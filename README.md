# Time Series Forecasting: A Comparative Study of ARIMA, LSTM, and GRU

This project explores and compares classical and deep learning approaches for time series forecasting. Using both low-frequency and high-frequency datasets, we analyze the effectiveness of ARIMA, LSTM, and GRU models based on statistical and error-based metrics.

---

## 📌 Table of Contents

- [Background and Motivation](#background-and-motivation)
- [Project Objectives](#project-objectives)
- [Datasets Description](#datasets-description)
- [Methodology Overview](#methodology-overview)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Development](#2-model-development)
  - [3. Model Evaluation](#3-model-evaluation)
- [Experimental Results](#experimental-results)
- [Visual Insights](#visual-insights)
- [Discussion and Observations](#discussion-and-observations)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)

---

## 📖 Background and Motivation

Time series forecasting is a critical analytical task with applications across finance, economics, healthcare, and operations. Accurate prediction of future values from historical data can inform strategic decisions. While traditional statistical models like ARIMA have long been used, deep learning models such as LSTM and GRU have emerged as powerful tools for modeling nonlinear temporal dynamics.

This project investigates how these approaches perform under different data characteristics — specifically:
- **Stable, seasonal, low-frequency** data (Johnson & Johnson quarterly sales)
- **Volatile, high-frequency** data (Amazon daily stock prices)

---

## 🎯 Project Objectives

- Implement ARIMA, LSTM, and GRU for time series forecasting
- Preprocess time series data for both statistical and neural models
- Compare model performance across datasets with different properties
- Evaluate models using multiple metrics and identify the most suitable approach for each type of time series

---

## 📊 Datasets Description

### 1. Johnson & Johnson Quarterly Earnings
- Source: Yahoo Finance 
- Period: 1960 to 1980
- Frequency: Quarterly
- Characteristics: Smooth trend, seasonality, stable

### 2. Amazon Daily Stock Prices
- Source: Yahoo Finance
- Period: Recent 5 years
- Frequency: Daily
- Characteristics: Noisy, irregular trends, volatility

---

## ⚙️ Methodology Overview

### 1. Data Preprocessing

For both datasets, the following preprocessing steps were applied:

- **Visualization** to understand underlying patterns (trend/seasonality).
- **Log transformation** to stabilize variance.
- **Differencing** to ensure stationarity.
- **ADF and KPSS tests** to confirm stationarity.
- **Scaling** (MinMax) for neural networks.
- **Sliding window technique** for LSTM/GRU input structuring.

### 2. Model Development

#### ➤ ARIMA:
- Identification of `p`, `d`, and `q` using ACF/PACF plots
- Grid search for optimal parameters
- Residual analysis for validation

#### ➤ LSTM & GRU:
- Defined sequential Keras models
- Used one hidden layer with 64–128 units
- Dropout regularization
- Trained with Adam optimizer and Mean Squared Error loss
- Early stopping to prevent overfitting

### 3. Model Evaluation

We used the following metrics to evaluate predictions:

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Correlation Coefficient**: Strength of predicted vs actual alignment

---

## 📈 Experimental Results

| Dataset               | Model | RMSE   | MAE    | MAPE   | Correlation |
|-----------------------|--------|--------|--------|--------|-------------|
| J&J Quarterly Earnings| ARIMA  | 9.61   | 7.11   | 5.84%  | 0.96        |
|                       | LSTM   | 14.17  | 10.24  | 8.12%  | 0.89        |
|                       | GRU    | 13.85  | 9.87   | 7.96%  | 0.91        |
| Amazon Daily Stock    | ARIMA  | 25.72  | 20.18  | 13.41% | 0.74        |
|                       | LSTM   | 17.54  | 13.22  | 11.35% | 0.81        |
|                       | GRU    | 14.88  | 11.15  | 9.53%  | 0.87        |

---

## 📊 Visual Insights

> *(Add plots generated from the notebook such as below)*
> <img width="915" alt="Screenshot 2025-04-11 at 3 08 10 AM" src="https://github.com/user-attachments/assets/71fb373c-0b9e-41d7-8358-65ba91ea5d9d" />


- Time series before and after differencing
- ACF/PACF plots
- Model loss/accuracy per epoch
- Forecast vs Actual plots for each model
- Residual plots to confirm white noise

---

## 💡 Discussion and Observations

- **ARIMA** outperforms deep learning on stable, seasonal data like J&J due to its strength in linear patterns and regularity.
- **GRU** performs better than LSTM for high-frequency data with noise (Amazon), likely due to fewer parameters and faster convergence.
- Neural networks require significantly more preprocessing and tuning but generalize better in non-linear scenarios.
- LSTM showed overfitting tendencies on the smaller J&J dataset.
- All models improved significantly with log transformation and normalization.

---

## ✅ Conclusion

This study reinforces that model choice in time series forecasting should be driven by:
- Data frequency and volume
- Presence of linearity vs non-linearity
- Seasonality vs volatility

> **Recommendations:**
> - Use ARIMA for stationary, low-frequency data
> - Use GRU or LSTM for volatile, high-frequency datasets
> - Always validate model assumptions using statistical tests and residuals

---

## 🛠️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/timeseries-arima-lstm-gru.git
   cd timeseries-arima-lstm-gru
