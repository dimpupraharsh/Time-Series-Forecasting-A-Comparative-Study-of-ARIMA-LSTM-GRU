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

> <img width="915" alt="Screenshot 2025-04-11 at 3 08 10 AM" src="https://github.com/user-attachments/assets/71fb373c-0b9e-41d7-8358-65ba91ea5d9d" />
 <img width="992" alt="Screenshot 2025-04-11 at 3 13 58 AM" src="https://github.com/user-attachments/assets/795d00ed-7461-4046-aec1-3390bdc7199c" />
 <img width="1003" alt="Screenshot 2025-04-11 at 3 31 27 AM" src="https://github.com/user-attachments/assets/840e8f10-ed1a-497c-ab2b-942aefe39aca" />
 <img width="996" alt="Screenshot 2025-04-11 at 3 31 53 AM" src="https://github.com/user-attachments/assets/9ae50595-7b87-4b4d-a4c1-a367865e2907" />
 <img width="1014" alt="Screenshot 2025-04-11 at 3 39 27 AM" src="https://github.com/user-attachments/assets/65cf0173-f7d8-410b-9936-c071163cf686" />
 <img width="1010" alt="Screenshot 2025-04-11 at 4 30 51 AM" src="https://github.com/user-attachments/assets/9f044548-a24a-437f-9de7-64db659f63c9" />
 <img width="1004" alt="Screenshot 2025-04-11 at 4 31 14 AM" src="https://github.com/user-attachments/assets/ce729ebd-0869-41e1-bb2d-8f95cccaa695" />
 <img width="1009" alt="Screenshot 2025-04-11 at 4 49 22 AM" src="https://github.com/user-attachments/assets/b8d4c65b-ce6c-487d-be7d-1e5cd7eac23d" />
 <img width="1018" alt="Screenshot 2025-04-11 at 4 54 08 AM" src="https://github.com/user-attachments/assets/3eaf5699-736d-4bf5-a7c3-ddfed09d5a87" />
 <img width="998" alt="Screenshot 2025-04-11 at 4 58 56 AM" src="https://github.com/user-attachments/assets/e1f2f5d1-32b1-4d3e-80f1-1a041b4c7620" />
 <img width="1010" alt="Screenshot 2025-04-11 at 4 59 21 AM" src="https://github.com/user-attachments/assets/d6e15dd9-8f63-401a-b686-0666389e50dd" />
 <img width="922" alt="Screenshot 2025-04-11 at 5 04 05 AM" src="https://github.com/user-attachments/assets/fba50417-7788-4065-8277-3fda84c7d146" />
 <img width="1015" alt="Screenshot 2025-04-11 at 5 04 15 AM" src="https://github.com/user-attachments/assets/995ef3a7-c3e7-456c-b6e8-c9af6bb553cf" />
 <img width="1018" alt="Screenshot 2025-04-11 at 4 54 08 AM" src="https://github.com/user-attachments/assets/1cf69438-9548-4080-bacd-3a02825ced44" />








 




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
