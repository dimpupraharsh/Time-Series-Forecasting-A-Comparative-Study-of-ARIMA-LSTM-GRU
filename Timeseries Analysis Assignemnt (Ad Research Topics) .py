#!/usr/bin/env python
# coding: utf-8

# ## Johnson & Johnson Dataset ##

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# Deep Learning libraries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ======================== ARIMA FUNCTIONS ============================

def load_preprocess_data(file_path):
    """
    Load CSV data, convert date column, set index and add transformed columns.
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['month_index'] = range(len(df))
    df['log_data'] = np.log(df['data'])
    return df

def plot_series(df):
    """
    Plot the original and log-transformed time series.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['month_index'], df['data'], label='Sales for Johnson & Johnson', color='tab:blue')
    plt.title('Time Series Plot for Johnson & Johnson')
    plt.xlabel('Month Number')
    plt.ylabel('Value in $$$')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['month_index'], df['log_data'], label='Log Transformed Time Series', color='orange')
    plt.title('Log Transformed Time Series for Johnson & Johnson')
    plt.xlabel('Month Number')
    plt.ylabel('Log(Value)')
    plt.grid(True)
    plt.legend()
    plt.show()

def run_stationarity_tests(series, series_name="Series"):
    """
    Run ADF and KPSS tests on a given series and print the results.
    """
    print(f"\n=== ADF Test on {series_name} ===")
    adf_result = adfuller(series.dropna())
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value}")
    if adf_result[1] <= 0.05:
        print("Stationary by ADF test.")
    else:
        print(" Non-stationary by ADF test.")
    
    print(f"\n=== KPSS Test on {series_name} ===")
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    print(f"KPSS Statistic: {kpss_result[0]:.4f}")
    print(f"p-value: {kpss_result[1]:.4f}")
    for key, value in kpss_result[3].items():
        print(f"   {key}: {value}")
    if kpss_result[1] <= 0.05:
        print(" Non-stationary by KPSS test.")
    else:
        print(" Stationary by KPSS test.")

def difference_series(df, col='log_data'):
    """
    Calculate first difference of a given column.
    """
    df['log_diff'] = df[col].diff()
    return df

def plot_acf_pacf_side_by_side(series, title_suffix=''):
    """
    Plot ACF and PACF side by side for a time series.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), ax=axes[0])
    axes[0].set_title(f'ACF {title_suffix}')
    plot_pacf(series.dropna(), ax=axes[1], method='ywm')
    axes[1].set_title(f'PACF {title_suffix}')
    plt.tight_layout()
    plt.show()

def grid_search_arima(series, p_range=range(0, 9), d=1, q_range=range(0, 9)):
    """
    Grid search for the best ARIMA(p, d, q) model using AIC.
    Returns a DataFrame with sorted models.
    """
    pdq = list(itertools.product(p_range, [d], q_range))
    results = []
    for param in pdq:
        try:
            model = ARIMA(series, order=param).fit()
            results.append((param[0], param[1], param[2], model.aic, model.bic))
        except Exception as e:
            continue
    result_df = pd.DataFrame(results, columns=['p', 'd', 'q', 'AIC', 'BIC'])
    result_df.sort_values(by='AIC', inplace=True)
    print("\nTop 5 ARIMA Models Based on AIC:")
    print(result_df.head())
    return result_df

def fit_arima(series, order):
    """
    Fit the ARIMA model on the provided series with specified order.
    """
    model = ARIMA(series, order=order).fit()
    return model

def plot_arima_forecasts(df, predicted_log, predicted_original):
    """
    Plot actual vs predicted values for both log and original scales.
    """
    # Plot log scale
    plt.figure(figsize=(12, 6))
    plt.plot(df['log_data'], label='Actual (Log Data)', color='tab:blue')
    plt.plot(predicted_log, label='Predicted (Log Data)', color='tab:green', linestyle='--')
    plt.title('Actual vs Predicted (Log Scale)')
    plt.xlabel('Time')
    plt.ylabel('Log(Value)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot original scale
    month_index = range(len(df))
    plt.figure(figsize=(12, 6))
    plt.plot(month_index, df['data'], label='Actual Data (Original)', color='tab:blue')
    plt.plot(month_index, predicted_original, label='Predicted (Original)', color='green', linestyle='--')
    plt.title('Actual vs Predicted (Original Scale)')
    plt.xlabel('Month Number')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_accuracy_metrics(actual, predicted):
    """
    Compute and print accuracy metrics.
    """
    residuals = actual - predicted
    me = np.mean(residuals)
    mae = mean_absolute_error(actual, predicted)
    mpe = np.mean((residuals/actual) * 100)
    mape = np.mean(np.abs(residuals/actual) * 100)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    acf1 = pd.Series(residuals).autocorr(lag=1)
    corr, _ = pearsonr(actual, predicted)
    minmax = 1 - np.mean(np.minimum(actual, predicted) / np.maximum(actual, predicted))
    
    metrics = pd.DataFrame({
        'Metric': ['ME', 'MAE', 'MPE', 'MAPE', 'RMSE', 'ACF1', 'Correlation', 'Min-Max Error'],
        'Value': [me, mae, mpe, mape, rmse, acf1, corr, minmax]
    })
    print("\nAccuracy Metrics:")
    print(metrics)
    return metrics

def forecast_arima(model, steps, df):
    """
    Forecast future values using the fitted ARIMA model.
    Return forecasted mean and confidence intervals in original scale.
    """
    forecast_result = model.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)
    
    # Back-transform using exponential
    forecast_mean_original = np.exp(forecast_mean)
    forecast_ci_original = np.exp(forecast_ci)
    forecast_ci_original.columns = ['lower', 'upper']
    
    # Build forecast index (using month numbers for simplicity)
    forecast_index = range(len(df), len(df) + steps)
    
    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(df)), df['data'], label='Observed', color='tab:blue')
    plt.plot(forecast_index, forecast_mean_original.values, label='Forecast', color='green')
    plt.fill_between(forecast_index,
                     forecast_ci_original['lower'].values,
                     forecast_ci_original['upper'].values,
                     color='lightgreen', alpha=0.5, label='95% Confidence Interval')
    plt.title('ARIMA Forecast for Next {} Months'.format(steps))
    plt.xlabel('Month Number')
    plt.ylabel('Value in $$$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =================== Deep Learning Functions ======================

def prepare_dl_data(df, features=['data', 'quarter_sin', 'quarter_cos']):
    """
    Add quarter-based seasonality features and return the feature values.
    """
    df['quarter'] = df.index.quarter
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    data = df[features].values
    return data

def scale_data(data):
    """
    Scale all features and separately fit a scaler for the primary series.
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    data_scaler = MinMaxScaler()  # For inverse transforming the primary data
    data_scaler.fit(data[:, [0]])
    
    return data_scaled, scaler, data_scaler

def create_sequences(data, time_steps=24, forecast_horizon=24):
    """
    Create sequential samples for training/testing.
    """
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_horizon + 1):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps:i+time_steps+forecast_horizon, 0])
    return np.array(X), np.array(y)

def build_rnn_model(model_type, input_shape, forecast_horizon, units=128, dropout=0.2):
    """
    Build and compile either an LSTM or GRU based model.
    """
    model = Sequential()
    if model_type.lower() == 'lstm':
        model.add(LSTM(units, activation='tanh', return_sequences=False, input_shape=input_shape))
    elif model_type.lower() == 'gru':
        model.add(GRU(units, activation='tanh', return_sequences=False, input_shape=input_shape))
    else:
        raise ValueError("model_type must be either 'lstm' or 'gru'")
    
    model.add(Dropout(dropout))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_training_history(history, title="Training vs Validation Loss"):
    """
    Plot training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='tab:blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def forecast_future(model, last_window, data_scaler, forecast_horizon):
    """
    Forecast future values using a deep learning model and inverse transform the predictions.
    """
    forecast_scaled = model.predict(last_window, verbose=0)
    forecast_values = data_scaler.inverse_transform(forecast_scaled)[0]
    return forecast_values

def evaluate_deep_model(model, X_val, y_val, data_scaler):
    """
    Use the model to predict on validation data, inverse transform, plot predictions and compute metrics.
    """
    val_pred_scaled = model.predict(X_val, verbose=0)
    val_pred = data_scaler.inverse_transform(val_pred_scaled)
    val_actual = data_scaler.inverse_transform(y_val)
    
    # Flatten predictions and actual values for metric calculations
    y_pred_all = val_pred.flatten()
    y_actual_all = val_actual.flatten()
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_actual_all, label='Actual (Test Set)', color='tab:blue')
    plt.plot(y_pred_all, label='Predicted (Test Set)', linestyle='--', color='green')
    plt.title("Predicted vs Actual (Test Set)")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Compute metrics
    mae = mean_absolute_error(y_actual_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_actual_all, y_pred_all))
    mape = np.mean(np.abs((y_actual_all - y_pred_all) / y_actual_all)) * 100
    corr = np.corrcoef(y_actual_all, y_pred_all)[0, 1]
    acf1 = pd.Series(y_actual_all - y_pred_all).autocorr(lag=1)
    me = np.mean(y_actual_all - y_pred_all)
    mpe = np.mean((y_actual_all - y_pred_all) / y_actual_all * 100)
    minmax = 1 - np.mean(np.minimum(y_actual_all, y_pred_all) / np.maximum(y_actual_all, y_pred_all))
    
    metrics = pd.DataFrame({
        'Metric': ['ME', 'MAE', 'MPE', 'MAPE', 'RMSE', 'ACF1', 'Correlation', 'Min-Max Error'],
        'Value': [me, mae, mpe, mape, rmse, acf1, corr, minmax]
    })
    print("\nEvaluation Metrics on Test Set:")
    print(metrics)
    return metrics

# ========================= MAIN EXECUTION ============================

if __name__ == "__main__":
    
    # ---------- ARIMA Forecasting ----------
    file_path = '/Users/dimpu/Downloads/Downloads/jj.csv'
    df_arima = load_preprocess_data(file_path)
    plot_series(df_arima)
    
    print("### Stationarity Tests on Log Transformed Data ###")
    run_stationarity_tests(df_arima['log_data'], series_name="Log Transformed Data")
    
    # Differencing and retest
    df_arima = difference_series(df_arima, col='log_data')
    print("\n### Stationarity Tests on Log-Differenced Data ###")
    run_stationarity_tests(df_arima['log_diff'], series_name="Log-Differenced Data")
    
    # Plot ACF and PACF for original and differenced series
    plot_acf_pacf_side_by_side(df_arima['log_data'], title_suffix='(Log Data)')
    plot_acf_pacf_side_by_side(df_arima['log_diff'], title_suffix='(Log-Differenced Data)')
    
    # Grid search for best ARIMA model
    result_df = grid_search_arima(df_arima['log_data'], p_range=range(0,9), d=1, q_range=range(0,9))
    best_row = result_df.iloc[0]
    best_order = (int(best_row['p']), int(best_row['d']), int(best_row['q']))
    print(f"\nSelected Best ARIMA Model: {best_order}")
    
    # Fit best ARIMA model and predict
    best_model_arima = fit_arima(df_arima['log_data'], order=best_order)
    predicted_log = best_model_arima.predict(start=0, end=len(df_arima['log_data']) - 1, typ='levels')
    predicted_log.index = df_arima.index
    predicted_original = np.exp(predicted_log)  # convert back from log space
    
    plot_arima_forecasts(df_arima, predicted_log, predicted_original)
    compute_accuracy_metrics(df_arima['data'], predicted_original)
    forecast_arima(best_model_arima, steps=24, df=df_arima)
    
    # Optionally, view ARIMA diagnostics
    best_model_arima.plot_diagnostics(figsize=(12, 8))
    plt.suptitle("ARIMA Model Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # ---------- Deep Learning Forecasting (LSTM & GRU) ----------
    # Load data separately for DL models
    df_dl = pd.read_csv(file_path)
    df_dl['date'] = pd.to_datetime(df_dl['date'])
    df_dl.set_index('date', inplace=True)
    
    # Prepare data with seasonality features
    data = prepare_dl_data(df_dl, features=['data', 'quarter_sin', 'quarter_cos'])
    data_scaled, scaler, data_scaler = scale_data(data)
    
    # Create sequences for DL models
    time_steps = 24
    forecast_horizon = 24  # adjust if needed
    X, y = create_sequences(data_scaled, time_steps, forecast_horizon)
    
    # Train-test split
    split_idx = int(len(X) * 0.9)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    # ------- LSTM Model -------
    model_lstm = build_rnn_model(model_type='lstm', input_shape=(X_train.shape[1], X_train.shape[2]), 
                                 forecast_horizon=forecast_horizon, units=128, dropout=0.2)
    history_lstm = model_lstm.fit(X_train, y_train, 
                                  epochs=200, batch_size=1, 
                                  validation_data=(X_val, y_val), 
                                  callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
                                  verbose=1)
    plot_training_history(history_lstm, title="LSTM Training vs Validation Loss")
    evaluate_deep_model(model_lstm, X_val, y_val, data_scaler)
    
    # Forecast future using LSTM
    last_window = data_scaled[-time_steps:].reshape(1, time_steps, data_scaled.shape[1])
    forecast_lstm_vals = forecast_future(model_lstm, last_window, data_scaler, forecast_horizon)
    forecast_dates = pd.date_range(df_dl.index[-1] + pd.offsets.QuarterBegin(), periods=forecast_horizon, freq='Q')
    df_forecast_lstm = pd.DataFrame({'forecast': forecast_lstm_vals}, index=forecast_dates)
    
    # Plot LSTM forecast along with a confidence interval based on training residuals
    train_pred = model_lstm.predict(X_train, verbose=0)
    train_pred_inv = data_scaler.inverse_transform(train_pred)
    train_actual_inv = data_scaler.inverse_transform(y_train)
    residual_std = np.std(train_actual_inv - train_pred_inv)
    ci_upper = df_forecast_lstm['forecast'] + 1.96 * residual_std
    ci_lower = df_forecast_lstm['forecast'] - 1.96 * residual_std
    
    plt.figure(figsize=(14, 6))
    plt.plot(df_dl.index, df_dl['data'], label='Historical Data', color='tab:blue')
    plt.plot(df_forecast_lstm.index, df_forecast_lstm['forecast'], 'o-', label='LSTM Forecast', color='darkorange')
    plt.fill_between(df_forecast_lstm.index, ci_lower, ci_upper, alpha=0.3, color='skyblue', label='95% CI')
    plt.title("LSTM Forecast for Next 24 Months")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # ------- GRU Model -------
    # Build and train GRU model (using the same prepared sequences)
    model_gru = build_rnn_model(model_type='gru', input_shape=(X_train.shape[1], X_train.shape[2]), 
                                forecast_horizon=forecast_horizon, units=128, dropout=0.2)
    history_gru = model_gru.fit(X_train, y_train, 
                                epochs=200, batch_size=1, 
                                validation_data=(X_val, y_val), 
                                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
                                verbose=1)
    plot_training_history(history_gru, title="GRU Training vs Validation Loss")
    evaluate_deep_model(model_gru, X_val, y_val, data_scaler)
    
    # Forecast future using GRU
    last_window = data_scaled[-time_steps:].reshape(1, time_steps, data_scaled.shape[1])
    forecast_gru_vals = forecast_future(model_gru, last_window, data_scaler, forecast_horizon)
    forecast_dates = pd.date_range(df_dl.index[-1] + pd.offsets.QuarterBegin(), periods=forecast_horizon, freq='Q')
    df_forecast_gru = pd.DataFrame({'forecast': forecast_gru_vals}, index=forecast_dates)
    
    # Plot GRU forecast
    train_pred = model_gru.predict(X_train, verbose=0)
    train_pred_inv = data_scaler.inverse_transform(train_pred)
    train_actual_inv = data_scaler.inverse_transform(y_train)
    residual_std = np.std(train_actual_inv - train_pred_inv)
    ci_upper = df_forecast_gru['forecast'] + 1.96 * residual_std
    ci_lower = df_forecast_gru['forecast'] - 1.96 * residual_std
    
    plt.figure(figsize=(14, 6))
    plt.plot(df_dl.index, df_dl['data'], label='Historical Data', color='tab:blue')
    plt.plot(df_forecast_gru.index, df_forecast_gru['forecast'], 'o-', label='GRU Forecast', color='darkorange')
    plt.fill_between(df_forecast_gru.index, ci_lower, ci_upper, alpha=0.3, color='skyblue', label='95% CI')
    plt.title("GRU Forecast for Next 24 Months")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Amazon Dataset ##

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# 1. Data Loading and Resampling
# --------------------------
def load_and_resample_data(csv_file):
    # Load data and set Date as index
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Daily data for plotting
    daily_data = df.copy()
    
    # Monthly resampling (using mean for ARIMA modeling)
    monthly_data = df['Adj Close'].resample('M').mean()
    return daily_data, monthly_data

def plot_daily_data(daily_data):
    plt.figure(figsize=(12, 5))
    plt.plot(daily_data['Adj Close'], label='Daily Adj Close', color='slateblue')
    plt.title('Amazon Daily Adjusted Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Adj Close in $$$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------
# 2. Stationarity Testing and ACF/PACF
# --------------------------
def run_stationarity_tests(series, series_name="Series"):
    print(f"=== ADF Test on {series_name} ===")
    adf_result = adfuller(series)
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value}")
    print(" Stationary." if adf_result[1] <= 0.05 else " Non-stationary.")
    
    print(f"\n=== KPSS Test on {series_name} ===")
    kpss_result = kpss(series, regression='c', nlags="auto")
    print(f"KPSS Statistic: {kpss_result[0]:.4f}")
    print(f"p-value: {kpss_result[1]:.4f}")
    for key, value in kpss_result[3].items():
        print(f"   {key}: {value}")
    print(" Non-stationary." if kpss_result[1] <= 0.05 else " Stationary.")

def plot_diff_acf_pacf(series, lags=20, title="Differenced Series"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('ACF')
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title('PACF')
    fig.suptitle(f"ACF and PACF of {title}", fontsize=14)
    plt.tight_layout()
    plt.show()

# --------------------------
# 3. Fourier Features
# --------------------------
def get_fourier_terms(index, order=3, constant=False):
    """
    Returns a DeterministicProcess instance and its in-sample Fourier features.
    """
    fourier = CalendarFourier(freq='M', order=order)
    dp = DeterministicProcess(index=index,
                              constant=constant,
                              order=0,
                              seasonal=False,
                              additional_terms=[fourier],
                              drop=True)
    fourier_terms = dp.in_sample()
    return dp, fourier_terms

# --------------------------
# 4. Grid Search for ARIMA(p,1,q) on Log Data
# --------------------------
def grid_search_arima(log_data, exog, p_range=range(6), q_range=range(6)):
    best_aic = np.inf
    best_order = None
    for p, q in itertools.product(p_range, q_range):
        try:
            model = ARIMA(log_data, order=(p, 1, q), exog=exog)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = (p, 1, q)
        except Exception as e:
            continue
    print(f"\nBest ARIMA order based on AIC: {best_order} with AIC = {best_aic:.2f}")
    return best_order, best_aic

# --------------------------
# 5. ARIMA Model Fitting and Diagnostics
# --------------------------
def fit_arima_model(log_data, order, exog):
    model = ARIMA(log_data, order=order, exog=exog)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def plot_arima_diagnostics(model_fit):
    model_fit.plot_diagnostics(figsize=(12, 8))
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(monthly_data, model_fit):
    # Fitted values from an ARIMA(.,1,.) model start at the second observation.
    in_sample_log_pred = np.exp(model_fit.fittedvalues)
    monthly_data_trimmed = monthly_data.iloc[1:]
    in_sample_log_trimmed = in_sample_log_pred.iloc[1:]
    
    plt.figure(figsize=(12, 5))
    plt.plot(monthly_data_trimmed.index, monthly_data_trimmed.values, label='Actual', color='blue')
    plt.plot(in_sample_log_trimmed.index, in_sample_log_trimmed.values, label=f'Predicted ARIMA', color='orange')
    plt.title('Actual vs Predicted ARIMA (Log-transformed model)')
    plt.xlabel('Date')
    plt.ylabel('Adj Close in $$$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return monthly_data_trimmed, in_sample_log_trimmed

# --------------------------
# 6. Forecasting
# --------------------------
def forecast_arima(model_fit, dp, forecast_steps=24):
    # Generate future Fourier features using the dp DeterministicProcess
    future_exog = dp.out_of_sample(steps=forecast_steps)
    forecast_result = model_fit.get_forecast(steps=forecast_steps, exog=future_exog)
    forecast_log_mean = forecast_result.predicted_mean
    forecast_log_ci = forecast_result.conf_int()
    
    # Convert forecast back from log scale
    forecast_mean = np.exp(forecast_log_mean)
    forecast_ci = np.exp(forecast_log_ci)
    return forecast_mean, forecast_ci

def plot_forecast(monthly_data, forecast_mean, forecast_ci, title_suffix=""):
    forecast_x = forecast_mean.index.to_numpy()
    forecast_y = forecast_mean.to_numpy()
    lower = forecast_ci.iloc[:, 0].astype(float).to_numpy()
    upper = forecast_ci.iloc[:, 1].astype(float).to_numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data.index, monthly_data.values, label='Observed', color='blue')
    plt.plot(forecast_x, forecast_y, label='Forecast', color='green')
    plt.fill_between(forecast_x, lower, upper,
                     color='lightgreen', alpha=0.5, label='95% Confidence Interval')
    plt.title(f'ARIMA Forecast for Next {len(forecast_x)} Months {title_suffix}')
    plt.xlabel('Date')
    plt.ylabel('Adj Close in $$$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------
# 7. Evaluation Metrics
# --------------------------
def evaluate_arima(monthly_data, model_fit):
    # Calculate predictions on the in-sample (fitted) period
    in_sample_log_pred = np.exp(model_fit.fittedvalues)
    # Trim the first observation (because of the integrated order 1)
    monthly_data_trimmed = monthly_data.iloc[1:]
    in_sample_log_trimmed = in_sample_log_pred.iloc[1:]
    
    residuals = monthly_data_trimmed - in_sample_log_trimmed
    
    mae = mean_absolute_error(monthly_data_trimmed, in_sample_log_trimmed)
    mpe = np.mean((residuals / monthly_data_trimmed) * 100)
    mape = mean_absolute_percentage_error(monthly_data_trimmed, in_sample_log_trimmed) * 100
    rmse = np.sqrt(mean_squared_error(monthly_data_trimmed, in_sample_log_trimmed))
    acf1 = residuals.autocorr(lag=1)
    corr = np.corrcoef(monthly_data_trimmed, in_sample_log_trimmed)[0, 1]
    minmax = 1 - np.mean(np.minimum(monthly_data_trimmed, in_sample_log_trimmed) / np.maximum(monthly_data_trimmed, in_sample_log_trimmed))
    
    print("\nEvaluation Metrics of ARIMA Model:")
    print(f"MAE: {mae:.4f}")
    print(f"MPE (%): {mpe:.4f}")
    print(f"MAPE (%): {mape:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"ACF1: {acf1:.4f}")
    print(f"Correlation: {corr:.4f}")
    print(f"Min-Max Error: {minmax:.4f}")
    
    return {
        'MAE': mae,
        'MPE': mpe,
        'MAPE': mape,
        'RMSE': rmse,
        'ACF1': acf1,
        'Correlation': corr,
        'Min-Max Error': minmax
    }

# --------------------------
# 8. Main Pipeline Function
# --------------------------
def run_arima_pipeline(csv_file, forecast_steps=24, fourier_order=3):
    # Step 1: Load data and plot daily observations
    daily_data, monthly_data = load_and_resample_data(csv_file)
    plot_daily_data(daily_data)
    
    # Step 2: Stationarity Tests on Original Monthly Data
    print("Stationarity Tests on Original Monthly Data:")
    run_stationarity_tests(monthly_data, series_name="Monthly Adj Close")
    
    # Differencing and testing on differenced series
    diff_1 = monthly_data.diff().dropna()
    print("\nStationarity Tests on Differenced Monthly Data:")
    run_stationarity_tests(diff_1, series_name="Differenced Monthly Adj Close")
    plot_diff_acf_pacf(diff_1, title="Differenced Monthly Adj Close")
    
    # Step 3: Prepare Fourier Terms
    # For original data (if needed) or for log-data modeling.
    dp, _ = get_fourier_terms(monthly_data.index, order=fourier_order, constant=False)
    
    # Step 4: Log Transformation and Fourier terms for log-data
    log_data = np.log(monthly_data)
    dp_log, fourier_log_terms = get_fourier_terms(log_data.index, order=fourier_order, constant=False)
    
    # Step 5: Grid Search for Best ARIMA(p,1,q) Order on Log Data
    best_order, best_aic = grid_search_arima(log_data, exog=fourier_log_terms)
    
    # Step 6: Fit the Final ARIMA Model
    model_log_final_fit = fit_arima_model(log_data, order=best_order, exog=fourier_log_terms)
    plot_arima_diagnostics(model_log_final_fit)
    
    # Plot Actual vs Predicted for the in-sample period
    monthly_data_trimmed, in_sample_pred = plot_actual_vs_predicted(monthly_data, model_log_final_fit)
    
    # Step 7: Forecasting Future Values
    forecast_mean, forecast_ci = forecast_arima(model_log_final_fit, dp_log, forecast_steps=forecast_steps)
    plot_forecast(monthly_data, forecast_mean, forecast_ci, title_suffix=f"(ARIMA{best_order} + Fourier)")
    
    # Step 8: Evaluate In-Sample Performance
    metrics = evaluate_arima(monthly_data, model_log_final_fit)
    
    return {
        "monthly_data": monthly_data,
        "log_data": log_data,
        "best_order": best_order,
        "model_fit": model_log_final_fit,
        "forecast_mean": forecast_mean,
        "forecast_ci": forecast_ci,
        "metrics": metrics
    }

# --------------------------
# Run the ARIMA Pipeline
# --------------------------
if __name__ == "__main__":
    csv_file = "/Users/dimpu/Downloads/Downloads/AMZN.csv"  # Update with your file path if needed
    results = run_arima_pipeline(csv_file, forecast_steps=24, fourier_order=3)


# In[16]:


import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf

# --------------------------
# Set reproducibility
# --------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------
# Data preprocessing function
# --------------------------
def preprocess_data(csv_file):
    # Load CSV file
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Resample to monthly frequency (using the last observation of each month)
    df_monthly = df['Adj Close'].resample('M').last().to_frame(name='adj_close')
    
    # Add seasonality features
    df_monthly['month'] = df_monthly.index.month
    df_monthly['month_sin'] = np.sin(2 * np.pi * df_monthly['month'] / 12)
    df_monthly['month_cos'] = np.cos(2 * np.pi * df_monthly['month'] / 12)
    
    # Prepare features and scale data for all features
    features = ['adj_close', 'month_sin', 'month_cos']
    data = df_monthly[features].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Fit a separate scaler for the target (adj_close) for proper inverse transformation
    adj_close_scaler = MinMaxScaler()
    adj_close_scaler.fit(df_monthly[['adj_close']])
    
    return df_monthly, data_scaled, adj_close_scaler

# --------------------------
# Sequence creation function
# --------------------------
def create_sequences(data, time_steps, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_horizon + 1):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps:i+time_steps+forecast_horizon, 0])
    return np.array(X), np.array(y)

# --------------------------
# Model building function
# --------------------------
def build_model(model_type, input_shape, forecast_horizon):
    model = Sequential()
    if model_type.upper() == "LSTM":
        model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='tanh', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16, activation='tanh', return_sequences=False))
        model.add(Dropout(0.2))
    elif model_type.upper() == "GRU":
        model.add(GRU(64, activation='tanh', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(32, activation='tanh', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(16, activation='tanh', return_sequences=False))
        model.add(Dropout(0.2))
    else:
        raise ValueError("Invalid model_type. Choose either 'LSTM' or 'GRU'.")
    
    # Output layer
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mse')
    return model

# --------------------------
# Training function
# --------------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=1):
    early_stop = EarlyStopping(patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    return history

# --------------------------
# Forecasting function
# --------------------------
def make_forecast(model, data_scaled, time_steps, forecast_horizon, adj_close_scaler, df_monthly):
    last_window = data_scaled[-time_steps:]
    last_window = last_window.reshape(1, time_steps, data_scaled.shape[1])
    forecast_scaled = model.predict(last_window, verbose=1)
    forecast_values = adj_close_scaler.inverse_transform(forecast_scaled)[0]
    
    forecast_dates = pd.date_range(df_monthly.index[-1] + pd.offsets.MonthBegin(), periods=forecast_horizon, freq='M')
    df_forecast = pd.DataFrame({'forecast': forecast_values}, index=forecast_dates)
    
    # Estimate 95% confidence interval using residual std from training predictions
    train_pred = model.predict(X_train, verbose=1)
    train_pred_inv = adj_close_scaler.inverse_transform(train_pred)
    train_actual_inv = adj_close_scaler.inverse_transform(y_train)
    residual_std = np.std(train_actual_inv - train_pred_inv)
    ci_upper = df_forecast['forecast'] + 1.96 * residual_std
    ci_lower = df_forecast['forecast'] - 1.96 * residual_std
    
    return df_forecast, ci_lower, ci_upper

# --------------------------
# Evaluation & plotting function
# --------------------------
def evaluate_and_plot(model, X_val, y_val, adj_close_scaler):
    # Predict on the validation set
    val_pred_scaled = model.predict(X_val, verbose=1)
    val_pred = adj_close_scaler.inverse_transform(val_pred_scaled)
    val_actual = adj_close_scaler.inverse_transform(y_val)
    
    y_pred_all = val_pred.flatten()
    y_actual_all = val_actual.flatten()
    
    # Compute Evaluation Metrics
    mae = mean_absolute_error(y_actual_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_actual_all, y_pred_all))
    mape = mean_absolute_percentage_error(y_actual_all, y_pred_all)
    corr = np.corrcoef(y_actual_all, y_pred_all)[0, 1]
    acf1 = pd.Series(y_actual_all - y_pred_all).autocorr(lag=1)
    me = np.mean(y_actual_all - y_pred_all)
    mpe = np.mean((y_actual_all - y_pred_all) / y_actual_all * 100)
    minmax = 1 - np.mean(np.minimum(y_actual_all, y_pred_all) / np.maximum(y_actual_all, y_pred_all))
    
    metrics_test = pd.DataFrame({
        'Metric': ['ME', 'MAE', 'MPE', 'MAPE', 'RMSE', 'ACF1', 'Correlation', 'Min-Max Error'],
        'Value': [me, mae, mpe, mape, rmse, acf1, corr, minmax]
    })
    
    print("\nEvaluation Metrics (Test Set):")
    print(metrics_test)
    
    # Plot predicted vs actual
    plt.figure(figsize=(14, 6))
    plt.plot(y_actual_all, label='Actual (Test Set)', color='tab:blue')
    plt.plot(y_pred_all, label='Predicted (Test Set)', linestyle='--', color='green')
    plt.title("Predicted vs Actual on Test Set")
    plt.xlabel("Time Steps")
    plt.ylabel("Adjusted Close")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return metrics_test

# --------------------------
# Plot training history
# --------------------------
def plot_training_history(history, model_type):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='tab:blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f"Training vs Validation Loss ({model_type})")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------
# Plot forecast with confidence interval
# --------------------------
def plot_forecast(df_monthly, df_forecast, ci_lower, ci_upper, model_type):
    plt.figure(figsize=(14, 6))
    plt.plot(df_monthly.index, df_monthly['adj_close'], label='Historical Data', color='tab:blue')
    plt.plot(df_forecast.index, df_forecast['forecast'], 'o-', label=f'{model_type} Forecast', color='darkorange')
    plt.fill_between(df_forecast.index, ci_lower, ci_upper, color='skyblue', alpha=0.5, label='95% Confidence Interval')
    plt.title(f"Amazon Stock Forecast Next {len(df_forecast)} Months ({model_type})")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------
# Main function to run the full pipeline
# --------------------------
def run_model(model_type, csv_file, time_steps=30, forecast_horizon=24):
    print(f"=== Running {model_type} model ===")
    
    # Preprocess data
    df_monthly, data_scaled, adj_close_scaler = preprocess_data(csv_file)
    
    # Create sequences
    X, y = create_sequences(data_scaled, time_steps, forecast_horizon)
    
    # Split into training and validation (90% training)
    split = int(len(X) * 0.9)
    global X_train, y_train  # used in forecast function for CI estimation
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    # Build the chosen model (LSTM or GRU)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(model_type, input_shape, forecast_horizon)
    
    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)
    plot_training_history(history, model_type)
    
    # Generate forecast and plot forecast with confidence interval
    df_forecast, ci_lower, ci_upper = make_forecast(model, data_scaled, time_steps, forecast_horizon, adj_close_scaler, df_monthly)
    plot_forecast(df_monthly, df_forecast, ci_lower, ci_upper, model_type)
    
    # Evaluate on the test set and plot predictions vs actual
    metrics = evaluate_and_plot(model, X_val, y_val, adj_close_scaler)
    
    return model, history, df_forecast, metrics

# --------------------------
# Run for both LSTM and GRU
# --------------------------
if __name__ == "__main__":
    csv_file = "/Users/dimpu/Downloads/Downloads/AMZN.csv"
    
    # Run LSTM model
    lstm_model, lstm_history, lstm_forecast, lstm_metrics = run_model("LSTM", csv_file)
    
    # Run GRU model
    gru_model, gru_history, gru_forecast, gru_metrics = run_model("GRU", csv_file)


# In[ ]:




