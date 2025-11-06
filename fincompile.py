import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

st.title("Population Time Series Forecasting and Environmental Analysis")

# === File Upload ===
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Successfully loaded data.")

        # === Auto-Rename Columns to Match Expected Format ===
        rename_map = {
            'Year': 'year',
            'Years': 'year',
            'Population': 'population',
            'Pop': 'population',
            'Temp': 'temperature',
            'Temperature': 'temperature',
            'Rain': 'rainfall',
            'Rainfall_mm': 'rainfall',
            'Rainfall': 'rainfall',
            'Habitat': 'habitat_index',
            'HabitatIndex': 'habitat_index',
            'Habitat_Index': 'habitat_index'
        }
        df.rename(columns={col: rename_map.get(col, col) for col in df.columns}, inplace=True)

        # Convert 'year' column to datetime and set as index
        if 'year' not in df.columns:
            st.error("The file must include a 'year' column (or a recognizable equivalent).")
            st.stop()

        df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
        df.dropna(subset=['year'], inplace=True)
        df.set_index('year', inplace=True)

        # === Check required columns ===
        required_cols = ['population', 'temperature', 'rainfall', 'habitat_index']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing column: {col}. Please include all: {required_cols}")
                st.stop()

        # === Split Data ===
        train_size = int(len(df) * 0.8)
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

        # === User selects algorithm ===
        algorithm_choice = st.selectbox(
            "Choose forecasting algorithm:", ["ARIMA", "SARIMA", "LSTM"]
        )
        st.write(f"Selected forecasting algorithm: **{algorithm_choice}**")

        # === Model selection logic ===
        if algorithm_choice == 'LSTM':
            # Normalize
            scaler = MinMaxScaler()
            train_data_scaled = scaler.fit_transform(train_data[['population']])
            test_data_scaled = scaler.transform(test_data[['population']])
            window_size = 5

            def create_sequences(data, window):
                X, y = [], []
                for i in range(len(data) - window):
                    X.append(data[i:i+window])
                    y.append(data[i+window])
                return np.array(X), np.array(y)

            X_train, y_train = create_sequences(train_data_scaled, window_size)
            X_test, y_test = create_sequences(test_data_scaled, window_size)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Build and Train LSTM
            model = Sequential([
                LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            with st.spinner("Training LSTM model..."):
                model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

            lstm_predictions_scaled = model.predict(X_test, verbose=0)
            lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
            mse = mean_squared_error(test_data['population'].iloc[window_size:], lstm_predictions)
            rmse = np.sqrt(mse)
            forecast_label = 'LSTM Forecast'
            forecast_test = lstm_predictions

            # LSTM future
            data_scaled_full = scaler.fit_transform(df[['population']])
            X_full, y_full = create_sequences(data_scaled_full, window_size)
            X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
            model_full = Sequential([
                LSTM(64, activation='relu', input_shape=(X_full.shape[1], X_full.shape[2])),
                Dense(1)
            ])
            model_full.compile(optimizer='adam', loss='mse')
            with st.spinner("Training LSTM model on full data for future forecast..."):
                model_full.fit(X_full, y_full, epochs=50, batch_size=8, verbose=0)

            future_steps = 100
            last_seq_full = data_scaled_full[-window_size:]
            lstm_preds_full = []
            for _ in range(future_steps):
                input_seq_full = last_seq_full.reshape(1, window_size, 1)
                pred_full = model_full.predict(input_seq_full, verbose=0)
                lstm_preds_full.append(pred_full[0, 0])
                last_seq_full = np.append(last_seq_full[1:], pred_full.reshape(1, 1), axis=0)
            forecast_full = scaler.inverse_transform(np.array(lstm_preds_full).reshape(-1, 1))

            extinction_year = None
            for i, val in enumerate(forecast_full):
                if val[0] <= 0:
                    extinction_year = df.index.year.max() + i + 1
                    break

        elif algorithm_choice == 'ARIMA':
            arima_model = ARIMA(train_data['population'], order=(3,1,1)).fit()
            arima_forecast_test = arima_model.forecast(steps=len(test_data))
            mse = mean_squared_error(test_data['population'], arima_forecast_test)
            rmse = np.sqrt(mse)
            forecast_label = 'ARIMA Forecast'
            forecast_test = arima_forecast_test

            future_steps = 100
            arima_model_full = ARIMA(df['population'], order=(3,1,1)).fit()
            forecast_full = arima_model_full.forecast(steps=future_steps)
            extinction_year = None
            for i, val in enumerate(forecast_full):
                if val <= 0:
                    extinction_year = df.index.year.max() + i + 1
                    break

        elif algorithm_choice == 'SARIMA':
            sarima_model = SARIMAX(train_data['population'], order=(1,1,1), seasonal_order=(0,0,0,0)).fit(disp=False)
            sarima_forecast_test = sarima_model.forecast(steps=len(test_data))
            mse = mean_squared_error(test_data['population'], sarima_forecast_test)
            rmse = np.sqrt(mse)
            forecast_label = 'SARIMA Forecast'
            forecast_test = sarima_forecast_test

            future_steps = 100
            sarima_model_full = SARIMAX(df['population'], order=(1,1,1), seasonal_order=(0,0,0,0)).fit(disp=False)
            forecast_full = sarima_model_full.forecast(steps=future_steps)
            extinction_year = None
            for i, val in enumerate(forecast_full):
                if val <= 0:
                    extinction_year = df.index.year.max() + i + 1
                    break

        # === Plot Forecast ===
        years_future = np.arange(df.index.year.max() + 1, df.index.year.max() + 1 + future_steps)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df.index.year, df['population'], label='Historical Population', linewidth=2)
        ax.plot(years_future, forecast_full, label=forecast_label, linestyle='--', color='orange')
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')
        ax.set_title(f'Population Forecast ({algorithm_choice})')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # === Display Evaluation Metrics ===
        st.write("### Model Evaluation Metrics:")
        st.write(f"{algorithm_choice} - MSE: {mse:.2f}, RMSE: {rmse:.2f}")

        # === Show Predicted Extinction Year ===
        if extinction_year:
            st.warning(f"{algorithm_choice} predicts extinction in the year: {extinction_year}")
        else:
            st.success(f"{algorithm_choice} did not predict extinction within {future_steps} years.")

        # === Thriving Population Range Input ===
        min_pop, max_pop = int(df['population'].min()), int(df['population'].max())
        low_thriving = st.number_input("Enter lower bound of thriving population range:", min_value=min_pop, max_value=max_pop, value=min_pop)
        high_thriving = st.number_input("Enter upper bound of thriving population range:", min_value=low_thriving, max_value=max_pop, value=max_pop)

        # === Filter and Report Environmental Vectors ===
        thriving_years = df[(df['population'] >= low_thriving) & (df['population'] <= high_thriving)]
        if thriving_years.empty:
            st.info("No years found in that thriving population range in the historical data.")
        else:
            environmental_means = thriving_years[['temperature', 'rainfall', 'habitat_index']].mean().round(2)
            st.write("#### Best Environmental and Climate Conditions for Survival (Metric Units):")
            st.write(f"Average Temperature: {environmental_means['temperature']} °C")
            st.write(f"Average Rainfall: {environmental_means['rainfall']} mm")
            st.write(f"Average Habitat Index: {environmental_means['habitat_index']}")

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
