# improvedpossiblitly_full.py


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import io
import warnings
import math
warnings.filterwarnings("ignore")

# Optional libraries (import if available)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

# Prophet can be installed as "prophet"
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# LSTM/TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Statsmodels for ARIMA/SARIMAX
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

st.set_page_config(page_title="AECP — ImprovedPossiblitly", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helper utilities
# ---------------------------
def safe_min_nonzero(series, floor=1e-6):
    m = series.min()
    return m if m > 0 else floor

def mape(y_true, y_pred):
    denom = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def nrmse(y_true, y_pred, kind='mean'):
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    if kind == 'range':
        denom = (np.max(y_true) - np.min(y_true))
    else:
        denom = np.mean(y_true)
    denom = denom if denom != 0 else 1e-6
    return rmse_val / denom

def evaluate_predictions(y_true, y_pred):
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nrmse_mean = nrmse(y_true, y_pred, kind='mean')
    nrmse_range = nrmse(y_true, y_pred, kind='range')
    return {
        'RMSE': float(rmse_val),
        'MAE': float(mae_val),
        'MAPE (%)': float(mape_val),
        'R2': float(r2),
        'NRMSE_mean': float(nrmse_mean),
        'NRMSE_range': float(nrmse_range)
    }

def add_time_features(df):
    # Ensure year column present and convert to index datetime
    df = df.copy()
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)
        df = df.sort_values('year').reset_index(drop=True)
        df.index = pd.to_datetime(df['year'], format='%Y')
        df.drop(columns=['year'], inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # assume index is years numeric
        try:
            df.index = pd.to_datetime(df.index.astype(int), format='%Y')
        except Exception:
            pass

    # create lag and rolling features
    df['pop_lag1'] = df['population'].shift(1)
    df['pop_lag2'] = df['population'].shift(2)
    df['pop_diff1'] = df['population'].diff(1)
    df['pop_roll_mean_3'] = df['population'].rolling(window=3, min_periods=1).mean()
    df['pop_roll_mean_5'] = df['population'].rolling(window=5, min_periods=1).mean()

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df

def align_years_for_plot(x_years, y_vals):
    # x_years: pandas Index of datetime; y_vals: array-like
    x_arr = np.array(x_years)
    y_arr = np.array(y_vals)
    if len(x_arr) == len(y_arr):
        return x_arr, y_arr
    elif len(x_arr) > len(y_arr):
        return x_arr[-len(y_arr):], y_arr
    else:
        # If predictions longer than x, create new years extension starting after last x
        last_year = pd.DatetimeIndex(x_arr).year.max()
        extra_years = np.arange(last_year + 1, last_year + 1 + (len(y_arr) - len(x_arr)))
        extra_dt = pd.to_datetime(extra_years.astype(str), format='%Y')
        full_x = np.concatenate([x_arr, extra_dt])
        return full_x, y_arr

def create_lstm_sequences(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

# ---------------------------
# Sidebar: Data input & options
# ---------------------------
st.sidebar.title("AECP — Controls")
st.sidebar.markdown("Upload your CSV or generate synthetic/sample datasets. Required columns: `year,population,temperature,rainfall,habitat_index`")

data_source = st.sidebar.radio("Data source", options=["Upload CSV", "Use Sample Dodo Dataset", "Generate Synthetic"])

uploaded_df = None

if data_source == "Upload CSV":
    f = st.sidebar.file_uploader("Upload CSV (annual)", type=['csv'])
    if f is not None:
        try:
            uploaded_df = pd.read_csv(f)
            st.sidebar.success("CSV loaded.")
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")

elif data_source == "Use Sample Dodo Dataset":
    # synthesize Dodo dataset 1450-1649 (200 rows) to be self-contained
    years = np.arange(1450, 1650)
    n = len(years)
    rng = np.random.default_rng(42)
    population = np.round(np.linspace(20000, 300, n) * (1 + rng.normal(0, 0.03, n))).astype(int)
    population = np.maximum(population, 100)
    temperature = np.round(np.linspace(25.0, 25.8, n) + rng.normal(0, 0.05, n), 2)
    rainfall = np.round(np.linspace(1800, 1600, n) + rng.normal(0, 12, n), 1)
    habitat_index = np.round(np.clip(np.linspace(0.95, 0.25, n) + rng.normal(0, 0.02, n), 0.05, 1.0), 3)
    uploaded_df = pd.DataFrame({
        'year': years,
        'population': population,
        'temperature': temperature,
        'rainfall': rainfall,
        'habitat_index': habitat_index
    })
    st.sidebar.success("Using built-in Dodo dataset (1450-1649).")

elif data_source == "Generate Synthetic":
    start_year = st.sidebar.number_input("Start year", value=1800, step=1)
    end_year = st.sidebar.number_input("End year", value=2020, step=1)
    initial_pop = st.sidebar.number_input("Initial population", value=100000, step=100)
    final_pop = st.sidebar.number_input("Approx final population", value=10000, step=100)
    temp_start = st.sidebar.number_input("Temp start (°C)", value=25.0)
    temp_end = st.sidebar.number_input("Temp end (°C)", value=27.0)
    rain_start = st.sidebar.number_input("Rain start (mm)", value=2000)
    rain_end = st.sidebar.number_input("Rain end (mm)", value=1800)
    habitat_start = st.sidebar.number_input("Habitat start (0-1)", value=0.95, min_value=0.01, max_value=1.0)
    habitat_end = st.sidebar.number_input("Habitat end (0-1)", value=0.40, min_value=0.01, max_value=1.0)
    noise_frac = st.sidebar.slider("Population noise fraction", min_value=0.0, max_value=0.2, value=0.06)
    if st.sidebar.button("Generate Synthetic"):
        years = np.arange(start_year, end_year + 1)
        n = len(years)
        rng = np.random.default_rng(123)
        # exponential-like decay for realism
        t = np.linspace(0, 1, n)
        pop_clean = initial_pop * ( (final_pop / initial_pop) ** t )
        pop_noise = pop_clean * rng.normal(1.0, noise_frac, size=n)
        pop_vals = np.round(np.clip(pop_noise, 1, None)).astype(int)
        temp = np.round(np.linspace(temp_start, temp_end, n) + rng.normal(0, 0.08, n), 2)
        rain = np.round(np.linspace(rain_start, rain_end, n) + rng.normal(0, 6, n), 1)
        habitat = np.round(np.clip(np.linspace(habitat_start, habitat_end, n) + rng.normal(0, 0.02, n), 0.01, 1.0), 3)
        uploaded_df = pd.DataFrame({
            'year': years,
            'population': pop_vals,
            'temperature': temp,
            'rainfall': rain,
            'habitat_index': habitat
        })
        st.sidebar.success("Synthetic dataset generated.")

# If no data selected, stop
if uploaded_df is None:
    st.info("Select a data source from the sidebar to begin.")
    st.stop()

# ---------------------------
# Main app: preprocessing & display
# ---------------------------
st.title("AECP — ImprovedPossiblitly")
st.markdown("""
This app preprocesses population time series, compares multiple forecasting methods, and produces multi-year forecasts.
Ensure your CSV has columns: `year,population,temperature,rainfall,habitat_index`.
""")

# Basic validation
required_cols = ['year', 'population', 'temperature', 'rainfall', 'habitat_index']
missing = [c for c in required_cols if c not in uploaded_df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Ensure proper types and continuity
df = uploaded_df.copy()
df['year'] = df['year'].astype(int)
df = df.sort_values('year').reset_index(drop=True)

# If duplicate years, average
if df['year'].duplicated().any():
    st.warning("Duplicate years found — aggregating by mean.")
    df = df.groupby('year', as_index=False).mean()

# warn if less than recommended rows
if len(df) < 150:
    st.warning(f"Dataset has {len(df)} rows. Recommended >=150 rows for robust training; proceed with caution.")

st.write(f"Data years: {df['year'].min()} — {df['year'].max()}  |  Rows: {len(df)}")
st.dataframe(df.head(10))

# Add engineered time features
df_proc = add_time_features(df)
st.subheader("Processed Data (first 10 rows)")
st.dataframe(df_proc.head(10))

# quick plot
st.subheader("Historical Population")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df_proc.index.year, df_proc['population'], linewidth=2)
ax.set_xlabel('Year'); ax.set_ylabel('Population'); ax.grid(alpha=0.2)
st.pyplot(fig)
plt.close(fig)

# ---------------------------
# Modeling UI options
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Modeling Options")

forecast_years = st.sidebar.slider("Forecast horizon (years)", min_value=10, max_value=200, value=50, step=10)
test_size = st.sidebar.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

model_items = ['ARIMA', 'SARIMAX', 'RandomForest', 'GradientBoosting']
if XGB_AVAILABLE:
    model_items.append('XGBoost')
if LGB_AVAILABLE:
    model_items.append('LightGBM')
if PROPHET_AVAILABLE:
    model_items.append('Prophet')
if TF_AVAILABLE:
    model_items.append('LSTM')

model_choices = st.sidebar.multiselect("Which models to run?", options=model_items, default=['RandomForest', 'ARIMA'])

log_transform_target = st.sidebar.checkbox("Log-transform target (log1p)", value=True)
scale_features = st.sidebar.checkbox("Scale numeric features (StandardScaler)", value=False)

# Model hyperparameters
n_estimators = st.sidebar.slider("Tree models: n_estimators", 50, 1000, 200, step=50)
max_depth = st.sidebar.slider("Tree models: max_depth (0 = None)", 0, 20, 6)

# LSTM hyperparams
if TF_AVAILABLE and 'LSTM' in model_choices:
    lstm_window = st.sidebar.slider("LSTM window size", 3, 30, 8)
    lstm_epochs = st.sidebar.number_input("LSTM epochs", value=50, min_value=5)
    lstm_batch = st.sidebar.number_input("LSTM batch size", value=16, min_value=1)

# ---------------------------
# Prepare supervised dataset
# ---------------------------
# We'll predict next year's population (t+1) using t features.
supervised = df_proc.copy()
supervised['target'] = supervised['population'].shift(-1)
supervised = supervised.dropna(subset=['target']).copy()

# Split train/test by time
split_idx = int(len(supervised) * (1 - test_size))
supervised_train = supervised.iloc[:split_idx].copy()
supervised_test = supervised.iloc[split_idx:].copy()

# Choose modeling features
feature_cols = ['population', 'pop_lag1', 'pop_lag2', 'pop_diff1', 'pop_roll_mean_3', 'pop_roll_mean_5',
                'temperature', 'rainfall', 'habitat_index']

X_train = supervised_train[feature_cols].values
y_train = supervised_train['target'].values
X_test = supervised_test[feature_cols].values
y_test = supervised_test['target'].values

# Optionally scale features
scaler_X = None
if scale_features:
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

# Optionally log-transform target
if log_transform_target:
    y_train_trans = np.log1p(y_train)
    y_test_trans = np.log1p(y_test)
else:
    y_train_trans = y_train.copy()
    y_test_trans = y_test.copy()

results = {}

# ---------------------------
# Train & evaluate chosen models
# ---------------------------

# RandomForest
if 'RandomForest' in model_choices:
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_trans)
    y_pred_tr = rf.predict(X_train)
    y_pred_te = rf.predict(X_test)
    if log_transform_target:
        y_pred_te_inv = np.expm1(y_pred_te)
        y_pred_tr_inv = np.expm1(y_pred_tr)
        y_test_inv = np.expm1(y_test_trans)
        y_train_inv = np.expm1(y_train_trans)
    else:
        y_pred_te_inv = y_pred_te
        y_pred_tr_inv = y_pred_tr
        y_test_inv = y_test
        y_train_inv = y_train
    results['RandomForest'] = {
        'model': rf,
        'train_metrics': evaluate_predictions(y_train_inv, y_pred_tr_inv),
        'test_metrics': evaluate_predictions(y_test_inv, y_pred_te_inv),
        'y_pred_test': y_pred_te_inv
    }

# GradientBoosting
if 'GradientBoosting' in model_choices:
    gb = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42)
    gb.fit(X_train, y_train_trans)
    y_pred_tr = gb.predict(X_train)
    y_pred_te = gb.predict(X_test)
    if log_transform_target:
        y_pred_te_inv = np.expm1(y_pred_te)
        y_pred_tr_inv = np.expm1(y_pred_tr)
        y_test_inv = np.expm1(y_test_trans)
        y_train_inv = np.expm1(y_train_trans)
    else:
        y_pred_te_inv = y_pred_te
        y_pred_tr_inv = y_pred_tr
        y_test_inv = y_test
        y_train_inv = y_train
    results['GradientBoosting'] = {
        'model': gb,
        'train_metrics': evaluate_predictions(y_train_inv, y_pred_tr_inv),
        'test_metrics': evaluate_predictions(y_test_inv, y_pred_te_inv),
        'y_pred_test': y_pred_te_inv
    }

# XGBoost
if XGB_AVAILABLE and 'XGBoost' in model_choices:
    try:
        xgbr = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42, n_jobs=-1, verbosity=0)
        xgbr.fit(X_train, y_train_trans)
        y_pred_tr = xgbr.predict(X_train)
        y_pred_te = xgbr.predict(X_test)
        if log_transform_target:
            y_pred_te_inv = np.expm1(y_pred_te)
            y_pred_tr_inv = np.expm1(y_pred_tr)
            y_test_inv = np.expm1(y_test_trans)
            y_train_inv = np.expm1(y_train_trans)
        else:
            y_pred_te_inv = y_pred_te
            y_pred_tr_inv = y_pred_tr
            y_test_inv = y_test
            y_train_inv = y_train
        results['XGBoost'] = {
            'model': xgbr,
            'train_metrics': evaluate_predictions(y_train_inv, y_pred_tr_inv),
            'test_metrics': evaluate_predictions(y_test_inv, y_pred_te_inv),
            'y_pred_test': y_pred_te_inv
        }
    except Exception as e:
        st.warning(f"XGBoost training failed: {e}")

# LightGBM
if LGB_AVAILABLE and 'LightGBM' in model_choices:
    try:
        lgbm = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42)
        lgbm.fit(X_train, y_train_trans)
        y_pred_tr = lgbm.predict(X_train)
        y_pred_te = lgbm.predict(X_test)
        if log_transform_target:
            y_pred_te_inv = np.expm1(y_pred_te)
            y_pred_tr_inv = np.expm1(y_pred_tr)
            y_test_inv = np.expm1(y_test_trans)
            y_train_inv = np.expm1(y_train_trans)
        else:
            y_pred_te_inv = y_pred_te
            y_pred_tr_inv = y_pred_tr
            y_test_inv = y_test
            y_train_inv = y_train
        results['LightGBM'] = {
            'model': lgbm,
            'train_metrics': evaluate_predictions(y_train_inv, y_pred_tr_inv),
            'test_metrics': evaluate_predictions(y_test_inv, y_pred_te_inv),
            'y_pred_test': y_pred_te_inv
        }
    except Exception as e:
        st.warning(f"LightGBM training failed: {e}")

# Prophet (univariate time-series on population)
if PROPHET_AVAILABLE and 'Prophet' in model_choices:
    try:
        # Prepare DataFrame with ds,y
        prophet_df = df_proc[['population']].reset_index().rename(columns={'index': 'ds', 'population': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        # Fit on train period
        train_prophet = prophet_df.iloc[:int(len(prophet_df)*(1-test_size))].copy()
        model_prophet = Prophet()
        model_prophet.fit(train_prophet)
        # Predict on test horizon
        future = model_prophet.make_future_dataframe(periods=len(df_proc) - len(train_prophet), freq='Y')
        forecast = model_prophet.predict(future)
        # Align predictions to test_df years
        pred_vals = forecast['yhat'].values[-len(supervised_test):]
        actual_vals = supervised_test['target'].values
        results['Prophet'] = {
            'model': model_prophet,
            'test_metrics': evaluate_predictions(actual_vals.astype(float), pred_vals.astype(float)),
            'y_pred_test': pred_vals.astype(float)
        }
    except Exception as e:
        st.warning(f"Prophet training failed: {e}")

# ARIMA / SARIMAX (univariate)
if STATSMODELS_AVAILABLE and ('ARIMA' in model_choices or 'SARIMAX' in model_choices):
    try:
        series_train = df_proc['population'].iloc[:int(len(df_proc)*(1-test_size))]
        if 'ARIMA' in model_choices:
            arima = ARIMA(series_train, order=(3,1,1)).fit()
            arima_pred_test = arima.forecast(steps=len(df_proc) - len(series_train))
            actual_vals = supervised_test['target'].values
            pred_vals = np.array(arima_pred_test).astype(float)[:len(actual_vals)]
            results['ARIMA'] = {
                'model': arima,
                'test_metrics': evaluate_predictions(actual_vals.astype(float), pred_vals),
                'y_pred_test': pred_vals
            }
        if 'SARIMAX' in model_choices:
            sarima = SARIMAX(series_train, order=(1,1,1), seasonal_order=(0,0,0,0)).fit(disp=False)
            sarima_pred_test = sarima.forecast(steps=len(df_proc) - len(series_train))
            actual_vals = supervised_test['target'].values
            pred_vals = np.array(sarima_pred_test).astype(float)[:len(actual_vals)]
            results['SARIMAX'] = {
                'model': sarima,
                'test_metrics': evaluate_predictions(actual_vals.astype(float), pred_vals),
                'y_pred_test': pred_vals
            }
    except Exception as e:
        st.warning(f"ARIMA/SARIMAX training failed: {e}")

# LSTM (sequence modeling) - using population series only
if TF_AVAILABLE and 'LSTM' in model_choices:
    try:
        pop_vals = df_proc['population'].values.reshape(-1,1).astype(float)
        minmax = MinMaxScaler()
        pop_scaled = minmax.fit_transform(pop_vals)
        window = lstm_window if 'lstm_window' in locals() else 8
        X_seq, y_seq = create_lstm_sequences(pop_scaled, window)
        # train/test split
        seq_split = int(len(X_seq) * (1 - test_size))
        X_seq_train, X_seq_test = X_seq[:seq_split], X_seq[seq_split:]
        y_seq_train, y_seq_test = y_seq[:seq_split], y_seq[seq_split:]
        if len(X_seq_train) < 5:
            st.warning("Not enough sequences to train LSTM reliably.")
        else:
            model_lstm = Sequential([
                LSTM(64, activation='tanh', input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mse')
            model_lstm.fit(X_seq_train, y_seq_train, epochs=lstm_epochs, batch_size=lstm_batch, verbose=0)
            y_seq_pred = model_lstm.predict(X_seq_test)
            y_seq_pred_inv = minmax.inverse_transform(y_seq_pred).flatten()
            y_seq_test_inv = minmax.inverse_transform(y_seq_test).flatten()
            results['LSTM'] = {
                'model': model_lstm,
                'test_metrics': evaluate_predictions(y_seq_test_inv, y_seq_pred_inv),
                'y_pred_test': y_seq_pred_inv
            }
    except Exception as e:
        st.warning(f"LSTM failed: {e}")

# ---------------------------
# Model comparison table
# ---------------------------
st.subheader("Model Comparison (Test Set Metrics)")
summary_rows = []
for name, info in results.items():
    m = info.get('test_metrics', {})
    summary_rows.append({
        'Model': name,
        'RMSE': m.get('RMSE', np.nan),
        'MAE': m.get('MAE', np.nan),
        'MAPE (%)': m.get('MAPE (%)', np.nan),
        'R2': m.get('R2', np.nan),
        'NRMSE_mean': m.get('NRMSE_mean', np.nan)
    })

if len(summary_rows) == 0:
    st.warning("No models were trained — ensure you selected models and dataset is valid.")
else:
    summary_df = pd.DataFrame(summary_rows).sort_values('RMSE')
    st.dataframe(summary_df.style.format({
        'RMSE': '{:,.2f}', 'MAE': '{:,.2f}', 'MAPE (%)': '{:.2f}', 'R2': '{:.3f}', 'NRMSE_mean': '{:.3f}'
    }))

# ---------------------------
# Diagnostics and plotting (safe alignment)
# ---------------------------
st.subheader("Prediction Diagnostics & Forecasting")

if len(results) == 0:
    st.info("No model results to plot.")
    st.stop()

# Pick best model by RMSE
best_model_name = min(results.keys(), key=lambda k: results[k]['test_metrics']['RMSE'])
st.info(f"Best model by RMSE: {best_model_name}")
best_info = results[best_model_name]
y_pred_test = np.array(best_info['y_pred_test']).astype(float)
# For LSTM, supervised_test index may not align; handle below

# Align test years and predictions safely
try:
    # supervised_test.index is datetime index for the rows we predicted (target = next year)
    x_years = supervised_test.index
    x_aligned, y_aligned = align_years_for_plot(x_years, y_pred_test)
    # Convert x_aligned to years for plotting
    x_years_nums = pd.DatetimeIndex(x_aligned).year
except Exception:
    # fallback: just create numeric year index matching len of predictions
    x_years_nums = np.arange(df_proc.index.year.max() - len(y_pred_test) + 1, df_proc.index.year.max() + 1)

# Plot historical and predicted
fig, ax = plt.subplots(figsize=(12,6))
plt.style.use('seaborn-v0_8-whitegrid')
ax.plot(df_proc.index.year, df_proc['population'], label='Historical Population', linewidth=3, color='#3498db')

# Plot test predictions (aligned)
try:
    ax.plot(x_years_nums, y_aligned, label=f'Predicted ({best_model_name})', linestyle='--', linewidth=2, color='#e74c3c')
except Exception as e:
    st.warning(f"Could not plot test predictions: {e}")

# Iterative multi-year forecast for tree-like models
future_forecast = []
if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
    model = best_info['model']
    # start from last row of df_proc
    current_row = df_proc.iloc[-1:].copy()
    for i in range(forecast_years):
        feat = current_row[feature_cols].values.astype(float).reshape(1, -1)
        if scale_features and scaler_X is not None:
            feat = scaler_X.transform(feat)
        pred_trans = model.predict(feat)
        if log_transform_target:
            pred = np.expm1(pred_trans)[0]
        else:
            pred = pred_trans[0]
        future_forecast.append(float(pred))
        # update current_row
        new_row = current_row.copy()
        new_row['population'] = pred
        new_row['pop_lag2'] = new_row['pop_lag1']
        new_row['pop_lag1'] = pred
        new_row['pop_diff1'] = new_row['population'] - new_row['pop_lag1']
        new_row['pop_roll_mean_3'] = (new_row['pop_roll_mean_3']*2 + pred)/3
        new_row['pop_roll_mean_5'] = (new_row['pop_roll_mean_5']*4 + pred)/5
        # environmental vars linear extrapolation based on last two historical points
        for col in ['temperature', 'rainfall', 'habitat_index']:
            last_two = df_proc[col].values[-2:]
            slope = (last_two[-1] - last_two[-2])
            new_row[col] = new_row[col] + slope
        new_year = current_row.index.year[0] + 1
        new_row.index = pd.to_datetime([str(new_year)], format='%Y')
        current_row = new_row

# ARIMA/SARIMAX/Prophet forecasting direct
elif best_model_name in ['ARIMA', 'SARIMAX', 'Prophet']:
    try:
        model = best_info['model']
        fc = model.forecast(steps=forecast_years)
        future_forecast = list(np.array(fc).astype(float))
    except Exception:
        future_forecast = []

# LSTM forecasting (simple iterative using last window)
elif best_model_name == 'LSTM' and TF_AVAILABLE:
    try:
        # reuse minmax from earlier LSTM preparation (we used local variable minmax in training block)
        # rebuild scaled series
        pop_vals = df_proc['population'].values.reshape(-1,1).astype(float)
        minmax_local = MinMaxScaler()
        pop_scaled_local = minmax_local.fit_transform(pop_vals)
        window = lstm_window if 'lstm_window' in locals() else 8
        last_seq = pop_scaled_local[-window:].reshape(1, window, 1)
        model = best_info['model']
        for i in range(forecast_years):
            pred_scaled = model.predict(last_seq)
            pred = float(minmax_local.inverse_transform(pred_scaled.reshape(-1,1))[0,0])
            future_forecast.append(pred)
            # shift last_seq
            last_seq = np.append(last_seq[:,1:,:], pred_scaled.reshape(1,1,1), axis=1)
    except Exception:
        future_forecast = []

# Plot future forecast safely
if len(future_forecast) > 0:
    try:
        future_years = np.arange(df_proc.index.year.max() + 1, df_proc.index.year.max() + 1 + len(future_forecast))
        # align lengths if needed (defensive)
        if len(future_years) != len(future_forecast):
            future_years = future_years[-len(future_forecast):]
        ax.plot(future_years, future_forecast, label='Future Forecast', linestyle=':', linewidth=2, color='#9b59b6')
    except Exception as e:
        st.warning(f"Could not plot future forecast: {e}")

# Optional extinction year detection (first year forecast <= 0)
extinction_year = None
for i, val in enumerate(future_forecast):
    try:
        if val <= 0:
            extinction_year = int(df_proc.index.year.max() + i + 1)
            break
    except Exception:
        pass

if extinction_year:
    ax.axvline(extinction_year, color='darkred', linestyle='--', linewidth=2, label=f'Predicted Extinction: {extinction_year}')
    ax.annotate(f'Extinction Year: {extinction_year}',
                (extinction_year, max(df_proc['population']) * 0.9),
                textcoords="offset points", xytext=(-10, 0), ha='right', color='darkred', fontsize=10)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Population Size', fontsize=12)
ax.set_title(f'Population Time Series & Forecast ({best_model_name})', fontsize=16)
ax.legend(loc='upper left')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)
plt.close(fig)

# ---------------------------
# Output metrics & download forecast CSV
# ---------------------------
st.markdown("---")
st.subheader("Selected Model Test Metrics")
st.write(f"Best model: **{best_model_name}**")
st.json(best_info['test_metrics'])

# Allow user to download forecast
if len(future_forecast) > 0:
    out_df = pd.DataFrame({
        'year': np.arange(df_proc.index.year.max() + 1, df_proc.index.year.max() + 1 + len(future_forecast)),
        'predicted_population': np.round(future_forecast).astype(int)
    })
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button("Download Forecast CSV", csv_buf.getvalue(), file_name=f'forecast_{best_model_name}.csv', mime='text/csv')

# Show model comparison summary again for convenience
st.markdown("---")
st.subheader("Model Comparison Summary")
st.dataframe(summary_df.style.format({
    'RMSE': '{:,.2f}', 'MAE': '{:,.2f}', 'MAPE (%)': '{:.2f}', 'R2': '{:.3f}', 'NRMSE_mean': '{:.3f}'
}))

# ---------------------------
# Tips & closing notes
# ---------------------------
st.markdown("""
### Tips to Improve Accuracy
- Use **log-transform** for targets that change exponentially (enabled in sidebar).
- Add more relevant features (e.g., human disturbance index, hunting pressure, conservation actions).
- Tune model hyperparameters using **TimeSeriesSplit** cross-validation.
- Use more data (longer yearly history) for LSTM or tree-based sequence methods.
- Be careful when forecasting far into the future: uncertainty grows quickly.

If you'd like, I can:
- Add **auto hyperparameter tuning** (GridSearch / RandomizedSearch) using TimeSeriesSplit.
- Add **prediction intervals** for ARIMA/Prophet forecasts.
- Save trained model artifacts to disk for later reuse.
""")

st.caption("AECP — ImprovedPossiblitly (full). If you want the app trimmed for lighter deployment or need a Dockerfile/requirements, tell me and I'll provide them.")

