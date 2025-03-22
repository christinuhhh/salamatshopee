import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# Custom CSS for Dark Orange Accent
# =============================================================================
st.markdown(
    """
    <style>
    /* Headings in dark orange */
    h1, h2, h3, h4, h5, h6 {
        color: #FF8C00 !important;
    }
    /* Buttons in dark orange */
    .stButton > button {
        background-color: #FF8C00;
        color: white;
    }
    /* Add some padding to the main container */
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hybrid ARIMA + XGBoost Forecasting App")
st.write("""
This application loads a processed CSV file, fits an ARIMA model on the target, trains an XGBoost model on the ARIMA residuals, and produces forecasts.
""")

# =============================================================================
# SECTION 0: Forecast Volume by Station and Area Cluster Preselection
# =============================================================================
st.header("Forecast Volume by Station and Area Cluster (Next 30 Days)")

# Mapping data from your table
data = {
    "station_id": [7, 9, 11, 13, 18, 19, 20, 273, 434, 903, 1232, 1242, 1365, 1476, 1705, 1707, 1770, 2180, 2201, 2266, 2268, 2384, 2647, 2680, 3062],
    "forecast_volume": [202, 535, 960, 370, 206, 171, 800, 120, 714, 221, 566, 314, 430, 558, 187, 472, 199, 971, 763, 230, 761, 408, 869, 443, 591],
    "area_cluster": [
        "NCR2 Metro Manila East",
        "NCR1 Metro Manila West",
        "NCR1 Metro Manila West",
        "NCR1 Metro Manila West",
        "NCR2 Metro Manila East",
        "NCR2 Metro Manila East",
        "SOL1 Laguna/Tanay",
        "SOL2 Cavite",
        "NCR1 Metro Manila West",
        "NOL5 Bulacan",
        "NCR2 Metro Manila East",
        "NCR2 Metro Manila East",
        "NCR2 Metro Manila East",
        "SOL2 Cavite",
        "SOL2 Cavite",
        "NCR1 Metro Manila West",
        "NCR1 Metro Manila West",
        "NCR1 Metro Manila West",
        "NCR2 Metro Manila East",
        "NCR1 Metro Manila West",
        "SOL2 Cavite",
        "NCR2 Metro Manila East",
        "NCR2 Metro Manila East",
        "NCR2 Metro Manila East",
        "NCR1 Metro Manila West"
    ]
}

df_mapping = pd.DataFrame(data)
area_clusters = sorted(df_mapping["area_cluster"].unique())
selected_cluster = st.selectbox("Select Area Cluster", area_clusters)

# Filter the mapping based on the selected area cluster
filtered_mapping = df_mapping[df_mapping["area_cluster"] == selected_cluster]
st.write(f"Forecast Volume for Area Cluster: **{selected_cluster}**")
st.dataframe(filtered_mapping)

# Let the user choose a station within the selected area cluster
selected_station = st.selectbox("Select Station", filtered_mapping["station_id"].tolist())
forecast_vol = filtered_mapping.loc[filtered_mapping["station_id"] == selected_station, "forecast_volume"].values[0]
st.write(f"**Station {selected_station} Forecast Volume:** {forecast_vol}")

# =============================================================================
# SECTION 1: Load and Preprocess Data for Forecasting
# =============================================================================

# Define station mapping to CSV file index
station_mapping = {
    7: 0,
    9: 1,
    11: 2,
    13: 3,
    18: 4,
    19: 5,
    20: 6,
    273: 7,
    434: 8,
    903: 9,
    1232: 10,
    1242: 11,
    1365: 12,
    1476: 13,
    1705: 14,
    1707: 15,
    1770: 16,
    2180: 17,
    2201: 18,
    2266: 19,
    2268: 20,
    2384: 21,
    2647: 22,
    2680: 23,
    3062: 24
}

station_index = station_mapping[selected_station]
csv_file = f"csv_files/Processed_Stations/station_{station_index}_with_sale_events.csv"
st.write(f"Selected Station for Forecasting: **{selected_station}** (File: `{csv_file}`)")

# Load CSV file
try:
    df = pd.read_csv(csv_file)
    st.success(f"Loaded data from {csv_file}")
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Preprocess the data
if 'report_date' not in df.columns:
    st.error("Column 'report_date' not found in the dataset.")
    st.stop()

df['report_date'] = pd.to_datetime(df['report_date'])
df = df.sort_values(by='report_date').reset_index(drop=True)

features = [
    'rr_pu_vol_lag1',
    'total_delivery_vol_lag1',
    'rr_pu_vol_ma7',
    'total_delivery_vol_ma7',
    'dayofweek',
    'month',
    'user_delivery_interaction',
    'sale_event'
]
target = 'rr_pu_vol'

missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

df = df.dropna(subset=features + [target]).copy()

# =============================================================================
# SECTION 2: Historical Forecast (Model Fit)
# =============================================================================
st.header("Historical Forecast (Model Fit)")

# Split data: training (before December 2024) and testing (December 2024 onward)
train = df[df['report_date'] < "2024-12-01"].copy()
test = df[df['report_date'] >= "2024-12-01"].copy()

if train.empty or test.empty:
    st.error("Either the training or testing set is empty. Check your 'report_date' range.")
    st.stop()

# Fit ARIMA on training data
arima_order = (1, 1, 1)
st.write(f"Using ARIMA order: {arima_order}")

try:
    arima_fit = ARIMA(train[target], order=arima_order).fit()
except Exception as e:
    st.error(f"Error fitting ARIMA model: {e}")
    st.stop()

# Get ARIMA fitted values for training and forecast for testing period
train['arima_fitted'] = arima_fit.fittedvalues
try:
    forecast_steps = len(test)
    test_forecast = arima_fit.forecast(steps=forecast_steps)
    test['arima_forecast'] = test_forecast
except Exception as e:
    st.error(f"Error during ARIMA forecasting: {e}")
    st.stop()

test['arima_forecast'] = test['arima_forecast'].ffill().bfill()

# Compute residuals and train XGBoost for residual correction
train['residual'] = train[target] - train['arima_fitted']
test['residual'] = test[target] - test['arima_forecast']

x_train, x_val, y_train, y_val = train_test_split(train[features], train['residual'], test_size=0.2, random_state=42)
xgb_model = XGBRegressor(n_estimators=500,
                         learning_rate=0.05,
                         max_depth=7,
                         subsample=0.9,
                         colsample_bytree=0.8,
                         eval_metric='mae',
                         random_state=42)
try:
    xgb_model.fit(x_train, y_train)
except Exception as e:
    st.error(f"Error training XGBoost: {e}")
    st.stop()

xgb_val_preds = xgb_model.predict(x_val)
val_mae = mean_absolute_error(y_val, xgb_val_preds)
st.write(f"XGBoost Residual Model Validation MAE: {val_mae:.2f}")

residual_preds = xgb_model.predict(test[features])
residual_preds = np.nan_to_num(residual_preds)
test['final_forecast'] = test['arima_forecast'] + residual_preds
test['final_forecast'] = test['final_forecast'].ffill().bfill()

# Evaluate performance
mae = mean_absolute_error(test[target], test['final_forecast'])
rmse = np.sqrt(mean_squared_error(test[target], test['final_forecast']))
smape = 100 * np.mean(np.abs(test[target] - test['final_forecast']) /
                      ((np.abs(test[target]) + np.abs(test['final_forecast'])) / 2))
r2 = r2_score(test[target], test['final_forecast'])

st.subheader("Model Performance Metrics (Historical Forecast)")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**SMAPE:** {smape:.2f}%")
st.write(f"**R² Score:** {r2:.3f}")

# Plot historical forecast using Plotly
fig_hist = px.line(test,
                   x='report_date',
                   y=[target, 'final_forecast'],
                   labels={'value': 'Return/Refund Parcel Volume', 'report_date': 'Date'},
                   title='Hybrid ARIMA + XGBoost Forecast (Model Fit)')
# Set the final forecast line to our dark orange accent
fig_hist.for_each_trace(lambda t: t.update(name=t.name.replace("variable=", ""),
                                             line=dict(color="#FF8C00") if "final_forecast" in t.name else {}))
st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# SECTION 3: 30-Day Forecast (Next 30 Days Prediction)
# =============================================================================
st.header("30-Day Forecast (Next 30 Days Prediction)")

# Use all historical data up to January 1, 2025 for future prediction
train_future = df[df['report_date'] < "2025-01-01"].copy()
if train_future.empty:
    st.error("Training set for future prediction is empty. Check your 'report_date' range.")
    st.stop()

st.write("Training data from **{}** to **{}**".format(
    train_future['report_date'].min().date(), train_future['report_date'].max().date()))

# Create future dates for the next 30 days
last_date = train_future['report_date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
future_df = pd.DataFrame({'report_date': future_dates})

# Create date-based features for future_df
future_df['dayofweek'] = future_df['report_date'].dt.dayofweek
future_df['month'] = future_df['report_date'].dt.month
# For remaining features, use the last observed values from train_future
for feat in ['rr_pu_vol_lag1', 'total_delivery_vol_lag1', 'rr_pu_vol_ma7', 'total_delivery_vol_ma7', 'user_delivery_interaction', 'sale_event']:
    future_df[feat] = train_future[feat].iloc[-1]
future_X = future_df[features]

# Fit ARIMA on the future training data and forecast 30 days ahead
try:
    arima_model_future = ARIMA(train_future[target], order=arima_order).fit()
except Exception as e:
    st.error(f"Error fitting ARIMA model for future prediction: {e}")
    st.stop()

train_future['arima_fitted'] = arima_model_future.fittedvalues
try:
    jan_arima_forecast = arima_model_future.forecast(steps=30)
except Exception as e:
    st.error(f"Error forecasting with ARIMA for future prediction: {e}")
    st.stop()

# Compute residuals and train XGBoost for future correction
train_future['residual'] = train_future[target] - train_future['arima_fitted']
x_train_future, x_val_future, y_train_future, y_val_future = train_test_split(
    train_future[features], train_future['residual'], test_size=0.2, random_state=42)
xgb_model_future = XGBRegressor(n_estimators=500,
                                learning_rate=0.05,
                                max_depth=7,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                eval_metric='mae',
                                random_state=42)
try:
    xgb_model_future.fit(x_train_future, y_train_future)
except Exception as e:
    st.error(f"Error training XGBoost for future prediction: {e}")
    st.stop()

xgb_val_preds_future = xgb_model_future.predict(x_val_future)
val_mae_future = mean_absolute_error(y_val_future, xgb_val_preds_future)
st.write(f"XGBoost Future Prediction Residual Model Validation MAE: {val_mae_future:.2f}")

jan_residual_preds = xgb_model_future.predict(future_X)
jan_residual_preds = np.nan_to_num(jan_residual_preds)
jan_final_forecast = jan_arima_forecast + jan_residual_preds

# (Optional) Generate synthetic actuals to compute R² for demonstration
noise_std = np.std(train_future['residual'])
synthetic_jan_actual = jan_final_forecast + np.random.normal(loc=0, scale=noise_std, size=jan_final_forecast.shape)
jan_r2 = r2_score(synthetic_jan_actual, jan_final_forecast)
st.write(f"Synthetic Future Forecast R² Score (for demonstration): {jan_r2:.3f}")

# Plot future forecast using Plotly (showing last 60 days of historical data and future forecast)
historical_slice = train_future[train_future['report_date'] >= (last_date - pd.Timedelta(days=60))]
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=historical_slice['report_date'], y=historical_slice[target],
                                mode='lines', name='Historical Data'))
fig_future.add_trace(go.Scatter(x=future_df['report_date'], y=jan_final_forecast,
                                mode='lines+markers', name='30-Day Forecast', 
                                line=dict(dash='dash', color="#FF8C00")))
fig_future.update_layout(title="Hybrid ARIMA + XGBoost 30-Day Forecast",
                         xaxis_title="Date",
                         yaxis_title="Return/Refund Parcel Volume")
st.plotly_chart(fig_future, use_container_width=True)

# =============================================================================
# SECTION 4: Future Forecast Table (Only)
# =============================================================================
st.header("Future Forecast Table")

# Prepare future forecast table from future_df
future_forecast_df = future_df.copy()
future_forecast_df["Forecast"] = jan_final_forecast.tolist() if hasattr(jan_final_forecast, 'tolist') else jan_final_forecast

# Show only the date and forecast number
st.dataframe(future_forecast_df[['report_date', "Forecast"]])

