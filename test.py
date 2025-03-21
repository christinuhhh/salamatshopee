import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Hybrid ARIMA + XGBoost Forecasting App")
st.write("""
This application loads a processed CSV file, fits an ARIMA model on the target, trains an XGBoost model on the ARIMA residuals, and then produces a final forecast.
""")

# -------------------------------
# 1. Load the CSV File
# -------------------------------
# Adjust the file path as needed. 
# In this example, the file is located in the 'csv_files/Processed_Stations' folder.
csv_file = "csv_files/Processed_Stations/station_0_with_sale_events.csv"
try:
    df = pd.read_csv(csv_file)
    st.success(f"Loaded data from {csv_file}")
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# -------------------------------
# 2. Preprocess the Data
# -------------------------------
if 'report_date' not in df.columns:
    st.error("Column 'report_date' not found in the dataset.")
    st.stop()

df['report_date'] = pd.to_datetime(df['report_date'])
df = df.sort_values(by='report_date').reset_index(drop=True)

# Define features and target
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

# Check for missing required columns
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    st.error(f"The following required columns are missing from the dataset: {missing_cols}")
    st.stop()

# Drop rows with missing values in required columns
df = df.dropna(subset=features + [target]).copy()

# -------------------------------
# 3. Split Data into Training and Test Sets
# -------------------------------
# Using data before December 2024 for training and December 2024 for testing.
train = df[df['report_date'] < "2024-12-01"].copy()
test = df[df['report_date'] >= "2024-12-01"].copy()

if train.empty or test.empty:
    st.error("Either the training or testing set is empty. Check your 'report_date' range in the CSV.")
    st.stop()

# -------------------------------
# 4. Fit ARIMA on Training Data
# -------------------------------
arima_order = (1, 1, 1)
st.write(f"Using ARIMA order: {arima_order}")

try:
    arima_fit = ARIMA(train[target], order=arima_order).fit()
except Exception as e:
    st.error(f"Error fitting ARIMA model: {e}")
    st.stop()

# Get in-sample fitted values for training and forecast for testing
train['arima_fitted'] = arima_fit.fittedvalues
try:
    forecast_steps = len(test)
    test_forecast = arima_fit.forecast(steps=forecast_steps)
    test['arima_forecast'] = test_forecast
except Exception as e:
    st.error(f"Error during ARIMA forecasting: {e}")
    st.stop()

# Fill any missing forecast values
test['arima_forecast'] = test['arima_forecast'].fillna(method='ffill').fillna(method='bfill')

# -------------------------------
# 5. Compute Residuals
# -------------------------------
train['residual'] = train[target] - train['arima_fitted']
test['residual'] = test[target] - test['arima_forecast']

# -------------------------------
# 6. Train XGBoost to Predict Residuals
# -------------------------------
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
    st.error(f"Error training the XGBoost model: {e}")
    st.stop()

# Validate XGBoost on the residual validation set
xgb_val_preds = xgb_model.predict(x_val)
val_mae = mean_absolute_error(y_val, xgb_val_preds)
st.write(f"XGBoost Residual Model Validation MAE: {val_mae:.2f}")

# Predict residuals on the test set
residual_preds = xgb_model.predict(test[features])
residual_preds = np.nan_to_num(residual_preds)

# -------------------------------
# 7. Compute Final Forecast
# -------------------------------
test['final_forecast'] = test['arima_forecast'] + residual_preds
test['final_forecast'] = test['final_forecast'].fillna(method='ffill').fillna(method='bfill')

# -------------------------------
# 8. Evaluate Model Performance
# -------------------------------
mae = mean_absolute_error(test[target], test['final_forecast'])
rmse = np.sqrt(mean_squared_error(test[target], test['final_forecast']))
smape = 100 * np.mean(np.abs(test[target] - test['final_forecast']) /
                      ((np.abs(test[target]) + np.abs(test['final_forecast'])) / 2))
r2 = r2_score(test[target], test['final_forecast'])

st.subheader("Model Performance Metrics")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**SMAPE:** {smape:.2f}%")
st.write(f"**R² Score:** {r2:.3f}")

# -------------------------------
# 9. Visualize the Results with Plotly
# -------------------------------
# Plotting the actual values and final forecast over time.
fig = px.line(test,
              x='report_date',
              y=[target, 'final_forecast'],
              labels={'value': 'Return/Refund Parcel Volume', 'report_date': 'Date'},
              title='Hybrid ARIMA + XGBoost Forecast (Model Fit)')

# Update the legend names
fig.for_each_trace(lambda t: t.update(name=t.name.replace("variable=", "")))

st.plotly_chart(fig, use_container_width=True)
