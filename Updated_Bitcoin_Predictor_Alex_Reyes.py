# Alex Reyes
# CAP5619 Spring 2025 0V61
# Linear Regression Model to predict Bitcoin Prices
# Using chainlet, satoshi, and price data (lagged)

# Import required libraries
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from datetime import datetime
import shap
import warnings
from matplotlib import pyplot as plt


warnings.simplefilter('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load datasets
price_daily = pd.read_csv("pricedBitcoin2009-2018.csv")
satoshis = pd.read_csv("AmoChainletsInTime.txt", sep="\t")
chainlets = pd.read_csv("OccChainletsInTime.txt", sep="\t")

# Differentiate between satoshi and chainlet amount columns
satoshis.columns = (satoshis.columns[:2]).union(satoshis.columns[2:] + 'Sats', sort=False)
chainlets.columns = (chainlets.columns[:2]).union(chainlets.columns[2:] + 'Occ', sort=False)

# Merge datasets and drop unnecessary columns
total_data = pd.merge(
    pd.merge(satoshis, chainlets, on=['year', 'day'], how='outer'),
    price_daily, on=['year', 'day'], how='outer').drop(columns=['totaltxOcc', 'year', 'day']).dropna().reset_index(drop=True)

# Convert date column to datetime
total_data['date'] = pd.to_datetime(total_data['date'])

# Calculate VWAP as a feature - volume weighted average price on a 7-day basis
total_data['price_volume'] = total_data['price'] * total_data['totaltxSats']

total_data['rolling_price_volume'] = total_data['price_volume'].rolling(
    window=7).sum()

total_data['rolling_volume'] = total_data['totaltxSats'].rolling(window=7).sum()

total_data['vwap_7d'] = total_data['rolling_price_volume'] / total_data['rolling_volume']


# Create lagged prices feature
# 10 days was the optimal days for this model in regard to RMSE
lags = 10
for lag in range(1, lags + 1):
    total_data[f'lag_{lag}'] = total_data['price'].shift(lag)
total_data.dropna(inplace=True)

# Separate test and train data with datetime library
split_date = datetime(2017, 12, 1)
train_val = total_data[total_data['date'] < split_date]
test_data = total_data[total_data['date'] >= split_date]

# This print statement helps confirm the train and validation data
# come before the test data, retaining temporal order and avoiding
# data leakage
print("Dates for training and validation data\n", train_val['date'])
print("\nDates for test data\n",test_data['date'],"\n")

# Set the selected features; does not include year, day, date.
# price feature is included to assign to y_train, y_val, and y_test
# price feature will be dropped from x_train, x_val, and x_test
features = ['lag_1', 'lag_7', 'lag_8', 'lag_4', 'lag_2', 'lag_6', 'vwap_7d', '1:20Sats', '1:20Occ', 'price']
train_val = train_val[features]
test_data = test_data[features]

# Create lists to store models and model results from each fold training round below
regressors = []
fold_metrics = []

# Use time series split to ensure temporal data remains in original order
# This ensures no data leakage occurs
tscv = TimeSeriesSplit(n_splits=12)
for fold, (train_index, val_index) in enumerate(tscv.split(train_val)):
    scaler = StandardScaler()

    # Create training and validation sets and scale data
    train_data = train_val.iloc[train_index]
    val_data = train_val.iloc[val_index]

    x_train = train_data[features].drop(columns='price')
    x_train = scaler.fit_transform(x_train)
    y_train = train_data['price']

    x_val = val_data[features].drop(columns='price')
    x_val = scaler.transform(x_val)
    y_val = val_data['price']

    # Train the model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Create model predictions and performance scores
    y_val_pred = regressor.predict(x_val)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    r2 = r2_score(y_val, y_val_pred)

    # Store the trained model and model results
    regressors.append(regressor)
    fold_metrics.append({'fold': fold + 1, 'rmse': rmse, 'r2': r2})

    # Print fold model results
    print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Create test data
x_test = test_data[features]
x_test = x_test.drop(columns='price')

# Scale using the final folds scaler from time series split above
x_test = scaler.transform(x_test)
y_test = test_data['price']

# Test performance with MSE, RMSE, R2, and Accuracy
# Using model from final TSCV split
y_pred_test = regressors[11].predict(x_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)
test_accuracy = np.mean(np.sign(y_pred_test) == np.sign(y_test))

print("\nTest Mean Squared Error: ", mse_test)
print("Test Root Mean Squared Error: ", rmse_test)
print("Test R2 Score: ", r2_test)
print("Test Accuracy Score: ", test_accuracy, '\n')

# Output December 2017 predictions and save predictions to CSV
dates = pd.date_range(start='2017-12-01', end='2017-12-31').strftime('%m/%d/%Y').to_frame(name='date').reset_index(
    drop=True)
dates['predicted_price'] = y_pred_test
dates.to_csv('Bitcoin_Predictions_Alex_Reyes.csv', index=False)
print(dates, "\n")

# See which feature are important
# SHAP Explanation Using KernelExplainer
explainer = shap.KernelExplainer(regressors[11].predict, x_train)
shap_values = explainer.shap_values(x_test)

# Since we are dealing with large bitcoin prices as the predicted value, we will
# also normalize shap values for better interpretability
# x_test values above are already scaled
normalized_shap_values = shap_values / np.abs(shap_values).sum(axis=1, keepdims=True)

# Remove price from features list to use in SHAP summary plot
features = [col for col in features if col != 'price']

# Plot SHAP values for first 5 days of December 2017 individually
# Then print to terminal text explaining the impact of each feature
# for that specific day
print()
for day in range(5):

    # Print SHAP values for this day to console
    print(f"Feature SHAP values for {dates.iloc[day, 0]}:")
    for feature, shap_value in zip(features, normalized_shap_values[day]):

        if shap_value > 0:
            print(f"- {feature} with a SHAP value of {shap_value:.6f} increased the models prediction from the baseline predicted value")
        elif shap_value < 0:
            print(f"- {feature} with a SHAP value of {shap_value:.6f} decreased the models prediction from the baseline predicted value")
        else:
            print(f"- {feature} with a SHAP value of {shap_value:.6f} did not increase or decrease the models baseline prediction value")
    print()

    # Print top 5 most influential SHAP values for this day to console
    print(f"Top 5 most influential SHAP values for {dates.iloc[day, 0]}:")
    for feature, shap_value in sorted(zip(features, normalized_shap_values[day]), key=lambda x: abs(x[1]), reverse=True)[:5]:
        if shap_value > 0:
            print(
                f"- {feature} with a SHAP value of {shap_value:.6f} increased the models prediction from the baseline predicted value")
        elif shap_value < 0:
            print(
                f"- {feature} with a SHAP value of {shap_value:.6f} decreased the models prediction from the baseline predicted value")
        else:
            print(
                f"- {feature} with a SHAP value of {shap_value:.6f} did not increase or decrease the models baseline prediction value")

    # Adjust SHAP values and features to correct format for summary plot
    day_shap = np.expand_dims(normalized_shap_values[day], axis=0)
    day_x_test = np.expand_dims(x_test[day], axis=0)

    # Plot shap values for the day
    shap.summary_plot(
        shap_values=day_shap,
        features=day_x_test,
        feature_names=features,
        show=False
                             )
    plt.title(f"SHAP Summary Plot For {dates.iloc[day, 0]}")
    plt.show()
    plt.close()

    print("\n=======================================================================================================\n")

# Print SHAP summary plot for all of December 2017
shap.summary_plot(normalized_shap_values, x_test, feature_names=features, show=False)
plt.title(f"SHAP Summary Plot For 12/1/2017 to 12/31/2017")
plt.show()
plt.close()









