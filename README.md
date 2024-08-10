# Stock Price Prediction using Machine Learning and LSTM

This project focuses on predicting stock prices using various machine learning models and a Long Short-Term Memory (LSTM) neural network. The primary goal is to compare different regression models and select the best one for predicting future stock prices based on historical data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup and Requirements](#setup-and-requirements)
- [Modeling with LazyPredict](#modeling-with-lazypredict)
- [Top Model Selection](#top-model-selection)
- [Advanced Modeling with LSTM](#advanced-modeling-with-lstm)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to predict stock prices by:
1. Applying different machine learning models using the `LazyPredict` library to find the best performing model.
2. Implementing an LSTM model for time series prediction.
3. Comparing and evaluating the models based on various metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

## Dataset
The dataset used in this project consists of stock prices with the following columns:
- **Ticker**: Stock symbol.
- **Date/Time**: Date and time of the transaction.
- **Open**: Opening price.
- **High**: Highest price during the time period.
- **Low**: Lowest price during the time period.
- **Close**: Closing price.
- **Volume**: Trading volume.
- **Open Interest**: Number of outstanding contracts.

## Setup and Requirements
### Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow/Keras
- LazyPredict
- Joblib

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/stock-price-prediction.git
    cd stock-price-prediction
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Modeling with LazyPredict
Using `LazyPredict`, several regression models were tested on the dataset. The models were ranked based on their performance, particularly focusing on RMSE.

### Example Code:
```python
lazy_reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=mean_squared_error)
models, predictions = lazy_reg.fit(X_train_lazy, X_test_lazy, y_train_lazy, y_test_lazy)
```
## Top Model Selection
The top models from LazyPredict were further evaluated using GridSearchCV with various scalers to find the best combination of model and scaler.

### Example Code:
```python
best_pipeline = Pipeline([
    ('scaler', best_scaler_),
    ('classifier', best_model.set_params(**best_params_cleaned))
])
best_pipeline.fit(X_train, y_train)
```
## Advanced Modeling with LSTM
An LSTM model was implemented for time series prediction. The LSTM model was trained and evaluated on the stock price data.

### LSTM Architecture:
Input Layer: Sequence of stock prices.
LSTM Layers: Two LSTM layers with 50 units each.
Dense Layer: Final dense layer for prediction.
### Example Code:
```python
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train, y_train, epochs=16, batch_size=128, verbose=2)
```
## Evaluation Metrics
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
R² Score
### Example Metrics:
```python
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)
```
