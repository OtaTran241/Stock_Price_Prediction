# Stock Price Prediction using Machine Learning and LSTM

This project focuses on predicting stock prices using various machine learning models and a Long Short-Term Memory (LSTM) neural network. The primary goal is to compare different regression models and select the best one for predicting future stock prices based on historical data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Custom Train Test Split](#custom-train-test-split)
- [Modeling with LazyPredict](#modeling-with-lazypredict)
- [Top Model Selection](#top-model-selection)
- [Advanced Modeling with LSTM](#advanced-modeling-with-lstm)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

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

### Why Create Additional `Close` Columns?

1. **Lagged Features for Prediction**:
   - **Problem**: Predicting future stock prices using only the current price might not be sufficient because stock prices are influenced by previous prices.
   - **Solution**: By creating lagged features (`Close1`, `Close2`, `Close3`, `Close4`), you provide the model with previous closing prices as input features. These columns represent the closing prices from one, two, three, and four time steps before the current row.

2. **Time Series Data Structure**:
   - **Problem**: Stock price data is sequential, meaning that the value at any given time depends on previous values.
   - **Solution**: By introducing lagged variables, you transform the dataset into a time series format where each row contains information from several past time steps. This allows the model to learn temporal dependencies, which is crucial for making accurate predictions.

3. **Future Prediction Target**:
   - **Problem**: To predict the stock price at a future time (e.g., 4 steps ahead), you need to define this future price as the target variable.
   - **Solution**: The `Close4` column is used as the target variable (`Y`). It represents the closing price four time steps ahead. This setup allows the model to learn the relationship between past prices (in `Close`, `Close1`, `Close2`, `Close3`) and the future price (`Close4`).

### Summary

Creating additional `Close` columns by shifting the original `Close` column is a common technique in time series forecasting. It enables the model to:
- Learn from past data points to make predictions about the future.
- Capture temporal dependencies in the stock prices.
- Predict a future price several steps ahead, which is the ultimate goal of the project.

By doing this, you effectively prepare the data for a machine learning model that can forecast stock prices based on historical data, making your predictions more robust and accurate.


## Setup

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/OtaTran241/Stock_Price_Prediction.git
    cd Stock_Price_Prediction
    ```
## Custom Train Test Split
In this project, a custom `train_test_split` function was implemented instead of using the standard `train_test_split` from scikit-learn. This decision was made due to the time series nature of the data.

### Why Not Use scikit-learn's `train_test_split`?
The standard `train_test_split` function in scikit-learn randomly splits the dataset into training and testing sets. While this approach is generally effective for many machine learning tasks, it is not suitable for time series data where the temporal order of data points is crucial. Randomly splitting time series data can lead to data leakage, where future data points are included in the training set, leading to overly optimistic results.

### Custom `train_test_split` Implementation
The custom `train_test_split` function preserves the temporal order of the data by splitting it sequentially. This ensures that the model is trained on past data and tested on future data, which more closely resembles real-world scenarios.

### Example Code:
```python
def train_test_split(X, Y, train_size=0.8):
    x_train = X[:int(train_size*len(X))]
    y_train = Y[:int(train_size*len(X))]
    x_test = X[int(train_size*len(X)):]
    y_test = Y[int(train_size*len(X))]
    return x_train, x_test, y_train, y_test
```
This approach ensures that the model's performance evaluation is realistic and that it generalizes well to future data points.

## Modeling with LazyPredict
Using `LazyPredict`, several regression models were tested on the dataset. The models were ranked based on their performance, particularly focusing on RMSE.

### Example Code:
```python
lazy_reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=mean_squared_error)
models, predictions = lazy_reg.fit(X_train_lazy, X_test_lazy, y_train_lazy, y_test_lazy)
```
## Top Model Selection
After identifying the best-performing models using `LazyPredict`, we further fine-tuned these models by applying hyperparameter optimization using `GridSearchCV`. This process involved evaluating multiple combinations of regressors and data scaling methods to determine the best model for predicting stock prices.

### Regressors and Scalers

1. **Regressors**:
   The `regressors` dictionary defines a set of machine learning models (regressors) and their respective hyperparameters for tuning. For each regressor, the dictionary contains:
   - The model itself, such as `BayesianRidge` or `LinearRegression`.
   - A corresponding dictionary of hyperparameters that `GridSearchCV` will search over to find the best parameters.

   Example Regressors:
   - **BayesianRidge**: A linear regression model with a Bayesian approach. The hyperparameter `n_iter` specifies the number of iterations.
   - **LassoLarsIC**: A Lasso model with Least Angle Regression (LARS) that uses information criteria (AIC or BIC) to select the best model.
   - **LinearSVR**: A linear Support Vector Regression model, with `C` and `epsilon` as hyperparameters controlling the regularization and margin of tolerance, respectively.

2. **Scalers**:
   The `scalers` dictionary defines different data scaling techniques that normalize or standardize the input features before training:
   - **Standard Scaler**: Standardizes features by removing the mean and scaling to unit variance.
   - **MinMax Scaler**: Scales each feature to a given range, typically between 0 and 1.
   - **Robust Scaler**: Scales features using statistics that are robust to outliers.
   - **No Scaler**: Indicates that no scaling is applied.

### Cross-Validation with Time Series Split

- **TimeSeriesSplit (tscv)**: A cross-validation technique specifically designed for time series data. It splits the data into training and testing sets while preserving the order of the data points, ensuring that future data points are never used in the training process.

### RMSE Scorer

- **RMSE Scorer**: Root Mean Squared Error (RMSE) is used as the scoring metric. The lower the RMSE, the better the model's predictions align with the actual values. In this code, `make_scorer` is used to create a custom scoring function that returns the negative RMSE. This is because `GridSearchCV` maximizes the scoring metric, so by using negative RMSE, we effectively minimize the RMSE.

### Summary

By combining different regressors, scalers, and utilizing `GridSearchCV` with time series cross-validation, this approach systematically searches for the best model configuration. This ensures that the selected model is not only accurate but also well-suited to the temporal nature of the stock price data.

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
## Results
The best model selected was OrthogonalMatchingPursuitCV with a Standard Scaler. The model achieved nearly perfect accuracy with an RMSE close to zero.

The LSTM model also performed well, achieving high accuracy in predicting stock prices.

## Usage
### Predicting Stock Prices
After training, you can use the saved models to predict future stock prices:

Load the trained model:
```python
loaded_model = load('stock_price_prediction.joblib')
```
Predict stock prices:
```python
y_pred = loaded_model.predict(X_test)
```
### Visualization
Visualize the actual vs predicted stock prices:

```python
plt.plot(results['Date'], results['Actual'], label='Actual Price', color='blue', linewidth=2)
plt.plot(results['Date'], results['Predicted'], label='Predicted Price', color='red', linewidth=2)
plt.show()
```
## Contributing
Contributions are welcome! If you have any ideas for improving the model or adding new features, feel free to submit a pull request or send an email to [tranducthuan220401@gmail.com](mailto:tranducthuan220401@gmail.com).
