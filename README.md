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
  
### Why Shift Columns?
In time series forecasting, shifting columns is a common technique used to create features that capture historical patterns and relationships in the data. Here's why this method is valuable:

```python
def shift_columns(df, target_column, n_shifts):
    """
    Tạo ra tập dữ liệu dự đoán giá (Close) từ dữ liệu từ n (n_shifts) phút trước.
  
    Parameters:
    df (pd.DataFrame): DataFrame chứa các cột gốc.
    target_column (str): Cột mục tiêu để tạo các cột dịch chuyển thêm một bước so với các cột khác.
    n_shifts (int): Số lượng bước dịch chuyển cho các cột mới.
  
    Returns:
    pd.DataFrame: DataFrame đã được mở rộng với các cột dịch chuyển.
    """
    df_shifted = df.copy()
  
    columns = df_shifted.columns.tolist()
    for column in columns:
      if column == target_column:
        for i in range(1, n_shifts + 2):
            df_shifted[f'{column}{i}'] = df_shifted[column].shift(-i)
      else:
        for i in range(1, n_shifts + 1):
            df_shifted[f'{column}{i}'] = df_shifted[column].shift(-i)
  
    df_shifted = df_shifted.dropna(subset=[f'{columns[0]}{n_shifts}'])
  
    return df_shifted
```

1. Lag Features Creation
Shifting columns generates lag features, which represent past values of a variable at different time steps. These lag features help the model learn temporal dependencies and patterns in the data. For instance, if you are forecasting stock prices, past prices (lags) can be indicative of future price movements.

2. Enhanced Model Inputs
By including historical values of the target variable (e.g., the closing price of a stock) and other features (e.g., open, high, low prices), you provide the model with more context and information. This can improve the model's ability to capture trends and seasonality in the data.

3. Predictive Power
Shifting columns allows the creation of features that can predict future values based on past observations. For example, if the model is trained with features such as 'Close-1', 'Close-2', and 'Close-3', it can use these to predict 'Close', improving forecasting accuracy.

4. Handling Non-Stationarity
Time series data often exhibit non-stationarity, where statistical properties like mean and variance change over time. By creating lag features, you help the model understand and adapt to these changes, making it more robust in predicting future values.

Example
In the provided code, the shift_columns function shifts each column by a specified number of steps (e.g., 1, 2, 3, etc.) and creates new features for each step. This helps in forming a dataset where past values of 'Close', 'Open', 'High', and 'Low' are used to predict future values of 'Close', capturing the dynamics of the time series.


## Setup

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/OtaTran241/Stock_Price_Prediction.git
    cd Stock_Price_Prediction
    ```

## Custom Train Test Split
In this project, a custom train_test_split function was implemented instead of using the standard train_test_split from scikit-learn. This decision was made due to the time series nature of the data and the need for data cleaning.

### Why Not Use scikit-learn's train_test_split?
The standard train_test_split function in scikit-learn randomly splits the dataset into training and testing sets. While effective for many machine learning tasks, this approach is not suitable for time series data where the temporal order of data points is crucial. Randomly splitting time series data can lead to data leakage, where future data points are included in the training set, leading to overly optimistic results.

### Custom `train_test_split` Implementation
The custom `train_test_split` function preserves the temporal order of the data by splitting it sequentially. Additionally, it removes any rows with NaN values from both the features and the target variables. This ensures that the model is trained on past data and tested on future data, reflecting real-world scenarios and providing clean datasets for training and evaluation.

Example Code:
```python
def train_test_split(X, Y, train_size=0.8):
    """
    Splits the data into training and testing sets based on the train_size ratio and removes NaN values.

    Parameters:
    X (list or np.array or pd.DataFrame): Input data (features) to be split.
    Y (list or np.array or pd.Series): Target data (labels) corresponding to the input data.
    train_size (float): Proportion of the data used for training. Default is 0.8 (80%).

    Returns:
    x_train_clean (pd.DataFrame or np.array): Training input data with NaN values removed.
    x_test_clean (pd.DataFrame or np.array): Testing input data with NaN values removed.
    y_train_clean (pd.Series or np.array): Target labels for the training data with NaN values removed.
    y_test_clean (pd.Series or np.array): Target labels for the testing data with NaN values removed.
    """
    x_train = X[:int(train_size * len(X))]
    y_train = Y[:int(train_size * len(X))]
    x_test = X[int(train_size * len(X)):]
    y_test = Y[int(train_size * len(X)):]

    train_df = pd.concat([pd.DataFrame(x_train), pd.Series(y_train)], axis=1).dropna()
    x_train_clean = train_df.iloc[:, :-1]
    y_train_clean = train_df.iloc[:, -1]
    
    test_df = pd.concat([pd.DataFrame(x_test), pd.Series(y_test)], axis=1).dropna()
    x_test_clean = test_df.iloc[:, :-1]
    y_test_clean = test_df.iloc[:, -1]
    
    return x_train_clean, x_test_clean, y_train_clean, y_test_clean
```
This custom implementation ensures that the model's performance evaluation is realistic and that it generalizes well to future data points, while also providing clean datasets free of NaN values.

## Modeling with LazyPredict
Using `LazyPredict`, several regression models were tested on the dataset. The models were ranked based on their performance, particularly focusing on RMSE.

### Example Code:
```python
lazy_reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=mean_squared_error)
models, predictions = lazy_reg.fit(X_train_lazy, X_test_lazy, y_train_lazy, y_test_lazy)
```
## Top Model Selection
After identifying the best-performing models using `LazyPredict`, we further fine-tuned these models by applying hyperparameter optimization using `GridSearchCV`. This process involved evaluating multiple combinations of regressors and data scaling methods to determine the best model for predicting stock prices.

### Example Code:
```python
regressors = {
    'BayesianRidge': (BayesianRidge(), {'regressor__n_iter': [300, 500, 1000]}),
    'TransformedTargetRegressor': (TransformedTargetRegressor(regressor=LinearRegression()), {}),
    'LinearRegression': (LinearRegression(), {}),
    'OrthogonalMatchingPursuitCV': (OrthogonalMatchingPursuitCV(), {}),
    'RANSACRegressor': (RANSACRegressor(), {}),
    'LassoLarsIC': (LassoLarsIC(), {'regressor__criterion': ['aic', 'bic']}),
    'Lars': (Lars(), {'regressor__n_nonzero_coefs': [5, 10, 20]}),
    'LassoLarsCV': (LassoLarsCV(), {'regressor__cv': [5]}),
    'LinearSVR': (LinearSVR(), {'regressor__C': [0.1, 1, 10], 'regressor__epsilon': [0.1, 0.2]}),
    'Ridge': (Ridge(), {'regressor__alpha': [0.1, 1, 10]})
}

scalers={'Standard Scaler' :StandardScaler(),
         'Min Max Scaler':MinMaxScaler(),
         'Roobust Scaler': RobustScaler(),
         'No Scaler': None
        }

tscv = TimeSeriesSplit(n_splits=5)
rmse_scorer = make_scorer(lambda y_true, y_pred: -mean_squared_error(y_true, y_pred, squared=False)) 
```
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
