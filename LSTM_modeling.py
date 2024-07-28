import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Data Collection
# Assuming you have downloaded the stock price data into a CSV file
data = pd.read_csv('path/to/your/stock_price_data.csv', date_parser=True)

# Data Preprocessing
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plotting the data
plt.figure(figsize=(14, 5))
plt.plot(data)
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Splitting into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshaping for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Model Development
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Model Training
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Model Evaluation
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse scaling
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Metrics
rmse_train = np.sqrt(mean_squared_error(y_train[0], train_predict[:, 0]))
mae_train = mean_absolute_error(y_train[0], train_predict[:, 0])
r2_train = r2_score(y_train[0], train_predict[:, 0])

rmse_test = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
mae_test = mean_absolute_error(y_test[0], test_predict[:, 0])
r2_test = r2_score(y_test[0], test_predict[:, 0])

print(f'Train RMSE: {rmse_train}, Train MAE: {mae_train}, Train R²: {r2_train}')
print(f'Test RMSE: {rmse_test}, Test MAE: {mae_test}, Test R²: {r2_test}')

# Plotting predictions
plt.figure(figsize=(14, 5))
plt.plot(data.index, data['Close'], label='Actual Price')
plt.plot(data.index[time_step:len(train_predict)+time_step], train_predict, label='Train Prediction')
plt.plot(data.index[len(train_predict)+(time_step*2)+1:len(data)-1], test_predict, label='Test Prediction')
plt.legend()
plt.show()

# Save the model
model.save('stock_price_model.h5')
