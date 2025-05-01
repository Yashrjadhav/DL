import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load & scale training data
train = pd.read_csv('/content/Google_Stock_Price_Train.csv')
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train.iloc[:, 1:2].values)

X_train, y_train = [], []
for i in range(60, len(scaled_train)):
    X_train.append(scaled_train[i-60:i, 0])
    y_train.append(scaled_train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(-1, 60, 1)

# Plot stock prices
plt.plot(train['Open'], label='Open')
plt.title("Google Stock Prices")
plt.xlabel("Time"), plt.ylabel("Price")
plt.legend(), plt.show()

plt.plot(train['Low'], label='Low')
plt.title("Google Stock Prices")
plt.xlabel("Time"), plt.ylabel("Price")
plt.legend(), plt.show()

# Build & train model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)), Dropout(0.2),
    LSTM(50, return_sequences=True), Dropout(0.2),
    LSTM(50, return_sequences=True), Dropout(0.2),
    LSTM(50), Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Prepare test data
test = pd.read_csv('/content/Google_Stock_Price_Train.csv')
real_price = test.iloc[:, 1:2].values
total = pd.concat((train['Open'], test['Open']), axis=0)
inputs = scaler.transform(total[len(total)-len(test)-60:].values.reshape(-1, 1))

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test).reshape(-1, 60, 1)

# Predict & plot
predicted = scaler.inverse_transform(model.predict(X_test))
plt.plot(real_price, color='red', label='Real Price')
plt.plot(predicted, color='blue', label='Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time'), plt.ylabel('Price')
plt.legend(), plt.show()
