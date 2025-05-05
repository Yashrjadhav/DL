import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Load and preprocess dataset
df = pd.read_csv('/content/Google_Stock_Price_Train.csv')
df['Close'] = df['Close'].str.replace(',', '').astype(float)
data = df['Close'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_dataset(data, step=60):
    X = np.array([data[i-step:i, 0] for i in range(step, len(data))])
    y = data[step:, 0]
    return X, y

X, y = create_dataset(scaled_data)
X = X[..., np.newaxis]  # Add third dimension for RNN input

# Build RNN model
model = Sequential([
    SimpleRNN(50, input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X, y, epochs=5, batch_size=32)

# Predict
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Plot
plt.plot(real_prices, label='Real')
plt.plot(predicted_prices, label='Predicted')
plt.legend()
plt.show()
