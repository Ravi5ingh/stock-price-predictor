import utility.util as ut
import sklearn.preprocessing as pp
import numpy as np
import keras.models as km
import keras.layers as kl
import keras as kr
# import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ut.widen_df_display()

# Get data and add technical indicators
aapl_train = ut.read_csv('aapl_train.csv')

# Add technical indicators
aapl_train['SMA_10'] = aapl_train['Open'].shift(1).rolling(10).mean()
aapl_train['SMA_20'] = aapl_train['Open'].shift(1).rolling(20).mean()
aapl_train['SMA_50'] = aapl_train['Open'].shift(1).rolling(50).mean()
aapl_train['STDDEV_10'] = aapl_train['Open'].shift(1).rolling(10).std()
aapl_train['BOLL_MIN'] = aapl_train['Open'] - aapl_train['STDDEV_10']
aapl_train['BOLL_MAX'] = aapl_train['Open'] + aapl_train['STDDEV_10']
aapl_train = aapl_train.drop(['Date', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'STDDEV_10'], axis=1)

# Reshape data
training_set = aapl_train.values
scaler = pp.MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)

x_train = []
y_train = []
for i in range(50, 2000):
    x_train.append(training_set_scaled[i-50:i,])
    y_train.append(training_set_scaled[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

# Build the LSTM
regressor = km.Sequential()

regressor.add(kl.LSTM(units=40, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
regressor.add(kl.Dropout(0.2))


regressor.add(kl.LSTM(units=40, return_sequences=True))
regressor.add(kl.Dropout(0.2))

regressor.add(kl.LSTM(units=40, return_sequences=True))
regressor.add(kl.Dropout(0.2))

regressor.add(kl.LSTM(units=40))
regressor.add(kl.Dropout(0.2))

regressor.add(kl.Dense(units=1))

adam = kr.optimizers.Adam(lr=0.00005)

# Train the LSTM
regressor.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
regressor.fit(x_train, y_train, epochs=400, batch_size=64)

# Persist
regressor.save('saved_models/aapl-mv-model-epoch-400-bs-64-do-0.2-un-40-lr-0.00005')

# regressor = km.load_model('saved_models/model-epoch-150-bs-64-do-0.2-un-40-lr-0.00001')

# Make predictions on test set
aapl_test = ut.read_csv('aapl_test.csv')

# Add Technical Indicators to test data
aapl_test['SMA_10'] = aapl_test['Open'].shift(1).rolling(10).mean()
aapl_test['SMA_20'] = aapl_test['Open'].shift(1).rolling(20).mean()
aapl_test['SMA_50'] = aapl_test['Open'].shift(1).rolling(50).mean()
aapl_test['STDDEV_10'] = aapl_test['Open'].shift(1).rolling(10).std()
aapl_test['BOLL_MIN'] = aapl_test['Open'] - aapl_test['STDDEV_10']
aapl_test['BOLL_MAX'] = aapl_test['Open'] + aapl_test['STDDEV_10']
aapl_test = aapl_test.drop(['Date', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'STDDEV_10'], axis=1)

real_stock_prices = aapl_test.iloc[:,0].values

dataset_total = pd.concat((aapl_train, aapl_test), axis=0)
inputs = dataset_total[len(dataset_total) - len(aapl_test) - 50:].values

inputs = scaler.transform(inputs)
x_test = []
for i in range(50, 568):
    x_test.append(inputs[i-50:i,])

x_test = np.array(x_test)

print(x_test)
predicted_stock_price = regressor.predict(x_test)
print(predicted_stock_price)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(real_stock_prices, color='black', label='AAPL Actual')
plt.plot(predicted_stock_price, color='green', label='AAPL Predicted')
plt.title('AAPL Predicted vs. Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()