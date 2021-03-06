import utility.util as ut
import sklearn.preprocessing as pp
import numpy as np
import keras.models as km
import keras.layers as kl
import keras as kr
import pandas as pd
import matplotlib.pyplot as plt

# Get and reshape data
msft_train = ut.read_csv('msft_train.csv')

training_set = msft_train.iloc[:, 1:2].values

scaler = pp.MinMaxScaler(feature_range=(0, 1))

training_set_scaled = scaler.fit_transform(training_set)

x_train = []
y_train = []
for i in range(50, 2000):
    x_train.append(training_set_scaled[i-50:i,0])
    y_train.append(training_set_scaled[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM
regressor = km.Sequential()

regressor.add(kl.LSTM(units=40, return_sequences=True, input_shape=(x_train.shape[1], 1)))
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
regressor.save('saved_models/model-epoch-600-bs-64-do-0.2-un-40-lr-0.00005')

# regressor = km.load_model('saved_models/model-epoch-400-bs-64-do-0.2-un-40-lr-0.00005')

# Make predictions on test set
msft_test = ut.read_csv('msft_test.csv')
real_stock_prices = msft_test.iloc[:,1:2].values

dataset_total = pd.concat((msft_train['High'], msft_test['High']), axis=0)
inputs = dataset_total[len(dataset_total) - len(msft_test) - 50:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
x_test = []
for i in range(50, 568):
    x_test.append(inputs[i-50:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(real_stock_prices, color='black', label='MSFT Actual')
plt.plot(predicted_stock_price, color='green', label='MSFT Predicted')
plt.title('MSFT Predicted vs. Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

#region Old

#
# # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# msft = yf.Ticker('MSFT').history(period='1Y')
# # swan = yf.Ticker('SWAN').history(period='1Y')
# # viot = yf.Ticker('VIOT').history(period='1Y')

# msft = yf.Ticker('MSFT').history(period='1mo').reset_index()
# msft.to_csv('msft.csv', index=False)
#
# chart_df = pd.DataFrame()
# chart_df['Date'] = msft['Date']
# chart_df['Close'] = msft['Close']
# chart_df = chart_df.append({
#     'Date': msft['Date'].iloc[-1] + dt.timedelta(days=1),
#     'Close': np.nan,
#     'Predicted_Close': np.nan
# }, ignore_index=True)
# chart_df['Predicted_Close'] = chart_df['Close'].shift(1)
#
# print(chart_df.tail()

#endregion