import pandas as pd
import sklearn.preprocessing as pp
import utility.util as ut
import numpy as np
import keras as kr
import keras.layers as kl
import keras.models as km
import sklearn.metrics as mt
import math as ma

class LongShortTermMemoryModel:

    def fit_transform(self, train_df, out_model_filename, out_scaler_filename):
        """
        Take in training data and fit the model to it, then persist the model and scaler
        :param train_df: The training data
        :param out_model_filename: The file name to save the trained model to
        :param out_scaler_filename: The file name to save the fitted scaler to
        """

        # Add technical indicators
        train_df['SMA_10'] = train_df['Open'].shift(1).rolling(10).mean()
        train_df['SMA_20'] = train_df['Open'].shift(1).rolling(20).mean()
        train_df['SMA_50'] = train_df['Open'].shift(1).rolling(50).mean()
        train_df['STDDEV_10'] = train_df['Open'].shift(1).rolling(10).std()
        train_df['BOLL_MIN'] = train_df['Open'] - train_df['STDDEV_10']
        train_df['BOLL_MAX'] = train_df['Open'] + train_df['STDDEV_10']
        train_df = train_df.drop(['Date', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'STDDEV_10'],
                                   axis=1)

        # Normalize
        training_set = train_df.values
        scaler = pp.MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = scaler.fit_transform(training_set)

        # Save the scaler for inference
        ut.to_pkl(scaler, out_scaler_filename)

        # Stagger the time series into batches
        x_train = []
        y_train = []
        for i in range(50, 2000):
            x_train.append(training_set_scaled[i - 50:i, ])
            y_train.append(training_set_scaled[i, 0])

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
        regressor.save(out_model_filename)

    def predict(self, train_df, test_df, model_filename, scaler_filename):
        """
        Makes predictions based on the test df
        :param train_df: The train data (so we can look back 50 iterations before the test set)
        :param test_df: The test data to predict with
        :param model_filename: The name of the file of the saved model
        :param scaler_filename: The name of the file of the saved scaler
        :return: a chart and RMSE
        """

        # Add Technical Indicators to test data
        test_df['SMA_10'] = test_df['Open'].shift(1).rolling(10).mean()
        test_df['SMA_20'] = test_df['Open'].shift(1).rolling(20).mean()
        test_df['SMA_50'] = test_df['Open'].shift(1).rolling(50).mean()
        test_df['STDDEV_10'] = test_df['Open'].shift(1).rolling(10).std()
        test_df['BOLL_MIN'] = test_df['Open'] - test_df['STDDEV_10']
        test_df['BOLL_MAX'] = test_df['Open'] + test_df['STDDEV_10']
        aapl_test = test_df.drop(['Date', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'STDDEV_10'],
                                   axis=1)

        real_stock_prices = aapl_test.iloc[:, 0].values

        dataset_total = pd.concat((train_df, aapl_test), axis=0)
        inputs = dataset_total[len(dataset_total) - len(aapl_test) - 50:].values

        scaler = ut.read_pkl(scaler_filename)

        inputs = scaler.transform(inputs)
        x_test = []
        for i in range(50, 568):
            x_test.append(inputs[i - 50:i, ])

        x_test = np.array(x_test)

        regressor = ut.read_pkl(model_filename)

        predicted_stock_price = regressor.predict(x_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        chart_df = pd.DataFrame()
        chart_df['Actual'] = real_stock_prices
        chart_df['Predicted'] = predicted_stock_price

        mse = mt.mean_squared_error(chart_df['Actual'], chart_df['Predicted'])
        rmse = ma.sqrt(mse)

        return chart_df, rmse

        # plt.plot(real_stock_prices, color='black', label='AAPL Actual')
        # plt.plot(predicted_stock_price, color='green', label='AAPL Predicted')
        # plt.title('AAPL Predicted vs. Actual')
        # plt.xlabel('Time')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.show()