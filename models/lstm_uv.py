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

        training_set = train_df.iloc[:, 1:2].values

        scaler = pp.MinMaxScaler(feature_range=(0, 1))

        training_set_scaled = scaler.fit_transform(training_set)

        ut.to_pkl(scaler, out_scaler_filename)

        x_train = []
        y_train = []
        for i in range(50, 2000):
            x_train.append(training_set_scaled[i - 50:i, 0])
            y_train.append(training_set_scaled[i, 0])

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

        regressor = km.load_model(model_filename)
        scaler = ut.read_pkl(scaler_filename)

        # Make predictions on test set
        real_stock_prices = test_df.iloc[:, 1:2].values

        dataset_total = pd.concat((train_df['Open'], test_df['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(test_df) - 50:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        x_test = []
        for i in range(50, 568):
            x_test.append(inputs[i - 50:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        print(x_test)
        predicted_stock_price = regressor.predict(x_test)
        print(predicted_stock_price)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        predicted_stock_price = np.array(predicted_stock_price).flatten()

        chart_df = pd.DataFrame()
        chart_df['Predicted'] = predicted_stock_price
        chart_df['Actual'] = real_stock_prices

        mse = mt.mean_squared_error(chart_df['Actual'], chart_df['Predicted'])
        rmse = ma.sqrt(mse)

        return chart_df, rmse