import numpy as np
import pandas as pd
import datetime as dt
import sklearn.metrics as me
import math as ma

class LinearRegressionModel:
    """
    This model uses linear regression to make predictions for the next day's stock price
    """

    def __init__(self, n):
        """
        .ctor
        The train dataframe must have columns 'Date' and 'Close' at least
        :param n: At inference time, perform linear regression over the 'n' most recent data points
        """

        self.__n__ = n

    def predict_one(self, prices_df):
        """
        Takes in a window of prices and returns the predicted price for the next day
        :param prices_df: The dataframe of the window of recent prices.
        :return: a value for the next day's price
        """

        n = self.__n__

        prices_df = prices_df.tail(n)

        X = np.arange(n)
        Y = prices_df['Close']
        m, b = np.polyfit(X, Y, deg=1)

        return m * n + b

    def predict_hist(self, prices_df):
        """
        Takes in a window of prices and returns the predicted price for each day in the given window
        :param prices_df: The dataframe of the window of recent prices
        :return: A dataframe with the historical predicted prices
        """

        n = self.__n__

        chart_df = pd.DataFrame()
        chart_df['Date'] = prices_df['Date']
        chart_df['Close'] = prices_df['Close']
        chart_df = chart_df.append({
            'Date': prices_df['Date'].iloc[-1] + dt.timedelta(days=1),
            'Close': np.nan,
            'Predicted_Close': np.nan
        }, ignore_index=True)

        for index, row in chart_df.iterrows():

            if index <= 1:
                continue

            start = max(0, index - n)
            X = np.arange(index - start)
            Y = chart_df[start:index]['Close']
            m, b = np.polyfit(X, Y, deg=1)

            chart_df.at[index, 'Predicted_Close'] = m * n + b

        mse = me.mean_squared_error(chart_df['Close'][2:-1], chart_df['Predicted_Close'][2:-1])
        rmse = ma.sqrt(mse)

        return chart_df, rmse
