import pandas as pd
import numpy as np
import datetime as dt

class LastPriceModel:
    """
    This class represents the last price model which basically just predicts the closing price at day t to be
    the closing price at day t-1
    """

    def predict_one(self, prices_df):
        """
        Takes in a window of prices and returns the predicted price for the next day
        :param prices_df: The dataframe of the window of recent prices.
        :return: a value for the next day's price
        """

        return prices_df.tail(1)['Close']

    def predict_hist(selfs, prices_df):
        """
        Takes in a window of prices and returns the predicted price for each day in the given window
        :param prices_df: The dataframe of the window of recent prices
        :return: A dataframe with the historical predicted prices
        """

        chart_df = pd.DataFrame()
        chart_df['Date'] = prices_df['Date']
        chart_df['Close'] = prices_df['Close']
        chart_df = chart_df.append({
            'Date': prices_df['Date'].iloc[-1] + dt.timedelta(days=1),
            'Close': np.nan,
            'Predicted_Close': np.nan
        }, ignore_index=True)
        chart_df['Predicted_Close'] = chart_df['Close'].shift(1)

        return chart_df