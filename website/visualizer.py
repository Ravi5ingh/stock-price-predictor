import utility.util as ut
import yfinance as yf
import utility.config as cf
import models.last_price as lp
import pandas as pd
import numpy as np
import datetime as dt

def get_stock_ts(symbol, period):
    """
    Get the stock data for the given period
    :param symbol: The stock symbol
    :param period: The time period (eg. '1Y') in accordance with Yahoo Finance
    :return: The viz data for the stock chart
    """
    stock_data = yf.Ticker(symbol).history(period=period)

    return dict(
            data=[
                dict(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    type='line'
                )
            ],
            layout=dict(
                title=f'Daily closing price for {symbol}'
            )
    )

def get_last_day_predictions(symbol, period):
    """
    Get price predictions (historical and for tomorrow) based on the last price model
    :param symbol: The stock symbol
    :param period: The period to use for historical price prediction
    :return: The viz data for the historical price prediction
    """

    cutoff_date = cf.get_training_cutoff_date()
    stock_data = yf.Ticker(symbol).history(period=period).reset_index()

    # prune the stock data so that it is all after the cut-off date
    if stock_data['Date'][0] < np.datetime64(cutoff_date):
        stock_data = stock_data[stock_data['Close'] >= cutoff_date]

    model = lp.LastPriceModel()
    predicted_df = model.predict_hist(stock_data)

    return dict(
            data=[
                dict(
                    x=predicted_df['Date'],
                    y=predicted_df['Close'],
                    type='line',
                    name='Actual'
                ),
                dict(
                    x=predicted_df['Date'],
                    y=predicted_df['Predicted_Close'],
                    type='line',
                    name='Predicted'
                )
            ],
            layout=dict(
                title=f'Last Price Prediction Model for {symbol}'
            )
    )

