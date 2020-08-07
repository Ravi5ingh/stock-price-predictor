import utility.util as ut
import yfinance as yf
import utility.config as cf
import models.last_price as lp
import models.linear_regression as lr
import pandas as pd
import numpy as np
import datetime as dt
import models.lstm_uv as ls

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
    predicted_df, rmse = model.predict_hist(stock_data)

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

def get_linear_regression_predictions(symbol, period, n):
    """
    Get the price predictions (historical and for tomorrow) based on the linear regression model
    :param symbol: The stock symbol
    :param period: The period to use for historical price prediction
    :param n: The size of the moving window to use for the regression
    :return: The viz data for the historical price prediction
    """

    cutoff_date = cf.get_training_cutoff_date()
    stock_data = yf.Ticker(symbol).history(period=period).reset_index()

    # prune the stock data so that it is all after the cut-off date
    if stock_data['Date'][0] < np.datetime64(cutoff_date):
        stock_data = stock_data[stock_data['Close'] >= cutoff_date]

    model = lr.LinearRegressionModel(n)
    predicted_df, rmse = model.predict_hist(stock_data)

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
                title=f'Linear Regression Prediction Model for {symbol} with n = {n}'
            )
    )

def get_lstm_predictions(symbol):
    """

    :param symbol:
    :return:
    """

    train_df = pd.read_csv(f'../models/{symbol}/{symbol}_train.csv')
    test_df = pd.read_csv(f'../models/{symbol}/{symbol.lower()}_test.csv')

    chart_df = pd.read_csv(f'../models/{symbol}/chart_df.csv')

    return dict(
            data=[
                dict(
                    x=test_df['Date'],
                    y=chart_df['Actual'],
                    type='line',
                    name='Actual'
                ),
                dict(
                    x=test_df['Date'],
                    y=chart_df['Predicted'],
                    type='line',
                    name='Predicted'
                )
            ],
            layout=dict(
                title=f'LSTM model predictions for {symbol}'
            )
    )

