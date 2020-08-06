import models.linear_regression as lr
import models.last_price as lp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utility.util as ut

def plot_rmse_vs_n_for_linreg_model(test_df, stock_symbol, max_n):
    """
    For the linear regression model, plots the RMSE vs. n p
    :param test_df:
    :param stock_symbol:
    :param max_n:
    :return:
    """

    plot_df = pd.DataFrame()
    plot_df['n'] = np.arange(2, max_n + 1)
    plot_df['RMSE'] = np.zeros(max_n - 1)

    for n in plot_df['n']:

        model = lr.LinearRegressionModel(n)
        chart_df, rmse = model.predict_hist(test_df)
        plot_df.at[n - 2, 'RMSE'] = rmse
        ut.update_progress(n, max_n)

    plot = plot_df.plot(x='n')
    plot.set_title(f'RMSE vs. n for linear regression {stock_symbol}')

    plt.show()

def plot_lin_reg_mode(prices_df, stock_symbol, n):
    """
    Plots the predicted vs. actual for the linear regression model on the given stock prices
    :param prices_df: The stock prices to predict
    :param stock_symbol: The stock symbol
    :param n: The hyper-parameter
    """

    model = lr.LinearRegressionModel(n)

    chart_df, rmse = model.predict_hist(prices_df)

    plt.plot(chart_df['Close'], color='black', label=f'{stock_symbol} Actual')
    plt.plot(chart_df['Predicted_Close'], color='green', label=f'{stock_symbol} Predicted')
    plt.title(f'{stock_symbol} Predicted vs. Actual for Last Price Model')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_last_price_model(prices_df, stock_symbol):
    """
    Plots the predicted vs. actual for the last price model on the given stock prices
    :param prices_df: The stock prices to predict
    :param stock_symbol: The stock symbol
    """

    model = lp.LastPriceModel()

    chart_df, rmse = model.predict_hist(prices_df)

    plt.plot(chart_df['Close'], color='black', label=f'{stock_symbol} Actual')
    plt.plot(chart_df['Predicted_Close'], color='green', label=f'{stock_symbol} Predicted')
    plt.title(f'{stock_symbol} Predicted vs. Actual for Last Price Model')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()