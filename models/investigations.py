import models.linear_regression as lr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_rmse_vs_n_for_linreg_model(test_df, max_n):
    """
    For the linear regression model, plots the RMSE vs. n p
    :param test_df:
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

    plot = plot_df.plot(x='n')
    plot.set_title('RMSE vs. n for linear regression (MSFT 2mo)')

    plt.show()