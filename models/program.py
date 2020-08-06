import pandas as pd
import models.investigations as iv
import yfinance as yf

# # Last price chart
# msft = yf.Ticker('MSFT').history(period='1Y').reset_index().tail(100)
#
# iv.plot_last_price_model(msft, 'MSFT')

# # Lin Reg Chart
# msft = yf.Ticker('MSFT').history(period='1Y').reset_index().tail(100)
#
# iv.plot_lin_reg_mode(msft, 'MSFT', n=5)

# # Plot RMSE vs. n
# msft = yf.Ticker('MSFT').history(period='1Y').reset_index().tail(100)
#
# iv.plot_rmse_vs_n_for_linreg_model(msft, 50)

# Plot RMSE vs. n for different stocks
aapl = yf.Ticker('AAPL').history(period='1Y').reset_index().tail(100)
goog = yf.Ticker('GOOG').history(period='1Y').reset_index().tail(100)
tsla = yf.Ticker('TSLA').history(period='1Y').reset_index().tail(100)
ftse = yf.Ticker('^FTSE').history(period='1Y').reset_index().tail(100)

iv.plot_rmse_vs_n_for_linreg_model(aapl, 'AAPL', 10)
iv.plot_rmse_vs_n_for_linreg_model(goog, 'GOOG', 10)
iv.plot_rmse_vs_n_for_linreg_model(tsla, 'TLSA',10)
iv.plot_rmse_vs_n_for_linreg_model(ftse, 'FTSE 100',10)