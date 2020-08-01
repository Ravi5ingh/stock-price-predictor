import yfinance as yf
import utility.util as ut
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utility.config as cf
import models.last_price as lp
import datetime as dt
import website.visualizer as vz

# ms = yf.Ticker('MSFT').get_info()

ut.widen_df_display()
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
# print(chart_df.tail())

ss = vz.get_last_day_predictions('MSFT', period='1mo')

print(ss)