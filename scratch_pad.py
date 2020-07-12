import yfinance as yf
import utility.util as ut
import matplotlib.pyplot as plt
import pandas as pd
import utility.config as cf

# ms = yf.Ticker('MSFT').get_info()

# ut.widen_df_display()
#
# # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# msft = yf.Ticker('MSFT').history(period='1Y')
# # swan = yf.Ticker('SWAN').history(period='1Y')
# # viot = yf.Ticker('VIOT').history(period='1Y')
#
# chart_df = pd.DataFrame()
# chart_df['MSFT'] = msft['Close']
# # chart_df['SWAN'] = swan['Close']
# # chart_df['VIOT'] = viot['Close']
# chart_df['Date'] = msft.index
#
# chart_df = chart_df.reset_index(drop=True)
#
# print(chart_df)
#
# chart_df.plot(x='Date')
#
# plt.show()

# msft = yf.Ticker('MSFT').history(period='1Y')

print(cf.is_app_debug())