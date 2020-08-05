import utility.util as ut
import yfinance as yf

aapl = yf.Ticker('AAPL').history(period='10Y').reset_index()

aapl_train = aapl[0:2000]
aapl_test = aapl[2000:]

aapl_train.to_csv('aapl_train.csv', index=False)
aapl_test.to_csv('aapl_test.csv', index=False)