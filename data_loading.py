import utility.util as ut
import yfinance as yf

ftse = yf.Ticker('^FTSE').history(period='10Y').reset_index()

ftse_train = ftse[0:2000]
ftse_test = ftse[2000:]

ftse_train.to_csv('ftse_train.csv', index=False)
ftse_test.to_csv('ftse_test.csv', index=False)