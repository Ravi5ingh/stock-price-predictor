import yfinance as yf
import utility.util as ut

msft = yf.Ticker('MSFT').history(period='5Y').reset_index()
goog = yf.Ticker('GOOG').history(period='5Y').reset_index()
tsla = yf.Ticker('TSLA').history(period='5Y').reset_index()

msft.to_csv('msft.csv', index=False)
goog.to_csv('goog.csv', index=False)
tsla.to_csv('tsla.csv', index=False)