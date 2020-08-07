import yfinance as yf
import models.lstm_uv as ls
import pandas as pd

# Get and save the data
mcd = yf.Ticker('MCD').history(period='10Y').reset_index()
mcd[0:2000].to_csv('mcd_train.csv', index=False)
mcd[2000:].to_csv('mcd_test.csv', index=False)

mcd_train = pd.read_csv('mcd_train.csv')

model = ls.LongShortTermMemoryModel()

model.fit_transform(mcd_train, 'lstm_model', 'scaler.pkl')

chart_df, rmse = model.predict(mcd_train, pd.read_csv('mcd_test.csv'), 'lstm_model', 'scaler.pkl')

chart_df.to_csv('chart_df.csv', index=False)