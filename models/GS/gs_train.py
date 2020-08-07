import yfinance as yf
import models.lstm_uv as ls
import pandas as pd

# Get and save the data
gs = yf.Ticker('GS').history(period='10Y').reset_index()
gs[0:2000].to_csv('gs_train.csv', index=False)
gs[2000:].to_csv('gs_test.csv', index=False)

gs_train = pd.read_csv('gs_train.csv')

model = ls.LongShortTermMemoryModel()

model.fit_transform(gs_train, 'lstm_model', 'scaler.pkl')

chart_df, rmse = model.predict(gs_train, pd.read_csv('gs_test.csv'), 'lstm_model', 'scaler.pkl')

chart_df.to_csv('chart_df.csv', index=False)