import yfinance as yf
import models.lstm_uv as ls
import pandas as pd

# Get and save the data
dis = yf.Ticker('DIS').history(period='10Y').reset_index()
dis[0:2000].to_csv('dis_train.csv', index=False)
dis[2000:].to_csv('dis_test.csv', index=False)

ibm_train = pd.read_csv('dis_train.csv')

model = ls.LongShortTermMemoryModel()

model.fit_transform(ibm_train, 'lstm_model', 'scaler.pkl')