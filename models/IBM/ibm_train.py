import yfinance as yf
import models.lstm as ls
import pandas as pd

# Get and save the data
# ibm = yf.Ticker('IBM').history(period='10Y').reset_index()
# ibm[0:2000].to_csv('ibm_train.csv', index=False)
# ibm[2000:].to_csv('ibm_test.csv', index=False)

ibm_train = pd.read_csv('ibm_train.csv')

model = ls.LongShortTermMemoryModel()

model.fit_transform(ibm_train, 'lstm_model', 'scaler.pkl')