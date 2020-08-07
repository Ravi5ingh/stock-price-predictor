import yfinance as yf
import models.lstm_uv as ls
import pandas as pd

# Get and save the data
qcom = yf.Ticker('qcom').history(period='10Y').reset_index()
qcom[0:2000].to_csv('qcom_train.csv', index=False)
qcom[2000:].to_csv('qcom_test.csv', index=False)

qcom_train = pd.read_csv('qcom_train.csv')

model = ls.LongShortTermMemoryModel()

model.fit_transform(qcom_train, 'lstm_model', 'scaler.pkl')

chart_df, rmse = model.predict(qcom_train, pd.read_csv('qcom_test.csv'), 'lstm_model', 'scaler.pkl')

chart_df.to_csv('chart_df.csv', index=False)