import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbprophet import Prophet

import Constants

dir = Constants.dataDir + "Quandl/"
# 1938
data = np.load(dir + "10YearData.npy")[1938]    # (numTickers, numDaysForTicker, 5)

dates = []
y = []
for day in data:
    dates.append(day[0])
    y.append(day[-1])

testStart = int(len(dates) * 0.8)

dfTrain = pd.DataFrame()
dfTrain['ds'] = dates[:testStart]
dfTrain['y'] = y[:testStart]

dfTest = pd.DataFrame()
dfTest['ds'] = dates[testStart:]
dfTest['y'] = y[testStart:]

days = (dfTest['ds'][len(dfTest) - 1] - dfTest['ds'][0]).days

model = Prophet()
model.fit(dfTrain)

future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
predcitions = forecast[['ds', 'yhat']]

plt.plot(dfTrain['ds'], dfTrain['y'], label='Training Data')
plt.plot(dfTest['ds'], dfTest['y'], label='Test Data')
plt.plot(predcitions['ds'], predcitions['yhat'], label='Prophet')
plt.legend()
plt.show()
