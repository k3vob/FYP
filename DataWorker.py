import datetime as dt

import pandas as pd
import pandas_market_calendars
import quandl

import Constants

# First Date:   1962-01-02 (IBM + 8 others)
# Num Tickers:  3194

dir = Constants.dataDir + "Quandl/"

quandl.ApiConfig.api_key = 'JFhNxibR4aonVzfd98XC'

endDate = dt.date(2017, 12, 31)
startDate = endDate - dt.timedelta(days=int(365.25 * Constants.years))
tradingDays = pandas_market_calendars.get_calendar('NYSE').valid_days(
    start_date=startDate, end_date=endDate)
tradingDays = [day.date() for day in tradingDays]
startDate = tradingDays[0]
endDate = tradingDays[-1]

df = quandl.get_table(
    'WIKI/PRICES',
    ticker='AAPL',
    date={'gte': startDate,
          'lte': endDate},
    paginate=True)

df.drop(['date', 'ticker', 'open', 'high', 'low', 'close', 'ex-dividend',
         'volume', 'split_ratio'], axis=1, inplace=True)

df['change'] = df['adj_close'].pct_change()

df['moving_avg'] = df['adj_close'].rolling(
    window=Constants.sequenceLength).mean()

df.dropna(inplace=True)

adj_close = df['adj_close']
df.drop(labels=['adj_close'], axis=1, inplace=True)
df = pd.concat([df, adj_close], axis=1)

numFeatures = df.shape[1] - 1

df = (df - df.min()) / (df.max() - df.min())

totalDays = int(df.shape[0])
trainingDays = int((totalDays // (1 / Constants.trainingPercentage) //
                    Constants.sequenceLength) * Constants.sequenceLength)
testingDays = int(totalDays - trainingDays)

x = df.iloc[:, :-1].as_matrix()
y = df.iloc[:, -1].as_matrix()
y = y.reshape(y.shape[0], 1)
