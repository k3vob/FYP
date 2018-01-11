import datetime as dt

import pandas as pd
import pandas_market_calendars
import quandl

import Constants

# First Date:   1962-01-02 (IBM + 8 others)
# Num Tickers:  3194

dir = Constants.dataDir + "Quandl/"

quandl.ApiConfig.api_key = 'JFhNxibR4aonVzfd98XC'

# Get trading days
endDate = dt.date(2017, 12, 31)
startDate = endDate - dt.timedelta(days=int(365.25 * Constants.years))
tradingDays = pandas_market_calendars.get_calendar('NYSE').valid_days(
    start_date=startDate, end_date=endDate)
tradingDays = [day.date() for day in tradingDays]
startDate = tradingDays[0]
endDate = tradingDays[-1]

df = quandl.get_table(
    'WIKI/PRICES',
    ticker='BAC',
    date={'gte': startDate,
          'lte': endDate},
    paginate=True)

# Change date to YYYY-MM-DD
df['date'] = [day.date() for day in df['date']]
df.set_index('date', inplace=True)

# Unneeded columns
df.drop(['ticker', 'open', 'high', 'low', 'close', 'ex-dividend',
         'volume', 'split_ratio'], axis=1, inplace=True)

# Percent change of closing price from one day to the next
df['change'] = df['adj_close'].pct_change()

# Moving average of sequence length window
df['moving_avg'] = df['adj_close'].rolling(
    window=Constants.sequenceLength).mean()

# Remove all rows with nulls
df.dropna(inplace=True)

# Move label to last column
adj_close = df['adj_close']
df.drop(labels=['adj_close'], axis=1, inplace=True)
df = pd.concat([df, adj_close], axis=1)

numFeatures = df.shape[1] - 1

# Normalise all to [0, 1]
df = (df - df.min()) / (df.max() - df.min())

# Counts for training, testing and total data
totalDays = int(df.shape[0])
trainingDays = int((totalDays // (1 / Constants.trainingPercentage) //
                    Constants.sequenceLength) * Constants.sequenceLength)
testingDays = int(totalDays - trainingDays)

# Numpy arrays of data
x = df.iloc[:, :-1].as_matrix()    # [totalDays, numFeatures]
y = df.iloc[:, -1].as_matrix()      # [totalDays]
y = y.reshape(y.shape[0], 1)        # [totdalDays, 1]
