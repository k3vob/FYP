import datetime as dt
import math
import pickle as pk

import numpy as np
import pandas as pd
import pandas_market_calendars
import quandl

import Constants

dir = Constants.dataDir + "Quandl/"

quandl.ApiConfig.api_key = 'JFhNxibR4aonVzfd98XC'

# # Get trading days
# missingDay = dt.date(2017, 11, 8)   # MISSING FROM QUANDL
# endDate = dt.date(2017, 12, 31)
# startDate = endDate - dt.timedelta(days=int(365.25 * Constants.years))
# tradingDays = pandas_market_calendars.get_calendar('NYSE').valid_days(
#     start_date=startDate, end_date=endDate)
# tradingDays = [day.date() for day in tradingDays]
# tradingDays.remove(missingDay)
# startDate = tradingDays[0]
# endDate = tradingDays[-1]
#
# startTickers = quandl.get_table(
#     'WIKI/PRICES',
#     date=startDate,
#     paginate=True
# )['ticker'].as_matrix()
#
# dfs = []
# tickers = []
#
# for i, t in enumerate(startTickers):
#     print(i, "/", len(startTickers))
#     df = quandl.get_table(
#         'WIKI/PRICES',
#         ticker=t,
#         date={'gte': startDate,
#               'lte': endDate},
#         paginate=True
#     )
#
#     if df.shape[0] != len(tradingDays):
#         continue
#
#     tickers.append(t)
#
#     # Change date to YYYY-MM-DD
#     df['date'] = [day.date() for day in df['date']]
#     df.set_index('date', inplace=True)
#
#     # Unneeded columns
#     df.drop(['ticker', 'open', 'high', 'low', 'close', 'ex-dividend',
#              'volume', 'split_ratio'], axis=1, inplace=True)
#
#     # Percent change of closing price from one day to the next
#     df['change'] = df['adj_close'].pct_change()
#
#     # Moving average of sequence length window
#     df['moving_avg'] = df['adj_close'].rolling(
#         window=Constants.sequenceLength).mean()
#
#     # Remove all rows with nulls
#     df.dropna(inplace=True)
#
#     # Move label to last column
#     adj_close = df['adj_close']
#     df.drop(labels=['adj_close'], axis=1, inplace=True)
#     df = pd.concat([df, adj_close], axis=1)
#
#     # ### Normalise individually
#     df = (df - df.min()) / (df.max() - df.min())
#
#     dfs.append(df)
#
# df = pd.concat(dfs, keys=tickers, names=['ticker'])
#
# # ### Normalise collectively
# # df = (df - df.min()) / (df.max() - df.min())
#
# pk.dump(df, open(dir + "5YearDF_NormIndiv.p", "wb"))
# pk.dump(tickers, open(dir + "5YearTickers.p", "wb"))

df = pk.load(open(dir + "5YearDF_NormIndiv.p", "rb"))
tickers = pk.load(open(dir + "5YearTickers.p", "rb"))

numDays = df.loc[tickers[0]].shape[0]
numFeatures = df.shape[1]
numTickerGroups = math.ceil(len(tickers) / Constants.batchSize)


def getBatch(tickerPointer, dayPointer, isTraining=True):
    batchX, batchY = [], []
    batchSize = 1
    if isTraining:
        batchSize = min(
            Constants.batchSize,
            len(tickers) - tickerPointer - 1
        )   # - 1 -> Last ticker for testing
    for i in range(batchSize):
        ticker = tickers[tickerPointer + i]
        x = df.loc[ticker].iloc[dayPointer:dayPointer + Constants.sequenceLength].as_matrix()
        y = df.loc[ticker].iloc[dayPointer + 1: dayPointer +
                                Constants.sequenceLength + 1, -1].as_matrix()
        y = y.reshape(y.shape[0], 1)
        batchX.append(x)
        batchY.append(y)

    if isTraining:
        dayPointer += Constants.sequenceLength
    else:
        dayPointer += 1

    if dayPointer + Constants.sequenceLength + 1 >= numDays:
        dayPointer = 0      # + 1 -> y is shifted by 1

    if dayPointer == 0:
        if batchSize < Constants.batchSize:
            tickerPointer = 0
        else:
            tickerPointer += Constants.batchSize

    return batchX, batchY, tickerPointer, dayPointer
