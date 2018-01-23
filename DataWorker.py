import datetime as dt
import math
import pickle as pk

import bs4 as bs
import numpy as np
import pandas as pd
import pandas_market_calendars
import quandl
import requests as rq
from alpha_vantage.timeseries import TimeSeries

import Constants

av_api_key = 'XIZOWSOCZRYV23XJ'
quandl.ApiConfig.api_key = 'JFhNxibR4aonVzfd98XC'

dir = Constants.dataDir + "Quandl/"

# ################################################################################
# # Get trading days
# ################################################################################
#
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
# ################################################################################
# # Get S&P 500 tickers
# ################################################################################
#
# html = rq.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# soup = bs.BeautifulSoup(html.text, 'lxml')
# table = soup.find('table', {'class': 'wikitable sortable'})
# sp500_tickers = []
# for row in table.findAll('tr')[1:]:
#     ticker = row.findAll('td')[0].text
#     sp500_tickers.append(ticker)
#
# ################################################################################
# # Get S&P 500 index prices
# ################################################################################
#
# av = TimeSeries(key=av_api_key, output_format='pandas')
# sp500, _ = av.get_daily_adjusted(symbol='^GSPC', outputsize='full')
# sp500 = sp500['5. adjusted close'].to_frame()
# sp500.columns = ['sp500_adj_close']
#
# # Reduce to trading days
# sp500 = sp500.loc[str(startDate):str(endDate)]
# sp500.drop([str(missingDay)], inplace=True)
#
# # Convert string index to date
# dateStrings = sp500.index.values
# dates = []
# for i in range(len(tradingDays)):
#     parts = dateStrings[i].split('-')
#     date = dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
#     dates.append(date)
#
# sp500.set_index([dates], inplace=True)
#
# # Percent change of closing price from one day to the next
# sp500['sp500_change'] = sp500['sp500_adj_close'].pct_change()
#
# # Moving average of sequence length window
# sp500['sp500_moving_avg'] = sp500['sp500_adj_close'].rolling(
#     window=Constants.sequenceLength).mean()
#
# # Remove all rows with nulls
# sp500.dropna(inplace=True)
#
# # Move label to last column
# adj_close = sp500['sp500_adj_close']
# sp500.drop(labels=['sp500_adj_close'], axis=1, inplace=True)
# sp500 = pd.concat([sp500, adj_close], axis=1)
#
# # Normalise tp [0, 1]
# sp500 = (sp500 - sp500.min()) / (sp500.max() - sp500.min())
#
# ################################################################################
# # Get all tickers
# ################################################################################
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
#
#     if t not in sp500_tickers:
#         continue
#
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
#     # ### Normalise individually
#     df = (df - df.min()) / (df.max() - df.min())
#
#     # Concat sp500
#     df = pd.concat([df, sp500], axis=1)
#
#     # Move label to last column
#     adj_close = df['adj_close']
#     df.drop(labels=['adj_close'], axis=1, inplace=True)
#     df = pd.concat([df, adj_close], axis=1)
#
#     dfs.append(df)
#
# df = pd.concat(dfs, keys=tickers, names=['ticker'])
#
# # ### Normalise collectively
# # df = (df - df.min()) / (df.max() - df.min())
#
# pk.dump(df, open(dir + "5YearDF.p", "wb"))
# pk.dump(tickers, open(dir + "5YearTickers.p", "wb"))

################################################################################
# Load data & create batches
################################################################################

df = pk.load(open(dir + "5YearDF.p", "rb"))
tickers = pk.load(open(dir + "5YearTickers.p", "rb"))

numDays = df.loc[tickers[0]].shape[0]
numFeatures = df.shape[1] - 1
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
        x = df.loc[ticker].iloc[dayPointer:dayPointer + Constants.sequenceLength, :-1].as_matrix()
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
