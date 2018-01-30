import datetime as dt
import math
import pickle as pk
import random

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

################################################################################
# Get trading days
################################################################################

missingDay = dt.date(2017, 11, 8)   # MISSING FROM QUANDL
endDate = dt.date(2017, 12, 31)
startDate = endDate - dt.timedelta(days=int(365.25 * Constants.years))
tradingDays = pandas_market_calendars.get_calendar('NYSE').valid_days(
    start_date=startDate, end_date=endDate)
tradingDays = [day.date() for day in tradingDays]
tradingDays.remove(missingDay)
startDate = tradingDays[0]
endDate = tradingDays[-1]

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
# # Get SP500 Index adjusted closing price and change col name
# av = TimeSeries(key=av_api_key, output_format='pandas')
# sp500, _ = av.get_daily_adjusted(symbol='^GSPC', outputsize='full')
# sp500 = sp500['5. adjusted close'].to_frame()
# sp500.columns = ['sp500_adj_close']
#
# # Reduce to timeframe to trading days
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
# for ma in Constants.movingAverages:
#     sp500['sp500_{}_ma'.format(ma)] = sp500['sp500_adj_close'].rolling(ma).mean()
#
# # Remove all rows with nulls
# sp500.dropna(inplace=True)
#
# ################################################################################
# # Get daily prices for all S&P 500 companies
# ################################################################################
#
# startTickers = quandl.get_table(
#     'WIKI/PRICES',
#     date=startDate,
#     paginate=True
# )['ticker'].as_matrix()
#
# dfs = []        # List of DFs for all S&P 500 companies used
# tickers = []    # List of tickers for all S&P 500 companies used
#
# for i, t in enumerate(startTickers):
#     print(i, "/", len(startTickers))
#
#     # if not S&P 500 company, skip
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
#     # If any days missing, skip
#     if df.shape[0] != len(tradingDays):
#         continue
#
#     # Data will be stored for this ticker
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
#     # Construct moving averages of closing price
#     for ma in Constants.movingAverages:
#         df['{}_ma'.format(ma)] = df['adj_close'].rolling(ma).mean()
#
#     # Remove all rows with nulls
#     df.dropna(inplace=True)
#
#     # Concat sp500 data on to each ticker's data
#     df = pd.concat([df, sp500], axis=1)
#
#     dfs.append(df)
#
# # Concat DFs for all tickers into one
# # 2 layers of indices (1. ticker, 2. data)
# df = pd.concat(dfs, keys=tickers, names=['ticker'])
#
# # Normalise collectively
# normalisedDF = (df - df.min()) / (df.max() - df.min())
#
# # Daily percentage change of closing price and S&P 500 closing price
# df['sp500_pct_change'] = df['sp500_adj_close'].pct_change()
# df['pct_change'] = df['adj_close'].pct_change()
#
# # Copy un-normalised percentage changes into normalisedDF
# normalisedDF['sp500_pct_change'] = df['sp500_pct_change']
# normalisedDF['pct_change'] = df['pct_change']
#
# # Drop 1st row of each ticker (pct_change is from last ticker)
# idx = [df.xs(t, drop_level=False).index[0] for t in tickers]
# df = df.drop(idx)
# normalisedDF = normalisedDF.drop(idx)
#
# # Get max and min values for denormalisation
# minDF = df.min()
# maxDF = df.max()
#
# # 1 if True, 0 if False
# # Down -3%+
# normalisedDF['label_down3'] = (df['pct_change'] <= -Constants.returnTarget).astype(float)
# # Down 0% -> -3%
# normalisedDF['label_down'] = ((df['pct_change'] < 0) &
#                               (df['pct_change'] > -Constants.returnTarget)).astype(float)
# # Up 0% -> 3%
# normalisedDF['label_up'] = ((df['pct_change'] >= 0) &
#                             (df['pct_change'] < Constants.returnTarget)).astype(float)
# # Up 3%+
# normalisedDF['label_up3'] = (df['pct_change'] >= Constants.returnTarget).astype(float)
#
# # Store as label as one-hot list
# # normalisedDF['label'] = (df[['down3', 'down', 'up', 'up3']].values[:, :, None]).tolist()
#
# pk.dump(normalisedDF, open(dir + "5YearData.p", "wb"))
# pk.dump(minDF, open(dir + "min5YearData.p", "wb"))
# pk.dump(maxDF, open(dir + "max5YearData.p", "wb"))
# pk.dump(tickers, open(dir + "5YearTickers.p", "wb"))
#
# print("Stored {} year data for {} companies".format(Constants.years, len(tickers)))

################################################################################
# Load data & create batches
################################################################################

df = pk.load(open(dir + "5YearData.p", "rb"))
minDF = pk.load(open(dir + "min5YearData.p", "rb"))
maxDF = pk.load(open(dir + "max5YearData.p", "rb"))
tickers = pk.load(open(dir + "5YearTickers.p", "rb"))
testingTicker = random.choice(tickers)

numDays = df.loc[tickers[0]].shape[0]
numTrainingDays = int(numDays * Constants.trainingPercentage)
numFeatures = df.shape[1] - Constants.numLabels
numSlices = math.ceil(len(tickers) / Constants.batchSize)


def getBatch(dayPointer, tickerPointer=0, isTesting=True):
    batchX, batchY = [], []
    batchSize = 1
    if not isTesting:
        batchSize = min(
            Constants.batchSize,
            len(tickers) - tickerPointer - 1
        )   # - 1 -> Last ticker for testing

    for i in range(batchSize):
        ticker = tickers[tickerPointer + i]
        if not isTesting and ticker == testingTicker:
            continue
        if isTesting:
            ticker = testingTicker

        x = df.loc[ticker].iloc[
            dayPointer:dayPointer + Constants.sequenceLength,
            :-Constants.numLabels
        ].as_matrix()

        y = df.loc[ticker].iloc[
            dayPointer + 1: dayPointer + Constants.sequenceLength + 1,
            -Constants.numLabels:
        ].as_matrix()

        # y = y.reshape(y.shape[0], 4)
        batchX.append(x)
        batchY.append(y)

    if not isTesting:
        dayPointer += Constants.sequenceLength
        if dayPointer + Constants.sequenceLength + 1 >= numTrainingDays:
            dayPointer = 0      # + 1 -> y is shifted by 1
    else:
        dayPointer += 1
        if dayPointer + Constants.sequenceLength + 1 >= numDays:
            dayPointer = 0      # + 1 -> y is shifted by 1

    if dayPointer == 0:
        if batchSize < Constants.batchSize:
            tickerPointer = 0
        else:
            tickerPointer += Constants.batchSize

    return np.array(batchX), np.array(batchY), tickerPointer, dayPointer
