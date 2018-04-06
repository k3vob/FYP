import datetime as dt
import math
import pickle as pk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

import Constants
from Model import LSTM

ticker = 'GOOGL'
barchartAPIKey = '149144f23ae0f04a08f73bd8ec4c639d'
barchartURL = 'http://marketdata.websol.barchart.com/getHistory.csv?' + \
    'key={}&symbol={}&type={}&startDate={}'


def getStartDate():
    today = dt.datetime.now().date() + dt.timedelta(days=1)
    day = today.day
    year = today.year
    month = (today.month - 3) % 12
    if day == 1:
        month = today.month + 1
    if month >= 10:
        year -= 1
    return dt.date(year, month, day)


def getMarketData(startDate):
    # Create YYYYMMDD string for the URL
    yearString = str(startDate.year)
    monthString = str(startDate.month).zfill(2)
    dayString = str(startDate.day).zfill(2)
    startString = yearString + monthString + dayString
    url = barchartURL.format(barchartAPIKey, ticker, 'minutes', startString)
    # Download as DF and set timestamps in UTC
    data = pd.read_csv(url, parse_dates=['timestamp'])
    # Convert numpy.datetime64 to datetime objects
    datetimes = [pd.Timestamp(np_datetime).to_pydatetime() for np_datetime in data['timestamp']]
    data['timestamp'] = datetimes
    # Set date and time as two levels of indexes
    data.set_index([data['timestamp'].dt.date, data['timestamp'].dt.time], inplace=True)
    data.rename_axis(['date', 'time'], inplace=True)
    volume = data['volume']
    data.drop(['timestamp', 'symbol', 'tradingDay', 'volume'], axis=1, inplace=True)
    data.insert(0, 'volume', volume)
    return data


# ##############################################################################
# COLLECT DATA
# ##############################################################################

# startDate = getStartDate()
# marketData = getMarketData(startDate)
# pk.dump(marketData, open(Constants.dataDir + "Intraday/marketData.p", "wb"))

##############################################################################
# PRE-PROCESS DATA
##############################################################################

# startDate = getStartDate()
# marketData = pk.load(open(Constants.dataDir + "Intraday/marketData.p", "rb"))
#
# marketData['volume'] = (marketData['volume'] - marketData['volume'].mean()) / marketData['volume'].std()
# marketData['open'] = marketData['open'].pct_change()
# marketData['high'] = marketData['high'].pct_change()
# marketData['low'] = marketData['low'].pct_change()
# marketData['close'] = marketData['close'].pct_change()
#
# sentimentData = pd.read_csv(open(Constants.dataDir + "Knowsis/GOOGL.csv", "rb"))
# sentimentData = sentimentData.iloc[:, -5:]
#
# sentimentData['sentiment'] = (sentimentData['positive'] - sentimentData['negative']) / sentimentData['total']
#
# sentimentData['datetime'] = [dt.datetime.strptime(datetimeString, '%Y-%m-%d %H:%M:%S')
#                              for datetimeString in sentimentData['datetime']]
#
# sentimentData.set_index([sentimentData['datetime'].dt.date, sentimentData['datetime'].dt.time], inplace=True)
# sentimentData.rename_axis(['date', 'time'], inplace=True)
# # sentimentData.drop(['datetime'], axis=1, inplace=True)
# sentimentData.drop(['datetime', 'positive', 'negative', 'neutral'], axis=1, inplace=True)
#
# rowsToDrop = []
# for i in range(sentimentData.shape[0]):
#     if sentimentData.index.values[i][0].date() < startDate:
#         rowsToDrop.append(i)
# sentimentData.drop(sentimentData.index[rowsToDrop], inplace=True)
#
# datetimes = marketData.index.values
# sentimentScores = []
# sentimentCounts = []
# for i in range(marketData.shape[0]):
#     try:
#         sentimentScores.append(sentimentData.loc[datetimes[i]]['sentiment'])
#         sentimentCounts.append(sentimentData.loc[datetimes[i]]['total'])
#     except KeyError:
#         sentimentScores.append(0)
#         sentimentCounts.append(0)
#
#
# data = marketData.copy()
# data['sentiment count'] = sentimentCounts
# data['sentiment score'] = sentimentScores
#
# # data['down'] = (data['close'].shift(-1) < 0).astype(float)
# # data['up'] = (data['close'].shift(-1) >= 0).astype(float)
# data['down'] = (data['close'] < data['open']).astype(float)
# data['up'] = (data['close'] >= data['open']).astype(float)
#
#
# data = data[1:-1]
#
# print(data)
#
# pk.dump(data, open(Constants.dataDir + "Intraday/data.p", "wb"))

# ####################################################################################################################

data = pk.load(open(Constants.dataDir + "Intraday/data.p", "rb"))

minPrice = data['close'].min()
maxPrice = data['close'].max()
data = (data - data.min()) / (data.max() - data.min())


def getBatch(timeCursor):
    x, y = [], []

    x = [data.iloc[timeCursor:timeCursor + 10, :-2].as_matrix()]
    y = [data.iloc[timeCursor:timeCursor + 10, -2:].as_matrix()]

    timeCursor += 1
    if timeCursor + 10 >= data.shape[0]:
        timeCursor = 0

    return x, y, timeCursor


LSTM = LSTM(numFeatures=data.shape[1] - 2, numOutputs=2)

#################################
# TRAINING
#################################

learningRate = 0.01

for epoch in range(100):
    print("***** EPOCH:", epoch + 1, "/", 100, "*****\n")
    losses, accuracies = [], []
    timeCursor = -1
    batch = 0
    batchLosses = []
    batchAccuracies = []
    while timeCursor != 0:
        batch += 1
        timeCursor = max(timeCursor, 0)
        x, y, timeCursor = getBatch(timeCursor)
        LSTM.setBatch(x, y, learningRate, 0.0)
        LSTM.train()
        loss, accuracy, predictions, labels = LSTM.get(['loss', 'accuracy', 'predictions', 'labels'])
        batchLosses.append(loss)
        losses.append(loss)
        batchAccuracies.append(accuracy)
        accuracies.append(accuracy)
        LSTM.resetState()
        if batch % 100 == 0:
            print("\tBatch:\t\t", batch)
            print("\tBatch Loss:\t", sum(batchLosses) / len(batchLosses))
            print("\tBatch Accuracy:\t", "%.2f" % ((sum(batchAccuracies) / len(batchAccuracies)) * 100) + "%")
            print("")
    print("Loss:\t\t", sum(losses) / len(losses))
    print("Accuracy:\t", "%.2f" % ((sum(accuracies) / len(accuracies)) * 100) + "%")
    print("")
