import datetime as dt
import math
import os
import pickle as pk
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars
import pytz
import quandl
import seaborn as sb
from alpha_vantage.timeseries import TimeSeries

import Constants
from Model import LSTM

quandl.ApiConfig.api_key = Constants.quandlKey

ticker = 'GOOGL'


def getTradingDates():
    # Days NYSE was open
    trainingDates = (pandas_market_calendars.get_calendar('NYSE').valid_days(
        start_date=Constants.trainingStartDate,
        end_date=Constants.trainingEndDate)
    )
    testingDates = (pandas_market_calendars.get_calendar('NYSE').valid_days(
        start_date=Constants.testingStartDate,
        end_date=Constants.testingEndDate)
    )

    # Just save date (YYYY-MM-DD)
    trainingDates = [day.date() for day in trainingDates]
    testingDates = [day.date() for day in testingDates]

    if Constants.missingDate in trainingDates:
        trainingDates.remove(Constants.missingDate)
    if Constants.missingDate in testingDates:
        testingDates.remove(Constants.missingDate)

    pk.dump(trainingDates, open(Constants.dataDir + "trainingDates.p", "wb"))
    pk.dump(testingDates, open(Constants.dataDir + "testingDates.p", "wb"))
    return trainingDates, testingDates


def getMarketData(dates):
    tickerData = quandl.get_table(
        'WIKI/PRICES',
        ticker=ticker,
        date={'gte': dates[0],
              'lte': dates[-1]},
        paginate=True
    )

    # Change date to YYYY-MM-DD and set as index
    tickerData['date'] = [day.date() for day in tickerData['date']]
    tickerData.set_index('date', inplace=True)

    # Drop unneeded columns
    tickerData.drop(['ticker', 'open', 'high', 'low', 'close', 'volume',
                     'ex-dividend', 'split_ratio'], axis=1, inplace=True)

    return tickerData


def getSP500Index(dates):
    # Get SP500 Index adjusted closing price and change col name
    av = TimeSeries(key=Constants.alphaVantageKey, output_format='pandas')
    SP500Index, _ = av.get_daily_adjusted(symbol='^GSPC', outputsize='full')
    SP500Index = SP500Index['5. adjusted close'].to_frame()
    SP500Index.columns = ['SP500']

    # Reduce to timeframe to trading dates
    SP500Index = SP500Index.loc[str(dates[0]): str(dates[-1])]
    SP500Index.drop([str(Constants.missingDate)], inplace=True)

    # Convert string index to date (YYYY-MM-DD) & set as index
    dateStrings = SP500Index.index.values
    convertedDates = []
    for i in range(len(dates)):
        split = dateStrings[i].split('-')
        date = dt.date(int(split[0]), int(split[1]), int(split[2]))
        convertedDates.append(date)
    SP500Index.set_index([convertedDates], inplace=True)

    return SP500Index


def getSentimentData(dates):
    os.chdir(Constants.dataDir + "/Knowsis/")

    with open(ticker + ".csv") as sentimentFile:
        sentimentData = sentimentFile.readlines()

    date_sentimentScores = {}

    for date in dates:
        date_sentimentScores[date] = [0, 0, 0, 0]

    for line in sentimentData[1:]:
        split = line.split(",")

        # [POS, NEG, NEU, TOT] per line
        scores = [int(score.strip()) for score in split[-4:]]

        # Convert date & time string to datetime object in NYSE timezone
        dateTimeString = split[4]
        dateTime = dt.datetime.strptime(dateTimeString, '%Y-%m-%d %H:%M:%S')
        dateTime = dateTime.replace(tzinfo=pytz.utc)
        dateTime = dateTime.astimezone(pytz.timezone('America/New_York'))

        # Iterate until sentiment time stamp is within first trading date threshold
        if dateTime.date() < dates[0] - dt.timedelta(days=1):
            continue
        if (dateTime.date() == dates[0] - dt.timedelta(days=1)
                and dateTime.time() < Constants.sentimentThresholdTime):
            continue

        # If passed threshold time, move to next day
        if dateTime.time() >= Constants.sentimentThresholdTime:
            dateTime = dateTime + dt.timedelta(days=1)

        # Iterate through dates until trading day found
        # or end of trading dates range reached
        endReached = False
        while dateTime.date() not in date_sentimentScores:
            dateTime = dateTime + dt.timedelta(days=1)
            if dateTime.date() > dates[-1]:
                endReached = True
                break

        if not endReached:
            date_sentimentScores[dateTime.date()] = [
                a + b for
                a, b in zip(date_sentimentScores[dateTime.date()], scores)
            ]
        else:
            break

    for date, scores in date_sentimentScores.items():
        # if scores[-1] < 1000:
        #     date_sentimentScores[date] = 0
        # else:
        #     date_sentimentScores[date] = (scores[0] - scores[1]) / scores[-1]
        date_sentimentScores[date] = (scores[0] - scores[1]) / scores[-1]
        # date_sentimentScores[date] = math.log(scores[0] + 0.5, 2) - math.log(scores[1] + 0.5, 2)

    os.chdir(Constants.projectDir)
    return pd.DataFrame.from_dict(date_sentimentScores, orient='index')


def normalise(data):
    minDF = data.min()
    maxDF = data.max()
    data = (data - minDF) / (maxDF - minDF)
    return data, minDF['adj_close'], maxDF['adj_close']


def denormalise(price, min, max):
    return price * (max - min) + min


def addLabels(data):
    data['down'] = (data['adj_close'].shift(-Constants.predictionWindow) <
                    data['adj_close']).astype(float)
    data['up'] = (data['adj_close'].shift(-Constants.predictionWindow) >=
                  data['adj_close']).astype(float)
    return data


# trainingDates, testingDates = getTradingDates()
# allDates = trainingDates + testingDates
# data = getMarketData(allDates)
# sp500 = getSP500Index(allDates)
# sentiment = getSentimentData(allDates)
# pk.dump(data, open(Constants.dataDir + "data.p", "wb"))
# pk.dump(sp500, open(Constants.dataDir + "sp500.p", "wb"))
# pk.dump(sentiment, open(Constants.dataDir + "sentiment.p", "wb"))


trainingDates = pk.load(open(Constants.dataDir + "trainingDates.p", "rb"))
testingDates = pk.load(open(Constants.dataDir + "testingDates.p", "rb"))[:-Constants.predictionWindow]
allDates = trainingDates + testingDates
data = pk.load(open(Constants.dataDir + "data.p", "rb"))
sentiment = pk.load(open(Constants.dataDir + "sentiment.p", "rb"))
sp500 = pk.load(open(Constants.dataDir + "sp500.p", "rb"))

data['sp500'] = sp500
data, minPrice, maxPrice = normalise(data)
data['sentiment'] = sentiment
data = addLabels(data)
data = data[:-Constants.predictionWindow]

numFeatures = data.shape[1] - Constants.numLabels


def getBatch(dateCursor, isTraining):
    x, y = [], []

    x = [data.iloc[dateCursor:(dateCursor + Constants.sequenceLength), :-Constants.numLabels].as_matrix()]
    y = [data.iloc[dateCursor:(dateCursor + Constants.sequenceLength), -Constants.numLabels:].as_matrix()]

    if isTraining:
        dateCursor += 1
        if dateCursor + Constants.sequenceLength >= len(trainingDates):
            dateCursor = 0
    else:
        dateCursor += 1
        if dateCursor + Constants.sequenceLength >= len(allDates):
            dateCursor = 0

    return x, y, dateCursor


LSTM = LSTM(numFeatures=numFeatures, numOutputs=Constants.numLabels)

#################################
# TRAINING
#################################

learningRate = 0.01
for epoch in range(Constants.numEpochs):
    print("***** EPOCH:", epoch + 1, "/", Constants.numEpochs, "*****\n")
    losses, accuracies = [], []
    dateCursor = -1
    while dateCursor != 0:
        dateCursor = max(dateCursor, 0)
        x, y, dateCursor = getBatch(dateCursor, isTraining=True)
        LSTM.setBatch(x, y, learningRate, Constants.dropoutRate)
        LSTM.train()
        loss, accuracy = LSTM.get(['loss', 'accuracy'])
        losses.append(loss)
        accuracies.append(accuracy)
        LSTM.resetState()
    print("Loss:\t\t", sum(losses) / len(losses))
    print("Accuracy:\t", "%.2f" % ((sum(accuracies) / len(accuracies)) * 100) + "%")
    print("")

#################################
# TESTING
#################################

print("***** TESTING *****\n")
sb.set()
sb.set_style("dark")

prices = data['adj_close']
denormalisedPrices = []
for price in prices:
    denormalisedPrices.append(
        denormalise(price, minPrice, maxPrice)
    )
dates = prices.index

seenLosses, seenAccuracies = [], []
unseenLosses, unseenAccuracies = [], []
dateCursor = -1
while dateCursor != 0:
    dateCursor = max(dateCursor, 0)
    x, y, dateCursor = getBatch(dateCursor, isTraining=False)
    LSTM.setBatch(x, y, 0.0, 0.0)
    loss, accuracy, predictions = LSTM.get(['loss', 'accuracy', 'predictions'])
    if dateCursor + Constants.sequenceLength < len(trainingDates):
        seenLosses.append(loss)
        seenAccuracies.append(accuracy)
    else:
        unseenLosses.append(loss)
        unseenAccuracies.append(accuracy)

    prediction = np.argmax(predictions[-1][-1])

    if prediction == 0:
        plt.scatter(
            dates[dateCursor + Constants.sequenceLength - 1],
            denormalisedPrices[dateCursor + Constants.sequenceLength - 1],
            c='r'
        )
    else:
        plt.scatter(
            dates[dateCursor + Constants.sequenceLength - 1],
            denormalisedPrices[dateCursor + Constants.sequenceLength - 1],
            c='g'
        )

    LSTM.resetState()


seenLoss = sum(seenLosses) / len(seenLosses)
seenAccuracy = sum(seenAccuracies) / len(seenAccuracies)
unseenLoss = sum(unseenLosses) / len(unseenLosses)
unseenAccuracy = sum(unseenAccuracies) / len(unseenAccuracies)

plt.plot([dates[len(trainingDates)],
          dates[len(trainingDates)]],
         [min(denormalisedPrices),
          max(denormalisedPrices)],
         c='gray')

plt.plot(dates, denormalisedPrices)
plt.title("Training Dates: {} %, Future Dates: {} %".format(
    "%.2f" % (seenAccuracy * 100),
    "%.2f" % (unseenAccuracy * 100))
)

print("Seen Loss:\t", seenLoss)
print("Seen Accuracy:\t", "%.2f" % (seenAccuracy * 100) + "%")
print("Unseen Loss:\t", unseenLoss)
print("Unseen Accuracy:", "%.2f" % (unseenAccuracy * 100) + "%\n")

plt.tight_layout()
plt.show()
