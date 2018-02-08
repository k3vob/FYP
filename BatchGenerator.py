import random

import Constants
import DataWorker

data = DataWorker.SP500Prices
tickers = DataWorker.SP500Tickers
unseenTestingTicker = random.choice(tickers)
tickers.remove(unseenTestingTicker)
seenTestingTicker = unseenTestingTicker
while seenTestingTicker == unseenTestingTicker:
    seenTestingTicker = random.choice(tickers)


def getTrainingBatch(tickerPointer, dayPointer):
    batchX, batchY = [], []
    batchSize = min(
        Constants.batchSize,
        len(tickers) - tickerPointer - 1
    )   # - 1 -> Last ticker for testing

    for i in range(batchSize):
        ticker = tickers[tickerPointer + i]
        x = data.loc[ticker].iloc[
            dayPointer: dayPointer + Constants.sequenceLength,
            :-Constants.numLabels
        ].as_matrix()

        y = data.loc[ticker].iloc[
            dayPointer + 1: dayPointer + Constants.sequenceLength + 1,
            -Constants.numLabels:
        ].as_matrix()

        # y = y.reshape(y.shape[0], 4)
        batchX.append(x)
        batchY.append(y)

    dayPointer += Constants.sequenceLength
    if dayPointer + Constants.sequenceLength + 1 >= DataWorker.numTrainingDays:
        dayPointer = 0      # + 1 -> y is shifted by 1

    if dayPointer == 0:
        if batchSize < Constants.batchSize:
            tickerPointer = 0
        else:
            tickerPointer += Constants.batchSize

    return batchX, batchY, tickerPointer, dayPointer


def getTestingBatch(ticker, dayPointer):
    batchX, batchY = [], []

    x = data.loc[ticker].iloc[
        dayPointer:dayPointer + Constants.sequenceLength,
        :-Constants.numLabels
    ].as_matrix()

    y = data.loc[ticker].iloc[
        dayPointer + 1: dayPointer + Constants.sequenceLength + 1,
        -Constants.numLabels:
    ].as_matrix()

    # y = y.reshape(y.shape[0], 4)
    batchX.append(x)
    batchY.append(y)

    dayPointer += 1
    if dayPointer + Constants.sequenceLength + 1 >= DataWorker.numDays:
        dayPointer = 0      # + 1 -> y is shifted by 1

    return batchX, batchY, dayPointer
