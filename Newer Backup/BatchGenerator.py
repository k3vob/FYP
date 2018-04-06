import random

import Constants
import DataWorker as dw

unseenTestingTicker = 'AYI'
# unseenTestingTicker = random.choice(dw.tickers)
dw.tickers.remove(unseenTestingTicker)
# seenTestingTicker = unseenTestingTicker
# while seenTestingTicker == unseenTestingTicker:
#     seenTestingTicker = random.choice(dw.tickers)
seenTestingTicker = 'BAC'


def getTrainingBatch(tickerCursor, dateCursor):
    batchX, batchY = [], []
    batchSize = min(
        Constants.batchSize,
        len(dw.tickers) - tickerCursor - 1
    )  # - 1 -> Last ticker for testing

    for i in range(batchSize):
        ticker = dw.tickers[tickerCursor + i]
        if ticker == unseenTestingTicker:
            continue

        x = dw.data.loc[ticker].iloc[
            dateCursor: dateCursor + Constants.sequenceLength,
            :-Constants.numLabels
        ].as_matrix()

        y = dw.data.loc[ticker].iloc[
            dateCursor: dateCursor + Constants.sequenceLength,
            -Constants.numLabels:
        ].as_matrix()

        batchX.append(x)
        batchY.append(y)

    if dateCursor + Constants.sequenceLength >= len(dw.trainingDates):
        dateCursor = 0
    else:
        dateCursor += Constants.sequenceLength

    if dateCursor == 0:
        if batchSize < Constants.batchSize:
            tickerCursor = 0
        else:
            tickerCursor += Constants.batchSize

    return batchX, batchY, tickerCursor, dateCursor


def getTestingBatch(ticker, dateCursor):
    x = [dw.data.loc[ticker].iloc[
        dateCursor: dateCursor + Constants.sequenceLength,
        :-Constants.numLabels
    ].as_matrix()]

    y = [dw.data.loc[ticker].iloc[
        dateCursor: dateCursor + Constants.sequenceLength,
        -Constants.numLabels:
    ].as_matrix()]

    dateCursor += 1
    if dateCursor + Constants.sequenceLength >= len(dw.allDates):
        dateCursor = 0

    return x, y, dateCursor
