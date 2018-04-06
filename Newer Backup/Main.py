import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

import BatchGenerator as bg
import Constants
import DataWorker as dw
from Model import LSTM

# Disbale GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def decayLearningRate(learningRate, accuracy, threshold, thresholdChange):
    if accuracy > threshold and thresholdChange != 0.0:
        learningRate /= 10
        threshold += thresholdChange
        # thresholdChange -= 0.01
    return learningRate, threshold, thresholdChange


#################################
# TRAINING
#################################

lstm = LSTM(numFeatures=dw.numFeatures, numOutputs=Constants.numLabels)
# lstm.restore()

learningRate = Constants.learningRate
threshold = 0.67
thresholdChange = 0.02

for epoch in range(Constants.numEpochs):
    print("***** EPOCH:", epoch + 1, "/", Constants.numEpochs, "*****\n")
    tickerCursor = -1
    count = 1
    while tickerCursor != 0:
        tickerCursor = max(tickerCursor, 0)
        sliceLosses = []
        sliceAccuracies = []
        dateCursor = -1
        while dateCursor != 0:
            dateCursor = max(dateCursor, 0)
            x, y, tickerCursor, dateCursor = bg.getTrainingBatch(tickerCursor, dateCursor)
            lstm.setBatch(x, y, learningRate, Constants.dropoutRate)
            lstm.train()
            batchLoss, batchAccuracy = lstm.get(['loss', 'accuracy'])
            sliceLosses.append(batchLoss)
            sliceAccuracies.append(batchAccuracy)
        lstm.resetState()
        loss = sum(sliceLosses) / len(sliceLosses)
        accuracy = sum(sliceAccuracies) / len(sliceAccuracies)
        learningRate, threshold, thresholdChange = decayLearningRate(
            learningRate, accuracy, threshold, thresholdChange)
        print(epoch + 1, ":", count, "/", dw.numSlices)
        print("Loss:\t\t", loss)
        print("Accuracy:\t", "%.2f" % (accuracy * 100) + "%")
        print("Learning Rate:\t", learningRate)
        print("")
        count += 1

    lstm.save()

#################################
# TESTING
#################################

print("***** TESTING *****\n")

sb.set()
sb.set_style("dark")

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for i, ticker in enumerate([bg.seenTestingTicker, bg.unseenTestingTicker]):
    prices = dw.data.loc[ticker]['adj_close']
    denormalisedPrices = []
    for price in prices:
        denormalisedPrices.append(
            dw.denormalise(price, dw.minPrice, dw.maxPrice)
        )
    dates = prices.index

    seenLosses, unseenLosses, seenAccuracies, unseenAccuracies = [], [], [], []
    dateCursor = -1
    while dateCursor != 0:
        dateCursor = max(dateCursor, 0)
        x, y, dateCursor = bg.getTestingBatch(ticker, dateCursor)
        lstm.setBatch(x, y, 0, 0)
        batchLoss, batchAccuracy, batchPredictions = lstm.get(
            ['loss', 'accuracy', 'predictions'])
        if dateCursor + Constants.sequenceLength < len(dw.trainingDates):
            seenLosses.append(batchLoss)
            seenAccuracies.append(batchAccuracy)
        else:
            unseenLosses.append(batchLoss)
            unseenAccuracies.append(batchAccuracy)

        prediction = np.argmax(batchPredictions[-1][-1])

        if prediction == 0:
            if i == 0:
                ax1.scatter(dates[dateCursor + Constants.sequenceLength],
                            denormalisedPrices[dateCursor + Constants.sequenceLength], c='r')
            else:
                ax2.scatter(dates[dateCursor + Constants.sequenceLength],
                            denormalisedPrices[dateCursor + Constants.sequenceLength], c='r')
        if prediction == 1:
            if i == 0:
                ax1.scatter(dates[dateCursor + Constants.sequenceLength],
                            denormalisedPrices[dateCursor + Constants.sequenceLength], c='g',)
            else:
                ax2.scatter(dates[dateCursor + Constants.sequenceLength],
                            denormalisedPrices[dateCursor + Constants.sequenceLength], c='g',)

    seenLoss = sum(seenLosses) / len(seenLosses)
    seenAccuracy = sum(seenAccuracies) / len(seenAccuracies)
    unseenLoss = sum(unseenLosses) / len(unseenLosses)
    unseenAccuracy = sum(unseenAccuracies) / len(unseenAccuracies)

    if i == 0:
        ax1.plot([dates[len(dw.trainingDates)],
                  dates[len(dw.trainingDates)]],
                 [min(denormalisedPrices),
                  max(denormalisedPrices)],
                 c='gray')

        ax1.plot(dates, denormalisedPrices)
        ax1.set_title(ticker + " (Seen Ticker - Training Dates: {}%, Future Dates: {}%)".format(
            "%.2f" % (seenAccuracy * 100),
            "%.2f" % (unseenAccuracy * 100))
        )
        print("SEEN TICKER")

    else:
        ax2.plot([dates[len(dw.trainingDates)],
                  dates[len(dw.trainingDates)]],
                 [min(denormalisedPrices),
                  max(denormalisedPrices)],
                 c='gray')

        ax2.plot(dates, denormalisedPrices)
        ax2.set_title(ticker + " (Unseen Ticker - Training Dates: {}%, Future Dates: {}%)".format(
            "%.2f" % (seenAccuracy * 100),
            "%.2f" % (unseenAccuracy * 100))
        )
        print("UNSEEN TICKER")

    print("Seen Loss:\t", seenLoss)
    print("Seen Accuracy:\t", "%.2f" % (seenAccuracy * 100) + "%")
    print("Unseen Loss:\t", unseenLoss)
    print("Unseen Accuracy:", "%.2f" % (unseenAccuracy * 100) + "%\n")

plt.tight_layout()
plt.show()
