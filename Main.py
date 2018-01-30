import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

import Constants
import DataWorker as dw
from Model import LSTM

# Disbale GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

lstm = LSTM(numFeatures=dw.numFeatures, numOutputs=Constants.numLabels)


def decayLearningRate(learningRate, loss):
    # 0.01 -> 0.001 -> 0.0001 -> ...
    if loss < learningRate:  # * 10:
        learningRate /= 10
    return learningRate


# learningRate = Constants.seedLearningRate
learningRate = 0.0001

#################################
# TRAINING
#################################

for epoch in range(Constants.numEpochs):
    # if epoch == 1:
    #    learningRate = Constants.initialLearningRate
    print("***** EPOCH:", epoch + 1, "*****\n")
    tickerPointer = -1
    count = 1
    while tickerPointer != 0:
        tickerPointer = max(tickerPointer, 0)
        sliceLosses = []
        sliceAccuracies = []
        dayPointer = -1
        while dayPointer != 0:
            dayPointer = max(dayPointer, 0)
            x, y, tickerPointer, dayPointer = dw.getBatch(dayPointer, tickerPointer, False)
            lstm.setBatch(learningRate, x, y)
            lstm.train()
            batchLoss, batchAccuracy = lstm.get(['loss', 'accuracy'])
            sliceLosses.append(batchLoss)
            sliceAccuracies.append(batchAccuracy)
        lstm.resetState()
        loss = sum(sliceLosses) / len(sliceLosses)
        accuracy = sum(sliceAccuracies) / len(sliceAccuracies)
        print(count, "/", dw.numSlices)
        print("LR:\t\t", learningRate)
        print("Loss:\t\t", loss)
        print("Accuracy:\t", "%.2f" % (accuracy * 100) + "%")
        print("")
        # learningRate = decayLearningRate(learningRate, loss)
        count += 1

#################################
# TESTING
#################################

sb.set()
sb.set_style("dark")

prices = dw.df.loc[dw.testingTicker]['adj_close']
dates = prices.index

plt.plot([dates[dw.numTrainingDays],
          dates[dw.numTrainingDays]],
         [prices.min(), prices.max()],
         c='gray')

plt.plot(dates, prices)

seenLosses, unseenLosses, seenAccuracies, unseenAccuracies = [], [], [], []
dayPointer = -1
while dayPointer != 0:
    dayPointer = max(dayPointer, 0)
    x, y, _, dayPointer = dw.getBatch(dayPointer)
    lstm.setBatch(0, x, y)
    batchLoss, batchAccuracy, batchPredictions = lstm.get(['loss', 'accuracy', 'predictions'])
    if dayPointer + Constants.sequenceLength < dw.numTrainingDays:
        seenLosses.append(batchLoss)
        seenAccuracies.append(batchAccuracy)
    else:
        unseenLosses.append(batchLoss)
        unseenAccuracies.append(batchAccuracy)

    prediction = np.argmax(batchPredictions[-1][-1])

    if prediction == 0:
        plt.scatter(dates[dayPointer + Constants.sequenceLength],
                    prices[dayPointer + Constants.sequenceLength], c='r', marker='v')
    if prediction == 1:
        plt.scatter(dates[dayPointer + Constants.sequenceLength],
                    prices[dayPointer + Constants.sequenceLength], c='r',)
    if prediction == 2:
        plt.scatter(dates[dayPointer + Constants.sequenceLength],
                    prices[dayPointer + Constants.sequenceLength], c='g')
    if prediction == 3:
        plt.scatter(dates[dayPointer + Constants.sequenceLength],
                    prices[dayPointer + Constants.sequenceLength], c='g', marker='^')

seenLoss = sum(seenLosses) / len(seenLosses)
seenAccuracy = sum(seenAccuracies) / len(seenAccuracies)
unseenLoss = sum(unseenLosses) / len(unseenLosses)
unseenAccuracy = sum(unseenAccuracies) / len(unseenAccuracies)
print("Seen Loss:\t", seenLoss)
print("Seen Accuracy:\t", "%.2f" % (seenAccuracy * 100) + "%")
print("Unseen Loss:\t", unseenLoss)
print("Unseen Accuracy:", "%.2f" % (unseenAccuracy * 100) + "%")

plt.title(dw.testingTicker)
plt.show()
