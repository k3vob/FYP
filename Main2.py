###
# PREDICTS ACTUAL PRICE
###

import os
from itertools import cycle

import matplotlib.pyplot as plt

import Constants
import DataWorker2 as dw
from Model2 import LSTM

# Disbale GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

lstm = LSTM(numFeatures=dw.numFeatures, numOutputs=1)


def decayLearningRate(learningRate, loss):
    # 0.01 -> 0.001 -> 0.0001 -> ...
    if loss < learningRate:  # * 10:
        learningRate /= 10
    return learningRate


learningRate = Constants.seedLearningRate

LRs = cycle([0.01, 0.001, 0.0001, 0.00001])

#################################
# TRAINING
#################################

for epoch in range(Constants.numEpochs):
    # if epoch == 1:
    #     learningRate = Constants.initialLearningRate
    print("***** EPOCH:", epoch + 1, "*****\n")
    tickerPointer = -1
    count = 1
    while tickerPointer != 0:
        tickerPointer = max(tickerPointer, 0)
        tickerLosses = []
        dayPointer = -1
        while dayPointer != 0:
            dayPointer = max(dayPointer, 0)
            x, y, tickerPointer, dayPointer = dw.getBatch(tickerPointer, dayPointer)
            lstm.setBatch(learningRate, x, y)
            lstm.train()
            tickerLosses.append(lstm.getBatchLoss())
            #print(lstm.getBatchPredictions()[-1][-1][-1], lstm.getBatchLabels()[-1][-1][-1])
        lstm.resetState()
        loss = sum(tickerLosses) / len(tickerLosses)
        print(count, "/", dw.numSlices)
        print("LR:\t", learningRate)
        print("Loss:\t", loss)
        print("")
        #learningRate = decayLearningRate(learningRate, loss)
        count += 1
    learningRate = next(LRs)

#################################
# TESTING
#################################

actual = []
test = []
testProjections = []
testLosses = []
tickerPointer = len(dw.tickers) - 1
dayPointer = -1
lastLabel = None
count = 0
while dayPointer != 0:
    dayPointer = max(dayPointer, 0)
    x, y, _, dayPointer = dw.getBatch(tickerPointer, dayPointer, False)
    lstm.setBatch(0, x, y)
    lstm.train()
    testLosses.append(lstm.getBatchLoss())
    label = lstm.getBatchLabels()[-1][-1][-1]
    prediction = lstm.getBatchPredictions()[-1][-1][-1]
    actual.append(label)
    test.append(prediction)
    if lastLabel is not None:
        testProjections.append([[count, count + 1], [lastLabel, prediction]])
        count += 1
    lastLabel = label
loss = sum(testLosses) / len(testLosses)
print("Testing Loss:\t", loss)

plt.plot(actual)
# for t in testProjections:
#     plt.plot(t[0], t[1], c='r')
plt.plot(test)
plt.show()
