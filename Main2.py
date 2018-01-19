import os

import matplotlib.pyplot as plt

import Constants
import DataWorker2 as dw
from Model import LSTM

# Disbale GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

lstm = LSTM(numFeatures=dw.numFeatures, numOutputs=1)


def decayLearningRate(learningRate, loss):
    # 0.01 -> 0.001 -> 0.0001 -> ...
    if loss < learningRate * 10:
        learningRate /= 10
    return learningRate


learningRate = Constants.initialLearningRate

#################################
# TRAINING
#################################

for epoch in range(10):
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
            # print(lstm.getBatchPredictions()[-1][-1][-1], lstm.getBatchLabels()[-1][-1][-1])
        lstm.resetState()
        loss = sum(tickerLosses) / len(tickerLosses)
        print(count, "/", dw.numTickerGroups)
        print("LR:\t", learningRate)
        print("Loss:\t", loss)
        print("")
        learningRate = decayLearningRate(learningRate, loss)
        count += 1

#################################
# TESTING
#################################

actual = []
test = []
testLosses = []
tickerPointer = len(dw.tickers) - 1
dayPointer = -1
while dayPointer != 0:
    dayPointer = max(dayPointer, 0)
    x, y, _, dayPointer = dw.getBatch(tickerPointer, dayPointer, False)
    lstm.setBatch(0, x, y)
    lstm.train()
    testLosses.append(lstm.getBatchLoss())
    actual.append(lstm.getBatchLabels()[-1][-1][-1])
    test.append(lstm.getBatchPredictions()[-1][-1][-1])
loss = sum(testLosses) / len(testLosses)
print("Testing Loss:\t", loss)

plt.plot(actual, label="Actual")
plt.plot(test, label="Testing")
plt.legend()
plt.show()
