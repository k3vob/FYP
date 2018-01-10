import matplotlib.pyplot as plt

import Constants
import DataWorker
from Model import LSTM

lstm = LSTM(numFeatures=DataWorker.numFeatures, numOutputs=1)

#################################
# TRAINING
#################################


def decayLearningRate(learningRate, loss):
    if loss < learningRate:
        learningRate /= 10
    return learningRate


learningRate = Constants.initialLearningRate

for epoch in range(250):
    print("***** EPOCH:", epoch + 1, "*****\n")
    pointer = 0
    batchLosses = []
    while (pointer + Constants.sequenceLength) <= DataWorker.x.shape[0]:
        x = [DataWorker.x[pointer:(pointer + Constants.sequenceLength)]]
        y = [DataWorker.y[pointer:(pointer + Constants.sequenceLength)]]
        lstm.setBatch(learningRate, x, y)
        lstm.train()
        loss = lstm.getBatchLoss()
        batchLosses.append(loss)
        pointer += Constants.sequenceLength
    lstm.resetState()
    epochLoss = sum(batchLosses) / len(batchLosses)
    learningRate = decayLearningRate(learningRate, epochLoss)
    print(learningRate)
    print("Avg Loss:\t", epochLoss)
    print("")

pointer = 0
actual = []
predicted = []
batchLosses = []
while (pointer + Constants.sequenceLength) <= DataWorker.x.shape[0]:
    x = [DataWorker.x[pointer:(pointer + Constants.sequenceLength)]]
    y = [DataWorker.y[pointer:(pointer + Constants.sequenceLength)]]
    lstm.setBatch(x, y)
    actual.append(lstm.getBatchLabels()[-1][-1][-1])
    predicted.append(lstm.getBatchPredictions()[-1][-1][-1])
    batchLosses.append(lstm.getBatchLoss())
    pointer += 1

plt.plot(actual)
plt.plot(predicted)
plt.show()
