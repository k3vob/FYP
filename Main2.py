import os

import matplotlib.pyplot as plt

import Constants
import DataWorker2
from Model import LSTM

# Disbale GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

lstm = LSTM(numFeatures=DataWorker.numFeatures, numOutputs=1)


def decayLearningRate(learningRate, loss):
    # 0.01 -> 0.001 -> 0.0001 -> ...
    if loss < learningRate * 10:
        learningRate /= 10
    return learningRate


learningRate = Constants.initialLearningRate

#################################
# TRAINING
#################################

for epoch in range(Constants.numEpochs):
    print("***** EPOCH:", epoch + 1, "*****\n")
    pointer = 0
    batchLosses = []
    while (pointer + Constants.sequenceLength) < DataWorker.trainingDays:
        x = [DataWorker.x[pointer:(pointer + Constants.sequenceLength)]]
        y = [DataWorker.y[(pointer + 1):
                          (pointer + Constants.sequenceLength + 1)]]
        lstm.setBatch(learningRate, x, y)
        lstm.train()
        batchLosses.append(lstm.getBatchLoss())
        pointer += Constants.sequenceLength
    lstm.resetState()
    epochLoss = sum(batchLosses) / len(batchLosses)
    print("Learning Rate:\t", learningRate)
    print("Average Loss:\t", epochLoss)
    print("")
    learningRate = decayLearningRate(learningRate, epochLoss)

#################################
# TESTING
#################################

# Plot trained data
pointer = 0
actual = []
train = []
trainLosses = []
while (pointer + Constants.sequenceLength) < DataWorker.trainingDays:
    x = [DataWorker.x[pointer:(pointer + Constants.sequenceLength)]]
    y = [DataWorker.y[(pointer + 1):(pointer + Constants.sequenceLength + 1)]]
    lstm.setBatch(0, x, y)
    label = DataWorker.denormalise(lstm.getBatchLabels()[-1][-1])
    prediction = DataWorker.denormalise(lstm.getBatchPredictions()[-1][-1])
    actual.append(label)
    train.append(prediction)
    trainLosses.append(lstm.getBatchLoss())
    pointer += 1

# Plot unseen testing data
test = []
testLosses = []
while (pointer + Constants.sequenceLength) < DataWorker.totalDays:
    x = [DataWorker.x[pointer:(pointer + Constants.sequenceLength)]]
    y = [DataWorker.y[(pointer + 1):(pointer + Constants.sequenceLength + 1)]]
    lstm.setBatch(0, x, y)
    label = DataWorker.denormalise(lstm.getBatchLabels()[-1][-1])
    prediction = DataWorker.denormalise(lstm.getBatchPredictions()[-1][-1])
    actual.append(label)
    test.append(prediction)
    testLosses.append(lstm.getBatchLoss())
    pointer += 1

print("Seen Data Loss:  \t", sum(trainLosses) / len(trainLosses))
print("Unseen Data Loss:\t", sum(testLosses) / len(testLosses))

plt.plot(actual, label="Actual")
plt.plot(train, label="Training")
plt.plot([x for x in range(
    DataWorker.trainingDays - Constants.sequenceLength,
    DataWorker.totalDays - Constants.sequenceLength)],
    test,
    label="Testing")
plt.legend()
plt.show()
