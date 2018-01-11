import matplotlib.pyplot as plt

import Constants
import DataWorker
from Model import LSTM

lstm = LSTM(numFeatures=DataWorker.numFeatures, numOutputs=1)

#################################
# TRAINING
#################################


def decayLearningRate(learningRate, loss):
    # 0.01 -> 0.001 -> 0.0001 -> ...
    if loss < learningRate:
        learningRate /= 10
    return learningRate


learningRate = Constants.initialLearningRate

for epoch in range(Constants.numEpochs):
    print("***** EPOCH:", epoch + 1, "*****\n")
    pointer = 0
    batchLosses = []
    while (pointer + Constants.sequenceLength) <= DataWorker.trainingDays:
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
    print("Learning Rate:\t", learningRate)
    print("Average Loss:\t", epochLoss)
    print("")

#################################
# TESTING
#################################

# Plot trained data
pointer = 0
actual = []
train = []
trainLosses = []
while (pointer + Constants.sequenceLength) <= DataWorker.trainingDays:
    x = [DataWorker.x[pointer:(pointer + Constants.sequenceLength)]]
    y = [DataWorker.y[pointer:(pointer + Constants.sequenceLength)]]
    lstm.setBatch(0, x, y)
    actual.append(lstm.getBatchLabels()[-1][-1][-1])
    train.append(lstm.getBatchPredictions()[-1][-1][-1])
    trainLosses.append(lstm.getBatchLoss())
    pointer += 1

# Plot unseen testing data
test = []
testLosses = []
while (pointer + Constants.sequenceLength) <= DataWorker.totalDays:
    x = [DataWorker.x[pointer:(pointer + Constants.sequenceLength)]]
    y = [DataWorker.y[pointer:(pointer + Constants.sequenceLength)]]
    lstm.setBatch(0, x, y)
    actual.append(lstm.getBatchLabels()[-1][-1][-1])
    test.append(lstm.getBatchPredictions()[-1][-1][-1])
    testLosses.append(lstm.getBatchLoss())
    pointer += 1

print("Seen Data Loss:\t", sum(trainLosses) / len(trainLosses))
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
