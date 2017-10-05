import DataWorker
import Constants
from Model import LSTM

inputDim = [Constants.sequenceLength, DataWorker.numFeatures]
outputDim = [1]

lstm = LSTM(inputDimensionList=inputDim, outputDimensionList=outputDim)

# #############################################
# TRAINING
# #############################################
for epoch in range(Constants.numEpochs):
    print("***** EPOCH:", epoch + 1, "*****\n")
    IDPointer, TSPointer = 0, 0         # Pointers to current ID and timestamp
    epochComplete = False
    batchNum = 0
    while not epochComplete:
        batchNum += 1
        batchX, batchY, IDPointer, TSPointer, epochComplete = DataWorker.generateBatch(IDPointer, TSPointer)
        lstm.setBatchDict(batchX, batchY)
        batchPredictions, batchLabels, batchLoss, batchAccuracy = lstm.runBatch()
        if batchNum % 1000 == 0 or epochComplete:
            print("Iteration:", batchNum)
            print("Pred:", batchPredictions[-1], "\tLabel:", batchLabels[-1])
            print("Loss:", batchLoss)
            print("Accuracy:", str("%.2f" % (batchAccuracy * 100) + "%\n"))

# #############################################
# TESTING
# #############################################
testX, testY = DataWorker.generateTestBatch()
lstm.setBatchDict(testX, testY)
_, _, _, testAccuracy = lstm.runBatch()
print("Testing Accuracy:", str("%.2f" % (testAccuracy * 100) + "%"))

lstm.kill()
