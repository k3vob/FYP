import DataWorker
import Constants
from Model import LSTM

inputShape = [Constants.sequenceLength, DataWorker.numFeatures]
outputShape = [Constants.sequenceLength, 1]

LSTM = LSTM(inputShape, outputShape)

# #############################################
# TRAINING
# #############################################
for epoch in range(Constants.numEpochs):
    print("***** EPOCH:", epoch + 1, "*****\n")
    IDPointer, TSPointer = 0, 0
    epochComplete = False
    batchNum = 0
    while not epochComplete:
        batchNum += 1
        batchX, batchY, batchLengths, IDPointer, TSPointer, epochComplete = DataWorker.generateBatch(IDPointer, TSPointer)
        LSTM.setBatchDict(batchX, batchY, batchLengths)
        LSTM.processBatch()
        if batchNum % Constants.printStep == 0 or epochComplete:
            print("Batch:\t\t", batchNum)
            #print("Last Pred:\t", LSTM.batchPredictions()[-1][0])
            #print("Last Label:\t", LSTM.batchLabels()[-1][0])
            print("Loss:\t\t", LSTM.getBatchLoss())
            print("Accuracy:\t", str("%.2f" % LSTM.getBatchAccuracy() + "%\n"))

# #############################################
# TESTING
# #############################################
testX, testY, testLengths = DataWorker.generateTestBatch()
LSTM.setBatch(testX, testY)
testAccuracy = LSTM.batchAccuracy()
print("Testing Accuracy:", str("%.2f" % (testAccuracy * 100) + "%"))

LSTM.kill()
