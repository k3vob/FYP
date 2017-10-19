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
        batchSize, batchX, batchY, batchLengths, resetState, IDPointer, TSPointer, epochComplete = DataWorker.generateBatch(IDPointer, TSPointer)
        LSTM.setBatchDict(batchSize, batchX, batchY, batchLengths)
        if resetState:
            LSTM.resetState()
        LSTM.processBatch()
        if batchNum % Constants.printStep == 0 or epochComplete:
            print("Batch:\t\t", batchNum)
            print("Loss:\t\t", LSTM.getBatchLoss())
            print("Accuracy:\t", str("%.2f" % LSTM.getBatchAccuracy() + "%\n"))

            label = LSTM.getLastLabels()
            pred = LSTM.getLastPredictions()
            print("Labels\t\tPredictions")
            for lab_pred in zip(label, pred):
                print(lab_pred[0], "\t", lab_pred[1])
            print("\n")

# #############################################
# TESTING
# #############################################
testSize, testX, testY, testLengths, resetState = DataWorker.generateTestBatch()
LSTM.setBatchDict(testSize, testX, testY, testLengths)
if resetState:
    LSTM.resetState()
print("Testing Loss:\t\t", LSTM.getBatchLoss())
print("Testing Accuracy\t:", str("%.2f" % LSTM.getBatchAccuracy() + "%"))

LSTM.kill()
