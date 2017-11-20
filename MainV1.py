import Constants
import DataWorker
from Model import LSTM

inputShape = [Constants.sequenceLength, DataWorker.numFeatures]
outputShape = [Constants.sequenceLength, 1]

LSTM = LSTM(inputShape, outputShape)

# #############################################
# TRAINING
# #############################################
# Epoch:		 16
# Epoch Acc:	 ['85.52', '91.52', '91.96', '92.09', '92.15', '92.19', '92.21', '92.23', '92.25', '92.27', '92.28', '92.29', '92.29', '92.30', '92.30']
epochAccs = []
for epoch in range(Constants.numEpochs):
    print("***** EPOCH:", epoch + 1, "*****\n")
    IDPointer, TSPointer = 0, 0
    epochComplete = False
    batchNum = 0
    epochAcc = 0
    while not epochComplete:
        batchNum += 1
        batchSize, batchX, batchY, batchLengths, resetState, IDPointer, TSPointer, epochComplete = DataWorker.generateBatch(
            IDPointer, TSPointer)
        LSTM.setBatchDict(batchSize, batchX, batchY, batchLengths)
        if resetState:
            LSTM.resetState()
        LSTM.processBatch()
        epochAcc += LSTM.getBatchAccuracy()
        if epochComplete:
            epochAccs.append(str("%.2f" % (epochAcc / batchNum)))
        if batchNum % Constants.printStep == 0 or epochComplete:
            print("Epoch:\t\t", str(epoch + 1))
            print("Epoch Acc:\t", epochAccs)
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
