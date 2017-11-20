import Constants
import DataWorker
import QuandlDataWorker
from Model2 import LSTM


def quandl():

    lstm = LSTM(numFeatures=3)

    # #############################################
    # TRAINING
    # #############################################

    for epoch in range(Constants.numEpochs):
        print("***** EPOCH:", epoch + 1, "*****\n")
        epochComplete = False
        tickerPointer, datePointer = 0, 0
        batchNum = 0
        while not epochComplete:
            batchNum += 1
            batchX, batchY, batchLens, tickerPointer, datePointer \
                = QuandlDataWorker.generateBatch(tickerPointer, datePointer)
            if datePointer is None:
                lstm.resetState()
                datePointer = 0
            if tickerPointer is None:
                epochComplete = True

            lstm.setBatchDict(batchX, batchY, batchLens)
            lstm.processBatch()

            if batchNum % Constants.printStep == 0 or epochComplete:
                print("Batch:\t\t", batchNum)
                print("Loss:\t\t", lstm.getBatchLoss())

                label = lstm.getLastLabels()
                pred = lstm.getLastPredictions()
                print("\nLabels\t\t Predictions")
                for lab_pred in zip(label, pred):
                    print(lab_pred[0], "\t\t", lab_pred[1])
                print("\n")


def twoSigma():

    lstm = LSTM(numFeatures=108)

    # #############################################
    # TRAINING
    # #############################################

    for epoch in range(Constants.numEpochs):
        print("***** EPOCH:", epoch + 1, "*****\n")
        epochComplete = False
        IDPointer, TSPointer = 0, 0
        batchNum = 0
        while not epochComplete:
            batchNum += 1
            batchSz, batchX, batchY, batchLens, resetState, IDPointer, TSPointer, epochComplete \
                = DataWorker.generateBatch(IDPointer, TSPointer)
            if resetState:
                lstm.resetState()
            lstm.setBatchDict(batchX, batchY, batchLens)
            lstm.processBatch()

            if batchNum % Constants.printStep == 0 or epochComplete:
                print("Batch:\t\t", batchNum)
                print("Loss:\t\t", lstm.getBatchLoss())

                label = lstm.getLastLabels()
                pred = lstm.getLastPredictions()
                print("\nLabels\t\t Predictions")
                for lab_pred in zip(label, pred):
                    print(lab_pred[0], "\t", lab_pred[1])
                print("\n")


# quandl()
twoSigma()
