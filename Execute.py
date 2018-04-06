import pickle as pk

import matplotlib.pyplot as plt
import seaborn as sb

import Batcher
import Constants


def train(lstm, data, bestEpoch, bestLoss):
    print("\nOFFLINE TRAINING")
    for epoch in range(Constants.offlineTrainEpochs - bestEpoch):
        print("\nEPOCH", bestEpoch + epoch + 1, "/", Constants.offlineTrainEpochs)
        epochLosses, epochAccuracies = [], []
        cursor = 0
        epochComplete = False
        while not epochComplete:
            x, y, cursor = Batcher.getNextTrainBatch(data, cursor)
            lstm.setBatch(x, y, Constants.learningRate, Constants.dropout)
            loss, accuracy = lstm.get(['loss', 'accuracy'])
            epochLosses.append(loss)
            epochAccuracies.append(accuracy)
            lstm.train()
            if cursor == 0:
                epochComplete = True
        epochLoss = sum(epochLosses) / len(epochLosses)
        epochAccuracy = sum(epochAccuracies) / len(epochAccuracies)
        if epochLoss < bestLoss:
            lstm.save()
            pk.dump(epochLoss, open(Constants.modelDir + "bestLoss.p", "wb"))
            pk.dump(epoch + 1, open(Constants.modelDir + "bestEpoch.p", "wb"))
        print("Loss:", epochLoss)
        print("Acc: ", "%.2f" % (epochAccuracy * 100) + "%")


def simulate(lstm, data, prices, ticker):
    prices = prices[-(Constants.onlineLength + Constants.predictionWindow):]
    targets = prices[Constants.predictionWindow:]
    # PREPARE PLOTS
    red = "#D32F2F"
    blue = "#039BE5"
    black = "#424242"
    sb.set()
    sb.set_context("talk")
    sb.set_style("dark")
    plt.ion()
    figure, (pricesPlot, returnsPlot) = plt.subplots(2, 1)
    pricesPlot.set_xlim(0, 100)
    pricesPlot.set_ylim(min(prices) - 10, max(prices) + 10)
    pricesPlot.set_title("{} Stock Price (Last {} Days)".format(ticker, Constants.onlineLength))
    pricesPlot.set_ylabel("Price")
    returnsPlot.set_xlim(0, 100)
    returnsPlot.set_ylim(-100, 100)
    returnsPlot.set_title("LSTM Model Cumulative Percentage Returns")
    returnsPlot.set_xlabel("Days")
    returnsPlot.set_ylabel("Returns (%)")
    returnsPlot.plot([0, 100], [0, 0], c=black)

    # SIMULATION
    pricesX, pricesY = [], []
    returnsX, returnsY = [], []
    cumulativeReturns = 0
    trainLosses, trainAccuracies = [], []
    testLosses, testAccuracies = [], []
    cursor = 0
    dataComplete = False
    while not dataComplete:
        print("\nPREDICTION:", cursor + 1, '/', data.shape[0] - Constants.sequenceLength + 1)
        # ###########################################################
        # TRAIN
        # ###########################################################
        x, y = Batcher.getNextOnlineBatch(data, cursor)
        lstm.setBatch(x, y, Constants.learningRate, Constants.dropout)
        for epoch in range(Constants.onlineTrainEpochs):
            lstm.train()
            trainLoss, trainAccuracy = lstm.get(['loss', 'accuracy'])
            trainLosses.append(trainLoss)
            trainAccuracies.append(trainAccuracy)
        # ###########################################################
        # TEST
        # ###########################################################
        x, y, cursor = Batcher.getNextOnlineBatch(data, cursor, predict=True)
        lstm.setBatch(x, y)
        testLoss, testAccuracy, labels, predictions = lstm.get(
            ['loss', 'accuracy', 'labels', 'roundedPredictions'])
        testLosses.append(testLoss)
        testAccuracies.append(testAccuracy)
        print("Train Loss: ", sum(trainLosses) / len(trainLosses))
        print("Train Acc:  ", "%.2f" % ((sum(trainAccuracies) / len(trainAccuracies)) * 100) + "%")
        print("\nTest Loss:  ", sum(testLosses) / len(testLosses))
        print("Test Acc:   ", "%.2f" % ((sum(testAccuracies) / len(testAccuracies)) * 100) + "%")
        if cursor == 0:
            dataComplete = True

        # ###########################################################
        # UPDATE PLOTS
        # ###########################################################
        dayReturn = abs(((targets[cursor] - prices[cursor]) / prices[cursor]) * 100)
        if labels[0][0] != predictions[0][0]:
            dayReturn = -dayReturn
        cumulativeReturns += dayReturn

        print("\nDay Return:\t   ", "%.2f" % dayReturn + "%")
        print("Cumulative Return: ", "%.2f" % cumulativeReturns + "%")

        if cumulativeReturns > 100:
            returnsPlot.set_ylim(-100, cumulativeReturns + 10)
        if cumulativeReturns < -100:
            returnsPlot.set_ylim(cumulativeReturns - 10, 100)

        if cursor != 0:
            pricesX.append(cursor)
            pricesY.append(prices[cursor])
            returnsX.append(cursor)
            returnsY.append(cumulativeReturns)
        else:
            pricesX.append(Constants.sequenceLength)
            pricesY.append(prices[Constants.sequenceLength])
            returnsX.append(Constants.sequenceLength)
            returnsY.append(cumulativeReturns)

        pricesPlot.plot(pricesX, pricesY, c=blue)
        returnsPlot.plot(returnsX, returnsY, c=red)
        plt.pause(0.01)

        if cursor == 0:
            plt.savefig(Constants.workingDir + "Simulation Plot")
            while True:
                plt.pause(1)
