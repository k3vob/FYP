import os

# Current dir
wd = os.path.dirname(os.path.realpath(__file__))
dataDir = wd + '/Data/'
defaultFile = dataDir + 'train.h5'

labelPrecision = 2
sequenceLength = 25
batchSize = 10
numEpochs = 100
numLayers = 4
numHidden = 100
learningRate = 0.0001
forgetBias = 1.0
trainingPercentage = 0.8
printStep = 100
