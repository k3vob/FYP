import os

# Current dir
wd = os.path.dirname(os.path.realpath(__file__))
dataDir = wd + '/Data/'
defaultFile = dataDir + 'train.h5'

sequenceLength = 25
batchSize = 50
numEpochs = 100
numLayers = 2
numHidden = 100
learningRate = 0.0001
forgetBias = 1.0
dropoutRate = 0.2
trainingPercentage = 0.8
printStep = 100

years = 10
