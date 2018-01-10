import os

# Current dir
wd = os.path.dirname(os.path.realpath(__file__))
dataDir = wd + '/Data/'
defaultFile = dataDir + 'train.h5'

sequenceLength = 25
batchSize = 1
numEpochs = 2500
numLayers = 3
numHidden = 100
initialLearningRate = 0.0001
forgetBias = 1.0
dropoutRate = 0.0
trainingPercentage = 0.8
printStep = 50

years = 20
