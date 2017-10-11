import os

# Current dir
wd = os.path.dirname(os.path.realpath(__file__))
data_dir = wd + '/Data/'
default_file = data_dir + 'train.h5'

labelPrecision = 2
labelRange = 10 ** labelPrecision + 1
sequenceLength = 25
batchSize = 10
numEpochs = 100
numLayers = 1
numHidden = 100
learningRate = 0.0001
forgetBias = 1.0
trainingPercentage = 0.8
