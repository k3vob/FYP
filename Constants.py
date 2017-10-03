import os

# Current dir
wd = os.path.dirname(os.path.realpath(__file__))
data_dir = wd + '/Data/'
default_file = data_dir + 'train.h5'

sequenceLength = 20
batchSize = 10
numEpochs = 10
numHidden = 100
learningRate = 0.0001
forgetBias = 1.0
trainingPercentage = 0.8
