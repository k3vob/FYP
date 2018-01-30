import os

# Current dir
wd = os.path.dirname(os.path.realpath(__file__))
dataDir = wd + '/Data/'
defaultFile = dataDir + 'train.h5'


# LSTM Architecture
sequenceLength = 25
batchSize = 50
numEpochs = 600
numLayers = 5
numHidden = 150

# LSTM Hyperparameters
seedLearningRate = 0.001   # Needed to avoid predicting all 0.0
initialLearningRate = 0.0001
forgetBias = 1.0
dropoutRate = 0.0

# Data
numLabels = 4
years = 5
movingAverages = [5, 15, 30, 90]
returnTarget = 0.03
trainingPercentage = 0.8
