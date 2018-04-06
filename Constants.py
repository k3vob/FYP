import datetime as dt
import os

# Project directories
workingDir = os.path.dirname(os.path.realpath(__file__)) + '/'
dataDir = workingDir + 'Data/'
modelDir = workingDir + 'SavedModels/'
if not os.path.exists(dataDir):
    os.makedirs(dataDir)
if not os.path.exists(modelDir):
    os.makedirs(modelDir)

# Single company symbol to be trained on
ticker = 'AAPL'

# Time frame of entire daily data set
dailyStartDate = dt.datetime(2010, 1, 1)
dailyEndDate = endDate = dt.datetime(2018, 3, 16)

sequenceLength = 100    # Number of time steps LSTM is unrolled into
predictionWindow = 3    # Number of days into future to make prediction
onlineLength = 100      # Number of time points for online inference

numLabels = 1           # Up = 1, down = 0

learningRate = 0.0001
dropout = 0.5

offlineTrainEpochs = 250
onlineTrainEpochs = 100
