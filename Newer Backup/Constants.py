import datetime as dt
import os

# Directories
projectDir = os.path.dirname(os.path.realpath(__file__))
dataDir = projectDir + '/Data/'
savedModelsDir = projectDir + '/SavedModels/'

# API Autorisation Keys
alphaVantageKey = 'XIZOWSOCZRYV23XJ'
quandlKey = 'AysSyMCk5fZSBkHA-8_i'

# LSTM Architecture
sequenceLength = 10
batchSize = 100
numEpochs = 10
numLayers = 1
numHidden = 150

# LSTM Hyperparameters
learningRate = 0.001
forgetBias = 1.0
dropoutRate = 0.5

# Data
testingLength = 0.5  # Years
testingEndDate = dt.date(2018, 2, 9)
testingStartDate = testingEndDate - dt.timedelta(days=int(testingLength * 365.25))

trainingStartDate = dt.date(2014, 1, 1)
trainingEndDate = testingStartDate - dt.timedelta(days=1)

missingDate = dt.date(2017, 11, 8)   # MISSING FROM QUANDL
# 2017-08-07 missing from 10 tickers (including Apple and Amazon)
#   ABT ATVI ADBE AKAM AMZN AAPL CELG CMCSA COST INTC QCOM
# 2017-04-14 present for FAST and LOW but none other
# TRV and MMM missing ~20 dates

movingAverages = [3, 5, 10]

sentimentThresholdTime = dt.time(16, 0)

numLabels = 2
predictionWindow = 5

# Can't match ticker with Knowsis ID
hardcodedTickers = {
    'HPQ': 'hewlettpackardcompany',
    'GOOGL': 'googleinc',
    'BLK': 'blackrockinc',
    'HRB': 'hrblockinc',
    'IFF': 'internationalflavorsfragrancesinc',
    'LLY': 'elilillyandco',
    'RL': 'ralphlaurencorp'
}

# Not in Knowsis data
ignoredTickers = [
    'ARNC',
    'GGP',
    'WEC',
    'HCN',
    'AVGO',
    'DISCK'
]
