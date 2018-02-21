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
sequenceLength = 25
batchSize = 50
numEpochs = 100
numLayers = 2
numHidden = 200

# LSTM Hyperparameters
learningRate = 0.001
forgetBias = 1.0
dropoutRate = 0.5

# Data
years = 5
endDate = dt.date(2017, 12, 31)
startDate = dt.date(2014, 1, 1)
# startDate = endDate - dt.timedelta(days=int(365.25 * Constants.years))

missingDate = dt.date(2017, 11, 8)   # MISSING FROM QUANDL
# 2017-08-07 missing from 10 tickers (including Apple and Amazon)
#   ABT ATVI ADBE AKAM AMZN AAPL CELG CMCSA COST INTC QCOM
# 2017-04-14 present for FAST and LOW but none other
# TRV and MMM missing ~20 dates

movingAverages = [3, 5, 10]

sentimentThresholdTime = dt.time(17, 30)

trainingPercentage = 0.8
numLabels = 2
predictionWindow = 3
labelThreshold = 0.03

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
