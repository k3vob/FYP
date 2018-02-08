import os

# LSTM Architecture
sequenceLength = 25
batchSize = 50
numEpochs = 100
numLayers = 5
numHidden = 150

# LSTM Hyperparameters
learningRate = 0.0001
forgetBias = 1.0
dropoutRate = 0.25

# Data
projectDir = os.path.dirname(os.path.realpath(__file__))
dataDir = projectDir + '/Data/'
years = 5
movingAverages = [5, 15, 30, 90]
trainingPercentage = 0.8
numLabels = 2

alphaVantageKey = 'XIZOWSOCZRYV23XJ'
quandlKey = 'AysSyMCk5fZSBkHA-8_i'
# quandlKey = 'Y69LxsDnPJh1FcyYM39s'

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
