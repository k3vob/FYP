import pickle as pk
import sys

import Constants
import DailyData
import Execute
from Model import LSTM

# #####################################################################
# PROCESS COMMAND LINE ARGUMENTS
# #####################################################################
instructions = """
Incorrect options:

-dtype [or] -d ['daily', 'intraday']
-generate [or] -g ['y', 'n']
-restore [or] -r ['y', 'n']
-train [or] -t ['y', 'n']
-simulate [or] -s ['y', 'n']

(All must be specified)
"""

if len(sys.argv) != 9:
    print(instructions)
    sys.exit()

opts = {}
for i, arg in enumerate(sys.argv):
    if arg[0] != '-':
        continue
    choice = sys.argv[i + 1][0].lower()
    if arg in ['-g', '-generate'] and choice in ['y', 'n']:
        opts['g'] = choice
        continue
    if arg in ['-r', '-restore'] and choice in ['y', 'n']:
        opts['r'] = choice
        continue
    if arg in ['-t', '-train'] and choice in ['y', 'n']:
        opts['t'] = choice
        continue
    if arg in ['-s', '-simulate'] and choice in ['y', 'n']:
        opts['s'] = choice
        continue
    print(instructions)
    sys.exit()

# #####################################################################
# RETRIEVE DATA
# #####################################################################
if opts['g'] == 'y':
    DailyData.generateDataSet()
prices = pk.load(open(Constants.dataDir + 'dailyPrices.p', 'rb'))
data = pk.load(open(Constants.dataDir + 'dailyData.p', 'rb'))

offlineData = data[:-Constants.onlineLength]
onlineData = data[-Constants.onlineLength - Constants.sequenceLength + 1:]

numLabels = Constants.numLabels
numFeatures = data.shape[1] - numLabels

# #####################################################################
# GENERATE LSTM MODEL
# #####################################################################

lstm = LSTM(
    numFeatures=numFeatures,
    numOutputs=numLabels,
    sequenceLength=100,
    unitsPerLayer=[250, 100],
    regularise=True
)

bestLoss = 1.0
bestEpoch = 0

if opts['r'] == 'y':
    try:
        lstm.restore()
        bestLoss = pk.load(open(Constants.modelDir + 'bestLoss.p', 'rb'))
        bestEpoch = pk.load(open(Constants.modelDir + 'bestEpoch.p', 'rb'))
        print("\nMODEL LOADED (Loss: {})".format(bestLoss))
    except Exception:
        print("""
        ERROR:
        Unable to restore model.
        Does a stored model exist?
        Have you changed the LSTM architecture?
        """)
        sys.exit()

# #####################################################################
# TRAIN MODEL
# #####################################################################
if opts['t'] == 'y':
    Execute.train(lstm, offlineData, bestEpoch, bestLoss)

# #####################################################################
# SIMULATE PREDICTIONS
# #####################################################################
if opts['s'] == 'y':
    Execute.simulate(lstm, onlineData, prices, Constants.ticker)
