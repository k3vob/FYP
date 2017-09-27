import os

# Current dir
wd = os.path.dirname(os.path.realpath(__file__))
data_dir = wd + '/Data/'
default_file = data_dir + 'train.h5'

sequenceLength = 3
batchSize = 5
# epochSize = num batches in entire dataset
numEpochs = 500
numHidden = 100
learningRate = 0.001
forgetBias = 1.0
featureNames = ['derived', 'fundamental', 'technical']
