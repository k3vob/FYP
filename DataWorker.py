import pandas as pd
import numpy as np
import Constants
import matplotlib.pyplot as plt

# Shape:        1,710,756 x 111 (ID, Timestamp, 108 features, y)
# IDs:          1424     [0, 6, 7, ... , 2156, 2158]
# Timestamps:   1813     [0, ... , 1812]
# Value Range:  Features = [-3.63698e+16, 1.04028e+18]
#                      Y = [-0.0860941, 0.0934978]

df = pd.read_hdf(Constants.default_file)
df = df.fillna(0)

# Sort by last then first timestamp
df = df.assign(
                start=df.groupby('id')['timestamp'].transform('min'),
                end=df.groupby('id')['timestamp'].transform('max'))\
                .sort_values(by=['end', 'start', 'timestamp'])

cols = list(df)
featureNames = ['derived', 'fundamental', 'technical']
features = [col for col in cols if col.split('_')[0] in featureNames]
numFeatures = len(features)
IDs = list((df['id'].unique()))                 # Sorted by ascending last timestamp
TSs = list(df['timestamp'].unique())            # Sorted

# Normalise features to mean of 0
# squash range to max distance of 1 from 0
for column in features:
    df[column] = (df[column] - df[column].mean())
    if abs(df[column].max()) > abs(df[column].min()):
        df[column] = df[column] / abs(df[column].max())
    else:
        df[column] = df[column] / abs(df[column].min())

# Normalise labels to [0,1] for ReLU activation
df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())

labelRange = df['y'].max() - df['y'].min()

plt.scatter()

# Dict of { <ID> : <[TSs that ID exists in]> }
ID_TS_dict = {}
for ID in IDs:
    ID_TS_dict[ID] = df.loc[df['id'] == ID]['timestamp'].values

# Shape: (1424, ?, 108) = (numIDs, numIDTimestamps, numFeatures)
inputMatrix = np.array([df.loc[df['id'] == ID, [feature for feature in features]].as_matrix() for ID in IDs])
# Shape: (1424, ?, 1) = (numIDs, numIDTimestamps, y)
labelMatrix = np.array([df.loc[df['id'] == ID, ['y']].as_matrix() for ID in IDs])


# ##### Do all IDs span a  single range?
# ##### Brute force algo, to be cleaned up
def generateBatch(IDPointer, TSPointer, isTraining=True):
    if isTraining:
        availableIDs = IDs[:int(len(IDs) * Constants.trainingPercentage)]
        availableTSs = TSs[:int(len(TSs) * Constants.trainingPercentage)]
        batchSize = Constants.batchSize
    else:
        availableIDs = IDs[int(len(IDs) * Constants.trainingPercentage):]
        availableTSs = TSs[int(len(TSs) * Constants.trainingPercentage):]
        IDPointer = int(len(IDs) * Constants.trainingPercentage)
        TSPointer = int(len(TSs) * Constants.trainingPercentage)
        batchSize = len(availableIDs)

    IDsComplete = False
    if isTraining and IDPointer + Constants.batchSize >= len(availableIDs):     # If number of IDs left is < batchSize
        batchSize = len(availableIDs) - IDPointer                               # Reduce this batch to how many IDs left
        IDsComplete = True                                                      # All IDs have been processed

    firstTSFound = False                            # Find the earliest timestamp in this batch
    for TS in range(TSPointer, len(availableTSs)):
        for ID_ix in range(IDPointer, IDPointer + batchSize):
            if TS in ID_TS_dict[IDs[ID_ix]]:
                TSPointer = TS
                firstTSFound = True
                break
        if firstTSFound:
            break

    inputs = np.empty(shape=(batchSize, Constants.sequenceLength, numFeatures))
    labels = np.empty(shape=(batchSize, 1))
    for i, ID_ix in enumerate(range(IDPointer, IDPointer + batchSize)):                     # Iterate over IDs in this batch
        lastLabel = np.zeros(shape=(1,))                                                    # Stores last label if sequence is padded
        for j, TS in enumerate(range(TSPointer, TSPointer + Constants.sequenceLength)):     # Iterate over timestamps in thit batch
            if TS in ID_TS_dict[IDs[ID_ix]]:                                                # If this timestamp exist for this ID
                TS_ix = np.where(ID_TS_dict[IDs[ID_ix]] == TS)                              # Get index of this timestamp for this ID
                inputs[i][j] = inputMatrix[ID_ix][TS_ix]                                    # Store features at this timestamp for this ID
                lastLabel = labelMatrix[ID_ix][TS_ix]                                       # Set current label as last
                if j == Constants.sequenceLength - 1:                                       # If last timestamp in sequence
                    labels[i] = lastLabel                                                   # then store this label
            else:                                                                           # If this timestamp doesn't exist for this ID
                inputs[i][j] = np.zeros(shape=(108,))                                       # Pad with feature array of 0s
                if j == Constants.sequenceLength - 1:                                       # Set last timestep as last label to be predicted
                    labels[i] = lastLabel

    TSPointer += Constants.sequenceLength                   # Increment TSPointer for next batch

    TSsComplete = True                                      # Check if these batchSize IDs have no more timestamps
    for ID_ix in range(IDPointer, IDPointer + batchSize):
        if TSPointer in ID_TS_dict[IDs[ID_ix]]:
            TSsComplete = False
            break

    if TSsComplete:                                         # If so, Increment IDPointer for next batch
        IDPointer += batchSize
        TSPointer = 0

    epochComplete = False
    if IDsComplete and TSsComplete:                         # If all IDs and timestamps have been processed
        epochComplete = True                                # epoch is complete

    return inputs, labels, IDPointer, TSPointer, epochComplete


def generateTestBatch():
    inputs, labels, _, _, _ = generateBatch(0, 0, isTraining=False)
    return inputs, labels
