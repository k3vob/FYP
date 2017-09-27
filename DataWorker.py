import pandas as pd
import numpy as np
import Constants

# Shape:        1,710,756 x 111 (ID, Timestamp, 108 features, y)
# IDs:          1424     [0, 6, 7, ... , 2156, 2158]
# Timestamps:   1813     [0, ... , 1812]

df = pd.read_hdf(Constants.default_file)
df = df.fillna(df.mean())

# SORT BY LAST TIMESTAMP
df = df.assign(start=df.groupby('id')['timestamp'].transform('min'),
               end=df.groupby('id')['timestamp'].transform('max'))\
               .sort_values(by=['end', 'start', 'timestamp'])

cols = list(df)
featureNames = ['derived', 'fundamental', 'technical']
features = [col for col in cols if col.split('_')[0] in featureNames]
numFeatures = len(features)
IDs = list((df['id'].unique()))                 # Sorted by ascending last timestamp
numIDs = len(IDs)
timestamps = list(df['timestamp'].unique())     # Sorted
numTimestamps = len(timestamps)

ID_TS_dict = {}
for ID in IDs:
    ID_TS_dict[ID] = df.loc[df['id'] == ID]['timestamp'].values

# Shape: (1424, ?, 108) = (numIDs, numIDTimestamps, numFeatures)
inputMatrix = np.array([df.loc[df['id'] == ID, [feature for feature in features]].as_matrix() for ID in IDs])
# Shape: (1424, ?, 1) = (numIDs, numIDTimestamps, y)
labelMatrix = np.array([df.loc[df['id'] == ID, ['y']].as_matrix() for ID in IDs])


# DO ALL IDs SPAN ONE SINGLE RANGE?
# BRUTE FORCE ALGORITHM, TO BE CLEANED UP
def generateBatch(IDPointer, TSPointer):
    IDsComplete = False
    batchSize = Constants.batchSize
    if IDPointer + Constants.batchSize >= len(IDs):     # If number of IDs left is < batchSize
        batchSize = len(IDs) - IDPointer                # Reduce this batch to how many IDs left
        IDsComplete = True                              # All IDs have been processed

    firstTSFound = False                                # Find the earliest timestamp in this batch
    for TS in range(TSPointer, numTimestamps):
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


# ID = 0
# TS = 0
# epochComplete = False
# while not epochComplete:
#     a, b, ID, TS, epochComplete = generateBatch(ID, TS)
#
# print(a)
# # print(b)
#
# print(inputMatrix[-4][-1])
# print(inputMatrix[-3][-1])
# print(inputMatrix[-2][-1])
# print(inputMatrix[-1][-1])
