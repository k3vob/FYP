import numpy as np
import pandas as pd

import Constants

# Shape:        1,710,756 x 111 (ID, Timestamp, 108 features, y)
# IDs:          1424     [0, 6, 7, ... , 2156, 2158]
# Timestamps:   1813     [0, ... , 1812]
# Value Range:  Features = [-3.63698e+16, 1.04028e+18]
#                      Y = [-0`.0860941, 0.0934978]

df = pd.read_hdf(Constants.defaultFile)
df = df.fillna(0)

# Sort by last then first timestamp
df = df.assign(start=df.groupby('id')['timestamp'].transform('min'),
               end=df.groupby('id')['timestamp'].transform('max'))\
    .sort_values(by=['end', 'start', 'timestamp'])

cols = list(df)
featureNames = ['derived', 'fundamental', 'technical']
features = [col for col in cols if col.split('_')[0] in featureNames]
numFeatures = len(features)
IDs = list((df['id'].unique()))
TSs = list(df['timestamp'].unique())

# Dict of { <ID> : <[TSs that ID exists in]> }
ID_TS_dict = {}
sortingDict = {}            # ##### Only needed for overlap ratio
for ID in IDs:
    ID_TS_dict[ID] = df.loc[df['id'] == ID]['timestamp'].values
    sortingDict[ID] = df.loc[df['id'] == ID]['timestamp'].values

# # Create list of IDs sorted by overlap ratio
# sortedIDs = []
# for ID in IDs:
#     if len(ID_TS_dict[ID]) == len(TSs):
#         sortedIDs.append(ID)
#         del sortingDict[ID]
#         break
#
# while len(sortingDict) > 0:
#     lastID = sortedIDs[-1]
#     lastTSList = ID_TS_dict[lastID]
#     bestRatio = [None, -1]
#     for ID, TSList in sortingDict.items():
#         numOverlaps = len(set(lastTSList).intersection(set(TSList)))
#         overlapRatio = numOverlaps / len(lastTSList)
#         if overlapRatio > bestRatio[1]:
#             bestRatio = [ID, overlapRatio]
#     sortedIDs.append(bestRatio[0])
#     del sortingDict[bestRatio[0]]
#     print(len(sortingDict))

# Normalise features to mean of 0
# squash range to max distance of 1 from 0
# for column in features:
#     df[column] = (df[column] - df[column].mean())
#     if abs(df[column].max()) > abs(df[column].min()):
#         df[column] = df[column] / abs(df[column].max())
#     else:
#         df[column] = df[column] / abs(df[column].min())

# Normalise labels to [0,1] for ReLU activation
df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())

# # Round to n decimal places
# df['y'] = df['y'].round(Constants.labelPrecision)

# Shape: (1424, ?, 108) = (numIDs, numIDTimestamps, numFeatures)
inputMatrix = np.array(
    [df.loc[df['id'] == ID, [feature for feature in features]].as_matrix() for ID in IDs])
# Shape: (1424, ?, 1) = (numIDs, numIDTimestamps, y)
labelMatrix = np.array([df.loc[df['id'] == ID, ['y']].as_matrix() for ID in IDs])


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
    # If number of IDs left is < batchSize
    if isTraining and IDPointer + Constants.batchSize >= len(availableIDs):
        # Reduce this batch to how many IDs left
        batchSize = len(availableIDs) - IDPointer
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

    # ##### IF BREAK, FIND LENGTH BEFORE 2ND SEQUENCE
    # ##### STORE ALL LABELS (batch, sequenceLength, 1)
    # ##### STORE LENGTH IN ARRAY
    # ##### PAD REST UP TO ###25###

    actualSequenceLength = Constants.sequenceLength
    for ID_ix in range(IDPointer, IDPointer + batchSize):
        for i, TS in enumerate(range(TSPointer, TSPointer + Constants.sequenceLength - 1)):
            if TS not in ID_TS_dict[IDs[ID_ix]] and (TS + 1) in ID_TS_dict[IDs[ID_ix]]:
                if (i + 1) < actualSequenceLength:
                    actualSequenceLength = (i + 1)

    inputs = np.empty(shape=(batchSize, Constants.sequenceLength, numFeatures))
    labels = np.empty(shape=(batchSize, Constants.sequenceLength, 1))
    lengths = np.empty(shape=(batchSize,))
    # Iterate over IDs in this batch
    for i, ID_ix in enumerate(range(IDPointer, IDPointer + batchSize)):
        lengthFound = False
        # Iterate over timestamps in thit batch
        for j, TS in enumerate(range(TSPointer, TSPointer + Constants.sequenceLength)):
            # If this timestamp exist for this ID
            if TS in ID_TS_dict[IDs[ID_ix]]:
                # Get index of this timestamp for this ID
                TS_ix = np.where(ID_TS_dict[IDs[ID_ix]] == TS)
                # Store features at this timestamp for this ID
                inputs[i][j] = inputMatrix[ID_ix][TS_ix]
                labels[i][j] = labelMatrix[ID_ix][TS_ix]
            else:                                                                           # If this timestamp doesn't exist for this ID
                # Pad with feature array of 0s
                inputs[i][j] = np.zeros(shape=(108,))
                labels[i][j] = np.zeros(shape=(1,))
                if not lengthFound:
                    lengths[i] = j
                    lengthFound = True
        if not lengthFound:
            lengths[i] = Constants.sequenceLength

    TSPointer += Constants.sequenceLength                   # Increment TSPointer for next batch

    # Check if these batchSize IDs have no more timestamps
    TSsComplete = True
    for ID_ix in range(IDPointer, IDPointer + batchSize):
        if TSPointer in ID_TS_dict[IDs[ID_ix]]:
            TSsComplete = False
            break

    resetState = False
    if TSsComplete:                                         # If so, Increment IDPointer for next batch
        IDPointer += batchSize
        TSPointer = 0
        resetState = True

    epochComplete = False
    if IDsComplete and TSsComplete:                         # If all IDs and timestamps have been processed
        epochComplete = True                                # epoch is complete

    return batchSize, inputs, labels, lengths, resetState, IDPointer, TSPointer, epochComplete


def generateTestBatch():
    batchSize, inputs, labels, lengths, resetState, _, _, _ = generateBatch(0, 0, isTraining=False)
    return batchSize, inputs, labels, lengths, resetState
