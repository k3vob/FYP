import pandas as pd
import numpy as np
import Constants

# Shape:        1,710,756 x 111 (ID, Timestamp, 108 features, y)
# IDs:          1424     [0, 6, 7, ... , 2156, 2158]
# Timestamps:   1813     [0, ... , 1812]

df = pd.read_hdf(Constants.default_file)
df = df.fillna(df.mean())
df = df.assign(freq=df.groupby('id')['id'].transform('count')).sort_values(by=['freq', 'id', 'timestamp'], ascending=[False, True, True])

cols = list(df)
featureNames = ['derived', 'fundamental', 'technical']
features = [col for col in cols if col.split('_')[0] in featureNames]
numFeatures = len(features)
IDs = list((df['id'].unique()))                 # sorted by timestamp
timestamps = list(df['timestamp'].unique())     # sorted by timestamp

# Shape: (1424, ?, 108) = (numIDs, numIDTimestamps, numFeatures)
inputMatrix = np.array([df.loc[df['id'] == ID, [feature for feature in features]].as_matrix() for ID in IDs])
# Shape: (1424, ?, 1) = (numIDs, numIDTimestamps, y)
labelMatrix = np.array([df.loc[df['id'] == ID, ['y']].as_matrix() for ID in IDs])


def generateBatch(IDPointer, TSPointer):
    """doctring."""
    inputs, labels = [], []
    newID = False               # ############## NEEDED?
    for i in range(Constants.batchSize):
        sequence = inputMatrix[IDPointer][TSPointer + i * Constants.sequenceLength:TSPointer + (i + 1) * Constants.sequenceLength]
        if len(sequence) == Constants.sequenceLength:
            inputs.append(sequence)
            labels.append(labelMatrix[IDPointer][TSPointer + (i + 1) * Constants.sequenceLength - 1])
        else:
            pad = np.zeros((1, numFeatures))
            for _ in range(Constants.sequenceLength - len(sequence)):
                sequence = np.concatenate((pad, sequence))
            inputs.append(sequence)
            labels.append(labelMatrix[IDPointer][-1])
            IDPointer += 1
            TSPointer = 0
            newID = True
            return inputs, labels, IDPointer, TSPointer, newID
    TSPointer += Constants.batchSize * Constants.sequenceLength
    return inputs, labels, IDPointer, TSPointer, newID
