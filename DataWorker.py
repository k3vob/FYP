import pandas as pd
import numpy as np
import os
from Constants import *

# Current dir
wd = os.path.dirname(os.path.realpath(__file__))
default_file = 'train.h5'


# Read .h5 file and return as DataFrame
def readHDF(filename=None):
    if not filename:
        filename = default_file
    filename = wd + '/Data/' + filename
    if filename.split('.')[-1] != 'h5':
        filename += '.h5'
    return pd.read_hdf(filename)


# Write DataFrame as .h5 file
def writeHDF(df, filename=None):
    if not filename:
        filename = 'df.h5'
    filename = wd + '/Data/' + filename
    if filename.split('.')[-1] != 'h5':
        filename += '.h5'
    df.to_hdf(filename, 'w')


# Sort dataframe by descending ID lifespan, then by ascedning ID, then by ascending timestamp
def sortByIDLifespan(df):
    return df.assign(freq=df.groupby('id')['id'].transform('count'))\
        .sort_values(by=['freq', 'id', 'timestamp'], ascending=[False, True, True])


def generateBatch(inputMatrix, labelMatrix, IDPointer, TSPointer, numFeatures):
    inputs, labels = [], []
    newID = False               # ############## NEEDED?
    for i in range(batchSize):
        sequence = inputMatrix[IDPointer][TSPointer + i * sequenceLength:TSPointer + (i + 1) * sequenceLength]
        if len(sequence) == sequenceLength:
            inputs.append(sequence)
            labels.append(labelMatrix[IDPointer][TSPointer + (i + 1) * sequenceLength - 1])
        else:
            pad = np.zeros((1, numFeatures))
            for _ in range(sequenceLength - len(sequence)):
                sequence = np.concatenate((pad, sequence))
            inputs.append(sequence)
            labels.append(labelMatrix[IDPointer][-1])
            IDPointer += 1
            TSPointer = 0
            newID = True
            return inputs, labels, IDPointer, TSPointer, newID
    TSPointer += batchSize * sequenceLength
    return inputs, labels, IDPointer, TSPointer, newID
