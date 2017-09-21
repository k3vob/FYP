import pandas as pd
import os

wd = os.path.dirname(os.path.realpath(__file__))    # Current dir
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
