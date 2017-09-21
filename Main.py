import numpy as np
import tensorflow as tf
import DataWorker
# import SQLServer
from Constants import *

# Shape:        1,710,756 x 111
# IDs:          1424     [0, 6, 7, ... , 2156, 2158]
# Timestamps:   1813     [0, ... , 1812]

# Read entire raw dataset
df = DataWorker.readHDF()

# Fill NaNs with mean value of feature, then sorts by descending ID lifespan
df = DataWorker.sortByIDLifespan(df.fillna(df.mean()))

cols = list(df)
featureNames = ['derived', 'fundamental', 'technical']
features = [col for col in cols if col.split('_')[0] in featureNames]
numFeatures = len(features)
IDs = list((df['id'].unique()))                 # sorted by timestamp
timestamps = list(df['timestamp'].unique())     # sorted by timestamp

# Shape: (1424, ?, 108) = (numIDs, numIDTimestamps, numFeatures)
xMatrix = np.array([df.loc[df['id'] == ID, [feature for feature in features]].as_matrix() for ID in IDs])
# Shape: (1424, ?, 1) = (numIDs, numIDTimestamps, y)
yMatrix = np.array([df.loc[df['id'] == ID, ['y']].as_matrix() for ID in IDs])

x = tf.placeholder(tf.float32, [None, sequenceLength, numFeatures])
y = tf.placeholder(tf.float32, [None, 1])
xTensors = tf.unstack(x, axis=1)   # [seqLength tensors of shape (batchSize, numFeatures)]

W = tf.Variable(tf.random_normal([numHidden, 1]))   # Weighted matrix
b = tf.Variable(tf.random_normal([1]))              # Bias

cell = tf.contrib.rnn.BasicLSTMCell(numHidden, forget_bias=forgetBias)
outputs, finalState = tf.nn.static_rnn(cell, xTensors, dtype=tf.float32)
# predictions = [tf.add(tf.matmul(output, W), b) for output in outputs]     # List of predictions after each time step
prediction = tf.add(tf.matmul(outputs[-1], W), b)                           # Prediction after final time step
prediction = tf.tanh(prediction)                                            # Activation
mse = tf.losses.mean_squared_error(predictions=prediction, labels=y)        # Mean loss over entire batch
optimiser = tf.train.AdamOptimizer(learningRate).minimize(mse)              # Backpropagation

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    IDPointer, TSPointer = 0, 0         # Pointers to current ID and timestamp
    for batchNum in range(10000):
        batchX, batchY, IDPointer, TSPointer, newID = DataWorker.generateBatch(xMatrix, yMatrix, IDPointer, TSPointer)
        # if newID then reset state
        dict = {x: batchX, y: batchY}
        session.run(optimiser, dict)
        if (batchNum + 1) % 100 == 0 or batchNum in (0, 10000 - 1):
            batchLoss = session.run(mse, dict)
            print(str(batchNum + 1), ": ", str(batchLoss))
