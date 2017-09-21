import numpy as np
import tensorflow as tf
import DataWorker
# import SQLServer

sequenceLength = 15
batchSize = 25
numHidden = 100
learningRate = 0.01
forgetBias = 1.0

# Shape      = 1,710,756 x 111
# IDs        = 1424     [0, 6, 7, ... , 2156, 2158]
# Timestamps = 1813     [0, ... , 1812]
df = DataWorker.readHDF()
df_filled = df.fillna(df.mean())

cols = list(df)
featureNames = ['derived', 'fundamental', 'technical']
features = [col for col in cols if col.split('_')[0] in featureNames]
numFeatures = len(features)
IDs = list((df['id'].unique()))         # sorted by timestamp
timestamps = list(df['timestamp'].unique())

# (1424, ?, 108) = (numIDs, numIDTimestamps, numFeatures)
xMatrix = np.array([df_filled.loc[df_filled['id'] == ID, [feature for feature in features]].as_matrix() for ID in IDs])
# (1424, ?, 1) = (numIDs, numIDTimestamps, y)
yMatrix = np.array([df_filled.loc[df_filled['id'] == ID, ['y']].as_matrix() for ID in IDs])


def generateBatch(IDPointer, TSPointer):
    inputs, labels = [], []
    newID = False
    for i in range(batchSize):
        sequence = xMatrix[IDPointer][TSPointer + i * sequenceLength:TSPointer + (i + 1) * sequenceLength]
        if len(sequence) == sequenceLength:
            inputs.append(sequence)
            labels.append(yMatrix[IDPointer][TSPointer + (i + 1) * sequenceLength - 1])
        else:
            pad = np.zeros((1, numFeatures))
            for _ in range(sequenceLength - len(sequence)):
                sequence = np.concatenate((pad, sequence))
            inputs.append(sequence)
            labels.append(yMatrix[IDPointer][-1])
            IDPointer += 1
            TSPointer = 0
            newID = True
            return inputs, labels, IDPointer, TSPointer, newID
    TSPointer += batchSize * sequenceLength
    return inputs, labels, IDPointer, TSPointer, newID


x = tf.placeholder(tf.float32, [None, sequenceLength, numFeatures])
y = tf.placeholder(tf.float32, [None, 1])
xTensors = tf.unstack(x, axis=1)   # [seqLength tensors of shape (batchSize, numFeatures)]

W = tf.Variable(tf.random_normal([numHidden, 1]))
b = tf.Variable(tf.random_normal([1]))

cell = tf.contrib.rnn.BasicLSTMCell(numHidden, forget_bias=forgetBias)
outputs, finalState = tf.nn.static_rnn(cell, xTensors, dtype=tf.float32)
# predictions = [tf.add(tf.matmul(output, W), b) for output in outputs]
prediction = tf.add(tf.matmul(outputs[-1], W), b)
prediction = tf.tanh(prediction)
mse = tf.losses.mean_squared_error(predictions=prediction, labels=y)
optimiser = tf.train.AdamOptimizer(learningRate).minimize(mse)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    IDPointer, TSPointer = 0, 0
    for batchNum in range(10000):
        batchX, batchY, IDPointer, TSPointer, newID = generateBatch(IDPointer, TSPointer)
        # if newID then reset state
        dict = {x: batchX, y: batchY}
        session.run(optimiser, dict)
        if (batchNum + 1) % 100 == 0 or batchNum in (0, 10000 - 1):
            batchLoss = session.run(mse, dict)
            print(str(batchNum + 1), ": ", str(batchLoss))
