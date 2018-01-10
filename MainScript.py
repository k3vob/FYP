import matplotlib.pyplot as plt
import tensorflow as tf

import Constants
import DataWorker

numFeatures = DataWorker.x.shape[1]
numOutputs = DataWorker.y.shape[1]

inputs = tf.placeholder(
    tf.float32, [None, Constants.sequenceLength, numFeatures])
labels = tf.placeholder(
    tf.float32, [None, Constants.sequenceLength, numOutputs])
inputsFlat = tf.unstack(inputs, axis=1)
labelsFlat = tf.unstack(labels, axis=1)
weights = tf.Variable(tf.random_normal([Constants.numHidden, 1]))
bias = tf.Variable(tf.random_normal([1]))

layers = []
for _ in range(Constants.numLayers):
    layer = tf.contrib.rnn.BasicLSTMCell(Constants.numHidden)
    layer = tf.contrib.rnn.DropoutWrapper(
        layer, output_keep_prob=(1.0 - Constants.dropoutRate))
    layers.append(layer)
network = tf.contrib.rnn.MultiRNNCell(layers)

outputs, state = tf.nn.static_rnn(
    network,
    inputsFlat,
    dtype=tf.float32)

predictions = [
    tf.add(tf.matmul(output, weights), bias)
    for output in outputs
]
predictions = tf.minimum(tf.maximum(predictions, 0), 1)

loss = tf.losses.mean_squared_error(labels=labelsFlat, predictions=predictions)

optimiser = tf.train.AdamOptimizer(Constants.learningRate).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(Constants.numEpochs):
        print("***** EPOCH", epoch + 1, "*****\n")
        pointer = 0
        batchLosses = []
        while (pointer + Constants.sequenceLength) <= DataWorker.x.shape[0]:
            x = [DataWorker.x[pointer:(pointer + Constants.sequenceLength)]]
            y = [DataWorker.y[pointer:(pointer + Constants.sequenceLength)]]
            batchDict = {inputs: x, labels: y}
            session.run(optimiser, batchDict)
            batchLosses.append(session.run(loss, batchDict))
            pointer += Constants.sequenceLength
        print("Avg Loss:", sum(batchLosses) / len(batchLosses))
        print("")

    pointer = 0
    actual = []
    predicted = []
    batchLosses = []
    while (pointer + Constants.sequenceLength) <= DataWorker.x.shape[0]:
        x = [DataWorker.x[pointer:(pointer + Constants.sequenceLength)]]
        y = [DataWorker.y[pointer:(pointer + Constants.sequenceLength)]]
        batchDict = {inputs: x, labels: y}
        actual.append(y[-1][-1][-1])
        predicted.append(session.run(predictions, batchDict)[-1][-1][-1])
        batchLosses.append(session.run(loss, batchDict))
        pointer += 1


plt.plot(actual)
plt.plot(predicted)
plt.show()

# 0.00036 -> shift by sequenceLength
