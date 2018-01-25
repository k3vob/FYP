import matplotlib.pyplot as plt
import tensorflow as tf

import Constants
import DataWorker

numFeatures = DataWorker.numFeatures
numOutputs = 1

batchSize = tf.placeholder(tf.int32, [])
inputs = tf.placeholder(
    tf.float32, [None, Constants.sequenceLength, numFeatures]
)
labels = tf.placeholder(
    tf.float32, [None, Constants.sequenceLength, numOutputs]
)
inputsUnrolled = tf.unstack(inputs, axis=1)
labelsUnrolled = tf.unstack(labels, axis=1)

weights = tf.Variable(tf.random_normal([Constants.numHidden, 1]))
bias = tf.Variable(tf.random_normal([1]))

layers = []
for _ in range(Constants.numLayers):
    layer = tf.contrib.rnn.BasicLSTMCell(Constants.numHidden)
    layer = tf.contrib.rnn.DropoutWrapper(
        layer, output_keep_prob=0.8)
    layers.append(layer)
network = tf.contrib.rnn.MultiRNNCell(layers)

outputs, state = tf.nn.static_rnn(
    network,
    inputsUnrolled,
    dtype=tf.float32)

outputs = [
    tf.add(tf.matmul(output, weights), bias)
    for output in outputs
]

predictions = tf.nn.relu(outputs)
predictions = tf.unstack(predictions, axis=0)

loss = tf.losses.mean_squared_error(labels=labelsUnrolled, predictions=predictions)

optimiser = tf.train.AdamOptimizer(0.0001).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(Constants.numEpochs):
        print("***** EPOCH:", epoch + 1, "*****\n")
        tickerPointer = -1
        count = 1
        while tickerPointer != 0:
            tickerPointer = max(tickerPointer, 0)
            tickerLosses = []
            dayPointer = -1
            while dayPointer != 0:
                dayPointer = max(dayPointer, 0)
                x, y, tickerPointer, dayPointer = DataWorker.getBatch(tickerPointer, dayPointer)
                feed_dict = {inputs: x, labels: y}
                session.run(optimiser, feed_dict)
                err = session.run(loss, feed_dict)
                tickerLosses.append(err)
                pred = session.run(predictions, feed_dict)[-1][-1][-1]
                lab = session.run(labels, feed_dict)[-1][-1][-1]
                print(pred, lab)
            err = sum(tickerLosses) / len(tickerLosses)
            print(count, "/", DataWorker.numTickerGroups)
            print("Loss:\t", err)
            print("")
            count += 1
