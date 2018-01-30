import tensorflow as tf

import Constants
import DataWorker as dw

learningRate = tf.placeholder(tf.float32, [])
batchSize = tf.placeholder(tf.int32, [])
inputs = tf.placeholder(
    tf.float32, [None, Constants.equenceLength, dw.numFeatures]
)
labels = tf.placeholder(
    tf.float32, [None, Constants.sequenceLength, 1]
)
inputsUnrolled = tf.unstack(inputs, axis=1)
labelsUnrolled = tf.unstack(labels, axis=1)

weights = tf.Variable(
    tf.random_normal([Constants.numHidden, 1])
)
biases = tf.Variable(
    tf.random_normal([1])
)

layers = []
for _ in range(Constants.numLayers):
    layer = tf.contrib.rnn.BasicLSTMCell(Constants.numHidden)
    layer = tf.contrib.rnn.DropoutWrapper(
        layer, output_keep_prob=(1.0 - Constants.dropoutRate)
    )
    layers.append(layer)
network = tf.contrib.rnn.MultiRNNCell(layers)

outputs, _ = tf.nn.static_rnn(network, inputsUnrolled, dtype=tf.float32)
outputs = [tf.add(tf.matmul(output, weights), biases) for output in outputs]

predictions = tf.nn.sigmoid(outputs)
print(labelsUnrolled)
print(predictions)
