import tensorflow as tf
import DataWorker
import Constants
# import SQLServer

x = tf.placeholder(tf.float32, [None, Constants.sequenceLength, DataWorker.numFeatures])
y = tf.placeholder(tf.float32, [None, 1])
xTensors = tf.unstack(x, axis=1)   # [seqLength tensors of shape (batchSize, numFeatures)]

W = tf.Variable(tf.random_normal([Constants.numHidden, 1]))   # Weighted matrix
b = tf.Variable(tf.random_normal([1]))              # Bias

cell = tf.contrib.rnn.BasicLSTMCell(Constants.numHidden, forget_bias=Constants.forgetBias)
outputs, finalState = tf.nn.static_rnn(cell, xTensors, dtype=tf.float32)
# predictions = [tf.add(tf.matmul(output, W), b) for output in outputs]     # List of predictions after each time step
prediction = tf.add(tf.matmul(outputs[-1], W), b)                           # Prediction after final time step
prediction = tf.tanh(prediction)                                            # Activation
mse = tf.losses.mean_squared_error(predictions=prediction, labels=y)        # Mean loss over entire batch
optimiser = tf.train.AdamOptimizer(Constants.learningRate).minimize(mse)    # Backpropagation

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    IDPointer, TSPointer = 0, 0         # Pointers to current ID and timestamp
    epochComplete = False
    batchNum = 0
    while not epochComplete:
        batchNum += 1
        batchX, batchY, IDPointer, TSPointer, epochComplete = DataWorker.generateBatch(IDPointer, TSPointer)
        dict = {x: batchX, y: batchY}
        session.run(optimiser, dict)
        if (batchNum) % 1000 == 0 or epochComplete:
            batchLoss = session.run(mse, dict)
            print(str(batchNum), ": ", str(batchLoss))
