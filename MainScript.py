import tensorflow as tf

import Constants
import DataWorker

batchSize = tf.placeholder(tf.float32, [])
x = tf.placeholder(tf.float32, [None, Constants.sequenceLength, DataWorker.numFeatures])
y = tf.placeholder(tf.float32, [None, Constants.sequenceLength, 1])
xTensors = tf.unstack(x, axis=1)   # [seqLength tensors of shape (batchSize, numFeatures)]
yTensors = tf.unstack(y, axis=1)   # seqLen, ?, 1

lengths = tf.placeholder(tf.float32, [None])
masks = tf.cast(tf.cast(tf.range(Constants.sequenceLength), tf.float32) < tf.reshape(lengths, (-1, 1)), tf.float32)
masks = tf.expand_dims(masks, axis=2)   # ?, seqLen -> ?, seqLen, 1
masks = tf.transpose(masks, [1, 0, 2])  # seqLen, ?, 1

W = tf.Variable(tf.random_normal([Constants.numHidden, 1]))     # Weighted matrix
b = tf.Variable(tf.random_normal([1]))                          # Bias

layers = [tf.contrib.rnn.BasicLSTMCell(
    Constants.numHidden, forget_bias=Constants.forgetBias, state_is_tuple=True) for _ in range(Constants.numLayers)]
cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

outputs, state = tf.nn.static_rnn(cell, xTensors, dtype=tf.float32)
predictions = [tf.add(tf.matmul(output, W), b) for output in outputs]
predictions = tf.minimum(tf.maximum(predictions, 0), 1)     # seqLen, ?, 1

loss = tf.square((yTensors - predictions))                  # seqLen, ?, 1
loss = tf.multiply(loss, masks)                             # seqLen, ?, 1
loss = tf.reduce_sum(loss, axis=0)                          # ?, 1
loss = tf.reshape(loss, [-1])                               # ?, 1 -> ?
# ?, 1 / ?, 1 = ?, 1 (Avoids NaNs from div 0)
loss = tf.divide(loss, tf.maximum(lengths, 1))
loss = tf.reduce_mean(loss)                                 # average over all batches

accuracy = tf.abs(yTensors - predictions)               # seqLen, ?, 1
accuracy = tf.multiply(accuracy, masks)                 # seqLen, ?, 1
accuracy = tf.reduce_sum(accuracy, axis=0)              # ?, 1 -> sum of accuracies per batch
accuracy = tf.reshape(accuracy, [-1])                   # ?, 1 -> ?
accuracy = tf.divide(accuracy, tf.maximum(lengths, 1))  # ?, 1 -> average accuracy per batch
accuracy = tf.reduce_mean(accuracy)                     # average over all batches
accuracy = 1 - accuracy

optimiser = tf.train.GradientDescentOptimizer(
    Constants.learningRate).minimize(loss)     # Backpropagation

lastPredictions = predictions
lastPredictions = tf.transpose(lastPredictions, [1, 0, 2])[-1]
lastPredictions = tf.reshape(lastPredictions, [-1])
lastPredictions = lastPredictions[:tf.cast(lengths[-1], tf.int32)]

lastLabels = y[-1]
lastLabels = tf.reshape(lastLabels, [-1])
lastLabels = lastLabels[:tf.cast(lengths[-1], tf.int32)]

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    IDPointer, TSPointer = 0, 0
    for i in range(1):
        batchSz, batchX, batchY, batchLengths, _, IDPointer, TSPointer, epochComplete = DataWorker.generateBatch(
            IDPointer, IDPointer)
        dict = {batchSize: batchSz, x: batchX, y: batchY, lengths: batchLengths}
        session.run(optimiser, dict)
        print(i + 1)
        print(session.run(lastLabels, dict))
        print(session.run(lastPredictions, dict))

    # # #############################################
    # # TRAINING
    # # #############################################
    # for epoch in range(Constants.numEpochs):
    #     print("***** EPOCH:", epoch + 1, "*****\n")
    #     IDPointer, TSPointer = 0, 0
    #     epochComplete = False
    #     batchNum = 0
    #     while not epochComplete:
    #         batchNum += 1
    #         batchSz, batchX, batchY, batchLengths, _, IDPointer, TSPointer, epochComplete = DataWorker.generateBatch(IDPointer, TSPointer)
    #         dict = {batchSize: batchSz, x: batchX, y: batchY, lengths: batchLengths}
    #         session.run(optimiser, dict)
    #         if batchNum % Constants.printStep == 0 or epochComplete:
    #             batchLoss = session.run(loss, dict)
    #             batchAccuracy = session.run(accuracy, dict)
    #             print("Iteration:", batchNum)
    #             # print("Label:\t ", session.run(y[-1][0], dict))
    #             # print("Pred:\t ", session.run(predictions[-1][0], dict))
    #             print("Loss:\t ", batchLoss)
    #             print("Accuracy:", str("%.2f" % (batchAccuracy * 100) + "%\n"))
    #
    # # #############################################
    # # TESTING
    # # #############################################
    # testSize, testX, testY, testLengths, _ = DataWorker.generateTestBatch()
    # testAccuracy = session.run(accuracy, {batchSize: testSize, x: testX, y: testY})
    # print("Testing Accuracy:", str("%.2f" % (testAccuracy * 100) + "%"))
