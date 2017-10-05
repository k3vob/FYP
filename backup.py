import tensorflow as tf
import DataWorker
import Constants

x = tf.placeholder(tf.float32, [None, Constants.sequenceLength, DataWorker.numFeatures])
y = tf.placeholder(tf.float32, [None, 1])
xTensors = tf.unstack(x, axis=1)   # [seqLength tensors of shape (batchSize, numFeatures)]

W = tf.Variable(tf.random_normal([Constants.numHidden, 1]))     # Weighted matrix
b = tf.Variable(tf.random_normal([1]))                          # Bias

cell = tf.contrib.rnn.BasicLSTMCell(Constants.numHidden, forget_bias=Constants.forgetBias)
outputs, finalState = tf.nn.static_rnn(cell, xTensors, dtype=tf.float32)
# predictions = [tf.add(tf.matmul(output, W), b) for output in outputs]             # List of predictions after each time step
prediction = tf.add(tf.matmul(outputs[-1], W), b)                                   # Prediction after final time step
prediction = tf.tanh(prediction)                                                    # Activation
mse = tf.losses.mean_squared_error(predictions=prediction, labels=y)                # Mean loss over entire batch
accuracy = tf.reduce_mean(1 - (tf.abs(y - prediction) / DataWorker.labelRange))     # Accuracy over entire batch
optimiser = tf.train.GradientDescentOptimizer(Constants.learningRate).minimize(mse)            # Backpropagation

lastPred = prediction[-1][0]
lastLabel = y[-1][0]

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    # #############################################
    # TRAINING
    # #############################################
    for epoch in range(Constants.numEpochs):
        print("***** EPOCH:", epoch + 1, "*****\n")
        IDPointer, TSPointer = 0, 0         # Pointers to current ID and timestamp
        epochComplete = False
        batchNum = 0
        while not epochComplete:
            batchNum += 1
            batchX, batchY, IDPointer, TSPointer, epochComplete = DataWorker.generateBatch(IDPointer, TSPointer, isTraining=True)
            dict = {x: batchX, y: batchY}
            session.run(optimiser, dict)
            if batchNum % 1000 == 0 or epochComplete:
                batchLoss = session.run(mse, dict)
                batchAccuracy = session.run(accuracy, dict)
                print("Iteration:", batchNum)
                print("Pred:", str(session.run(lastPred, dict)), "\tLabel:", str(session.run(lastLabel, dict)))
                print(batchLoss)
                print(str("%.2f" % (batchAccuracy * 100) + "%\n"))

    # #############################################
    # TESTING
    # #############################################
    testX, testY, _, _, _ = DataWorker.generateBatch(0, 0, isTraining=False)
    testAccuracy = session.run(accuracy, {x: testX, y: testY})
    print("Testing Accuracy:", str("%.2f" % (testAccuracy * 100) + "%"))

# # Sort DF by ID lifespan
# df = df.assign(freq=df.groupby('id')['id'].transform('count')).sort_values(by=['freq', 'id', 'timestamp'], ascending=[False, True, True])
#
# # Padded input & output matrices ###TOO SLOW###
# inputMatrix = np.zeros(shape=(numIDs, numTimestamps, numFeatures))
# labelMatrix = np.zeros(shape=(numIDs, numTimestamps, 1))
# for i, ID in enumerate(IDs):
#     print(i)
#     IDarray = df.loc[df['id'] == ID]
#         for j, TS in enumerate(timestamps):
#             if TS in IDarray['timestamp'].values:
#                 inputMatrix[i, j] = IDarray.loc[IDarray['timestamp'] == TS, [feature for feature in features]].as_matrix().flatten()
#                 labelMatrix[i, j] = IDarray.loc[IDarray['timestamp'] == TS, ['y']]
#
# # Generates bacthes along 1 ID at a time
# def generateBatch(IDPointer, TSPointer):
#     inputs, labels = [], []
#     newID = False               # ############## NEEDED?
#     for i in range(Constants.batchSize):
#         sequence = inputMatrix[IDPointer][TSPointer + i * Constants.sequenceLength:TSPointer + (i + 1) * Constants.sequenceLength]
#         if len(sequence) == Constants.sequenceLength:
#             inputs.append(sequence)
#             labels.append(labelMatrix[IDPointer][TSPointer + (i + 1) * Constants.sequenceLength - 1])
#         else:
#             pad = np.zeros((1, numFeatures))
#             for _ in range(Constants.sequenceLength - len(sequence)):
#                 sequence = np.concatenate((pad, sequence))
#             inputs.append(sequence)
#             labels.append(labelMatrix[IDPointer][-1])
#             IDPointer += 1
#             TSPointer = 0
#             newID = True
#             return inputs, labels, IDPointer, TSPointer, newID
#     TSPointer += Constants.batchSize * Constants.sequenceLength
#     return inputs, labels, IDPointer, TSPointer, newID
