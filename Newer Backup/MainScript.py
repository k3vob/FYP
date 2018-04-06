import matplotlib.pyplot as plt
import tensorflow as tf

import Constants
import DataWorker as dw

inputs = tf.placeholder(
    tf.float32, [None, Constants.sequenceLength, dw.numFeatures]
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

loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=labelsUnrolled, logits=outputs)
)
predictions = tf.round(tf.nn.sigmoid(outputs))
predictions = tf.unstack(predictions, axis=0)
correctPredictions = tf.equal(predictions, labelsUnrolled[-1])
accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

# outputs = tf.nn.sigmoid(outputs)
# outputs = tf.unstack(outputs, axis=0)
# loss = tf.sqrt(tf.losses.mean_squared_error(labels=labelsUnrolled, predictions=outputs))
# predictions = tf.round(outputs)
# predictions = tf.unstack(predictions, axis=0)
# correctPredictions = tf.equal(predictions, labelsUnrolled)
# accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

train = tf.train.AdamOptimizer(Constants.initialLearningRate).minimize(loss)


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    #################################
    # TRAINING
    #################################

    for epoch in range(Constants.numEpochs):
        print("***** EPOCH:", epoch + 1, "*****\n")
        tickerPointer = -1
        count = 1
        while tickerPointer != 0:
            tickerPointer = max(tickerPointer, 0)
            sliceLosses = []
            sliceAccuracies = []
            dayPointer = -1
            while dayPointer != 0:
                dayPointer = max(dayPointer, 0)
                x, y, tickerPointer, dayPointer = dw.getBatch(tickerPointer, dayPointer)
                feed_dict = {inputs: x, labels: y}
                session.run(train, feed_dict)
                sliceLosses.append(session.run(loss, feed_dict))
                sliceAccuracies.append(session.run(accuracy, feed_dict))
            sliceLoss = sum(sliceLosses) / len(sliceLosses)
            sliceAccuracy = sum(sliceAccuracies) / len(sliceAccuracies)
            print(count, "/", dw.numSlices)
            print("Loss:\t\t", sliceLoss)
            print("Accuracy:\t", "%.2f" % (sliceAccuracy * 100) + "%")
            print("")
            count += 1

    #################################
    # TESTING
    #################################

    prices = dw.df.loc[dw.tickers[-1]]['adj_close']
    dates = prices.index

    plt.plot(dates, prices)

    tickerPointer = len(dw.tickers) - 1
    dayPointer = -1
    while dayPointer != 0:
        dayPointer = max(dayPointer, 0)
        x, y, _, dayPointer = dw.getBatch(tickerPointer, dayPointer, False)
        feed_dict = {inputs: x, labels: y}
        prediction = predictions[-1][-1]
        if prediction == 0.0:
            plt.scatter(dates[dayPointer], prices[dayPointer], c='r')
        else:
            plt.scatter(dates[dayPointer], prices[dayPointer], c='g')

    plt.show()
