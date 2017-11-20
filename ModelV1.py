import tensorflow as tf

import Constants


class LSTM():

    def __init__(self,
                 inputShape,
                 outputShape,
                 numLayers=Constants.numLayers,
                 numHidden=Constants.numHidden,
                 learningRate=Constants.learningRate):
        self.batchSize = tf.placeholder(tf.int32, [])
        self.inputs = tf.placeholder(tf.float32, [None] + inputShape)
        self.labels = tf.placeholder(tf.float32, [None] + outputShape)
        self.lengths = tf.placeholder(tf.float32, [None])
        self.inputTensors = tf.unstack(self.inputs, axis=1)
        self.labelTensors = tf.unstack(self.labels, axis=1)
        self.weights = tf.Variable(tf.random_normal([numHidden] + [outputShape[-1]]))
        self.bias = tf.Variable(tf.random_normal([outputShape[-1]]))
        self.masks = self.__createMasks()
        self.layers = self.__createStackedLSTM(numHidden, numLayers)
        self.optimiser = tf.train.AdamOptimizer(learningRate)
        self.batchDict = None
        self.outputs = None
        self.state = None
        self.predictions = None
        self.loss = None
        self.accuracy = None
        self.optimise = None
        self.lastLabels = None       # last set of sequenceLength number of labels of the batch
        self.lastPredictions = None  # last set of sequenceLength number of predictions of the batch
        self.session = tf.Session()
        self.resetState()
        self.__buildGraph()

    def __createStackedLSTM(self, numHidden, numLayers):
        layers = []
        for _ in range(numLayers):
            layer = tf.contrib.rnn.BasicLSTMCell(numHidden)
            layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=1.0 - Constants.dropout)
            layers.append(layer)
        return tf.contrib.rnn.MultiRNNCell(layers)

    def __buildGraph(self):
        self.outputs, self.state = tf.nn.static_rnn(
            self.layers, self.inputTensors, initial_state=self.state, dtype=tf.float32)
        self.predictions = self.__getPredictions(self.outputs)
        self.loss = self.__getLoss()
        self.accuracy = self.__getAccuracy()
        self.optimise = self.optimiser.minimize(self.loss)
        self.lastLabels = self.__getLastLabels()
        self.lastPredictions = self.__getLastPredictions()
        self.session.run(tf.global_variables_initializer())

    def resetState(self):
        self.state = self.layers.zero_state(self.batchSize, tf.float32)

    def __createMasks(self):
        masks = tf.cast(tf.cast(tf.range(Constants.sequenceLength), tf.float32) < tf.reshape(self.lengths, [-1, 1]), tf.float32)
        masks = tf.expand_dims(masks, axis=2)
        return tf.transpose(masks, [1, 0, 2])

    def __getPredictions(self, outputs):
        predictions = [tf.add(tf.matmul(output, self.weights), self.bias) for output in outputs]
        activatedPredictions = self.__activate(predictions)
        return activatedPredictions

    def __activate(self, predictions):
        return tf.minimum(tf.maximum(predictions, 0), 1)

    def __getLoss(self):
        squaredDifferences = tf.square((self.labelTensors - self.predictions))
        maskedSquaredDifferences = tf.multiply(squaredDifferences, self.masks)
        totalDifferencePerSequence = tf.reduce_sum(maskedSquaredDifferences, axis=0)
        totalDifferencePerSequence = tf.reshape(totalDifferencePerSequence, [-1])
        averageDifferencePerSequence = tf.divide(
            totalDifferencePerSequence, tf.maximum(self.lengths, 1))
        averageDifferenceOverBatch = tf.reduce_mean(averageDifferencePerSequence)
        return averageDifferenceOverBatch

    def __getAccuracy(self):
        errors = tf.abs(self.labelTensors - self.predictions)
        maskedErrors = tf.multiply(errors, self.masks)
        totalErrorPerSequence = tf.reduce_sum(maskedErrors, axis=0)
        totalErrorPerSequence = tf.reshape(totalErrorPerSequence, [-1])
        averageErrorPerSequence = tf.divide(totalErrorPerSequence, tf.maximum(self.lengths, 1))
        averageErrorOverBatch = tf.reduce_mean(averageErrorPerSequence)
        percentageAccuracy = (1 - averageErrorOverBatch) * 100
        return percentageAccuracy

    def __getLastLabels(self):
        lastLabels = tf.identity(self.labels[-1])
        lastLabels = tf.reshape(lastLabels, [-1])
        lastLabels = lastLabels[:tf.cast(self.lengths[-1], tf.int32)]
        return lastLabels

    def __getLastPredictions(self):
        lastPredictions = tf.identity(self.predictions)
        lastPredictions = tf.transpose(lastPredictions, [1, 0, 2])[-1]
        lastPredictions = tf.reshape(lastPredictions, [-1])
        lastPredictions = lastPredictions[:tf.cast(self.lengths[-1], tf.int32)]
        return lastPredictions

    def setBatchDict(self, batchSize, inputs, labels, lengths):
        self.batchDict = {self.batchSize: batchSize, self.inputs: inputs,
                          self.labels: labels, self.lengths: lengths}

    def getBatchLabels(self):
        return self.session.run(self.labels, self.batchDict)

    def getBatchPredictions(self):
        return self.session.run(self.predictions, self.batchDict)

    def getBatchLoss(self):
        return self.session.run(self.loss, self.batchDict)

    def getBatchAccuracy(self):
        return self.session.run(self.accuracy, self.batchDict)

    def getLastLabels(self):
        return self.session.run(self.lastLabels, self.batchDict)

    def getLastPredictions(self):
        return self.session.run(self.lastPredictions, self.batchDict)

    def processBatch(self):
        return self.session.run(self.optimise, self.batchDict)

    def kill(self):
        self.session.close()
