import tensorflow as tf

import Constants


class LSTM():

    def __init__(self,
                 numFeatures,
                 sequenceLength=Constants.sequenceLength,
                 numLayers=Constants.numLayers,
                 numPerLayer=Constants.numHidden,
                 learningRate=Constants.learningRate,
                 dropout=Constants.dropoutRate):

        self.batchSize = tf.placeholder(tf.int32, [])
        self.inputs = tf.placeholder(tf.float32, [None, sequenceLength, numFeatures])
        self.labels = tf.placeholder(tf.float32, [None, sequenceLength, 1])
        self.inputsFlat = tf.unstack(self.inputs, axis=1)
        self.labelsFlat = tf.unstack(self.labels, axis=1)
        self.lengths = tf.placeholder(tf.float32, [None])
        self.masks = self.__createMasks(sequenceLength)
        self.weights = tf.Variable(tf.random_normal([numPerLayer, 1]))
        self.bias = tf.Variable(tf.random_normal([1]))
        self.layers = self.__createStackedLSTM(numPerLayer, numLayers, dropout)
        self.optimiser = tf.train.AdamOptimizer(learningRate)
        self.state = None
        self.batchDict = None
        self.outputs = None
        self.predictions = None
        self.loss = None
        self.accuracy = None
        self.train = None
        self.lastLabels = None       # last set of sequenceLength number of labels of the batch
        self.lastPredictions = None  # last set of sequenceLength number of predictions of the batch
        self.session = tf.Session()
        self.__buildGraph()

    def __createStackedLSTM(self, numPerLayer, numLayers, dropout):
        layers = []
        for _ in range(numLayers):
            layer = tf.contrib.rnn.BasicLSTMCell(numPerLayer)
            layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=(1.0 - dropout))
            layers.append(layer)
        return tf.contrib.rnn.MultiRNNCell(layers)

    def __buildGraph(self):
        self.resetState()
        self.outputs, self.state = tf.nn.static_rnn(
            self.layers, self.inputsFlat, initial_state=self.state, dtype=tf.float32)
        self.predictions = self.__getPredictions()
        self.loss = self.__getLoss()
        self.accuracy = self.__getAccuracy()
        self.train = self.optimiser.minimize(self.loss)
        self.lastLabels = self.__getLastLabels()
        self.lastPredictions = self.__getLastPredictions()
        self.session.run(tf.global_variables_initializer())

    def resetState(self):
        self.state = self.layers.zero_state(self.batchSize, tf.float32)

    def __createMasks(self, sequenceLength):
        masks = tf.sequence_mask(self.lengths, sequenceLength, dtype=tf.float32)
        masks = tf.expand_dims(masks, axis=-1)
        return tf.transpose(masks, [1, 0, 2])

    def __getPredictions(self):
        predictions = [tf.add(tf.matmul(output, self.weights), self.bias)
                       for output in self.outputs]
        activatedPredictions = self.__activate(predictions)
        return activatedPredictions

    def __activate(self, predictions):
        return tf.nn.relu(predictions)

    def __getLoss(self):
        squaredDifferences = tf.square((self.labelsFlat - self.predictions))
        maskedSquaredDifferences = tf.multiply(squaredDifferences, self.masks)
        totalDifferencePerSequence = tf.reduce_sum(maskedSquaredDifferences, axis=0)
        totalDifferencePerSequence = tf.reshape(totalDifferencePerSequence, [-1])
        averageDifferencePerSequence = tf.divide(
            totalDifferencePerSequence, tf.maximum(self.lengths, 1))
        averageDifferenceOverBatch = tf.reduce_mean(averageDifferencePerSequence)
        return averageDifferenceOverBatch

    def __getAccuracy(self):
        pass

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

    def getLastLabels(self):
        return self.session.run(self.lastLabels, self.batchDict)

    def getLastPredictions(self):
        return self.session.run(self.lastPredictions, self.batchDict)

    def setBatchDict(self, inputs, labels, lengths):
        self.batchDict = {self.batchSize: len(inputs),
                          self.inputs: inputs,
                          self.labels: labels,
                          self.lengths: lengths}

    def getBatchLabels(self):
        return self.session.run(self.labels, self.batchDict)

    def getBatchPredictions(self):
        return self.session.run(self.predictions, self.batchDict)

    def getBatchLoss(self):
        return self.session.run(self.loss, self.batchDict)

    def getBatchAccuracy(self):
        return self.session.run(self.accuracy, self.batchDict)

    def processBatch(self):
        return self.session.run(self.train, self.batchDict)

    def kill(self):
        self.session.close()
