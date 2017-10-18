import tensorflow as tf
import Constants


class LSTM():

    def __init__(self,
                 inputShape,
                 outputShape,
                 numLayers=Constants.numLayers,
                 numHidden=Constants.numHidden,
                 learningRate=Constants.learningRate,
                 forgetBias=Constants.forgetBias):
        self.inputs = tf.placeholder(tf.float32, [None] + inputShape)
        self.labels = tf.placeholder(tf.float32, [None] + outputShape)
        self.inputTensors = tf.unstack(self.inputs, axis=1)
        self.labelTensors = tf.unstack(self.labels, axis=1)
        self.lengths = tf.placeholder(tf.float32, [None])
        self.masks = tf.cast(tf.cast(tf.range(Constants.sequenceLength), tf.float32) < tf.reshape(self.lengths, [-1, 1]), tf.float32)
        self.masks = tf.expand_dims(self.masks, axis=2)
        self.masks = tf.transpose(self.masks, [1, 0, 2])
        self.weights = tf.Variable(tf.random_normal([numHidden] + [outputShape[-1]]))
        self.bias = tf.Variable(tf.random_normal([outputShape[-1]]))
        layers = [tf.contrib.rnn.LSTMCell(numHidden, forget_bias=forgetBias, state_is_tuple=True) for _ in range(numLayers)]
        self.cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)
        self.optimiser = tf.train.GradientDescentOptimizer(learningRate)
        self.batchDict = None
        self.outputs = None
        self.finalStates = None
        self.predictions = None
        self.loss = None
        self.accuracy = None
        self.optimise = None
        self.session = tf.Session()
        self.__buildGraph()

    def __buildGraph(self):
        self.outputs, self.finalStates = tf.nn.static_rnn(self.cell, self.inputTensors, dtype=tf.float32)
        self.predictions = self.__getPredictions(self.outputs)
        self.loss = self.__getLoss()
        self.accuracy = self.__getAccuracy()
        self.optimise = self.optimiser.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

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
        averageDifferencePerSequence = tf.divide(totalDifferencePerSequence, tf.maximum(self.lengths, 1))
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

    def setBatchDict(self, inputs, labels, lengths):
        self.batchDict = {self.inputs: inputs, self.labels: labels, self.lengths: lengths}

    def getBatchLabels(self):
        return self.session.run(self.labels, self.batchDict)

    def getBatchPredictions(self):
        return self.session.run(self.predictions, self.batchDict)

    def getBatchLoss(self):
        return self.session.run(self.loss, self.batchDict)

    def getBatchAccuracy(self):
        return self.session.run(self.accuracy, self.batchDict)

    def processBatch(self):
        return self.session.run(self.optimise, self.batchDict)

    def kill(self):
        self.session.close()
