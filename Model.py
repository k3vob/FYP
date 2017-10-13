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
        self.weights = tf.Variable(tf.random_normal([numHidden] + outputShape))
        self.bias = tf.Variable(tf.random_normal(outputShape))
        layers = [tf.contrib.rnn.BasicLSTMCell(numHidden, forget_bias=forgetBias) for _ in range(numLayers)]
        self.cell = tf.contrib.rnn.MultiRNNCell(layers)
        self.optimiser = tf.train.GradientDescentOptimizer(learningRate)
        self.forgetBias = forgetBias
        self.batchDict = None
        self.session = tf.Session()
        self.outputs = None
        self.finalStates = None
        self.predictions = None
        self.loss = None
        self.accuracy = None
        self.optimise = None
        self.__buildGraph()

    def __buildGraph(self):
        outputs, finalStates = tf.nn.static_rnn(self.cell, self.inputTensors, dtype=tf.float32)
        predictions = tf.add(tf.matmul(outputs[-1], self.weights), self.bias)
        self.predictions = tf.minimum(tf.maximum(predictions, 0), 1)
        self.loss = tf.losses.mean_squared_error(predictions=self.predictions, labels=self.labels)
        self.accuracy = tf.reduce_mean(1 - tf.abs(self.labels - self.predictions) / 1.0)
        self.optimise = self.optimiser.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def __execute(self, operation):
        return self.session.run(operation, self.batchDict)

    def setBatch(self, inputs, labels):
        self.batchDict = {self.inputs: inputs, self.labels: labels}

    def batchLabels(self):
        return self.__execute(self.labels)

    def batchPredictions(self):
        return self.__execute(self.predictions)

    def batchLoss(self):
        return self.__execute(self.loss)

    def batchAccuracy(self):
        return self.__execute(self.accuracy)

    def processBatch(self):
        self.__execute(self.optimise)

    def kill(self):
        self.session.close()
