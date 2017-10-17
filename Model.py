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
        self.weights = tf.Variable(tf.random_normal([numHidden] + [outputShape[-1]]))
        self.bias = tf.Variable(tf.random_normal([outputShape[-1]]))
        layers = [tf.contrib.rnn.LSTMCell(numHidden, forget_bias=forgetBias, state_is_tuple=True) for _ in range(numLayers)]
        self.cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)
        self.optimiser = tf.train.GradientDescentOptimizer(learningRate)
        self.forgetBias = forgetBias
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
        outputs, finalStates = tf.nn.static_rnn(self.cell, self.inputTensors, dtype=tf.float32)
        predictions = [tf.add(tf.matmul(output, self.weights), self.bias) for output in outputs]
        self.predictions = tf.minimum(tf.maximum(predictions, 0), 1)
        self.loss = tf.losses.mean_squared_error(predictions=self.predictions, labels=self.labelTensors)
        self.accuracy = tf.reduce_mean(1 - tf.abs(self.labelTensors - self.predictions) / 1.0)
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
