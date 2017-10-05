import tensorflow as tf
import Constants
import DataWorker       # Remove this dependency


class LSTM():
    """docstring."""

    def __init__(self,
                 inputDimensionList,
                 outputDimensionList,
                 numLayers=Constants.numLayers,
                 numHidden=Constants.numHidden,
                 learningRate=Constants.learningRate,
                 forgetBias=Constants.forgetBias
                 ):
        """docstring."""
        self.batchInputs = tf.placeholder(tf.float32, [None] + inputDimensionList)
        self.batchLabels = tf.placeholder(tf.float32, [None] + outputDimensionList)
        self.weightedMatrix = tf.Variable(tf.random_normal([numHidden] + outputDimensionList))
        self.biasMatrix = tf.Variable(tf.random_normal(outputDimensionList))
        self.cell = tf.contrib.rnn.BasicLSTMCell(numHidden, forget_bias=forgetBias)
        self.numLayers = numLayers
        self.numHidden = numHidden
        self.learningRate = learningRate
        self.forgetBias = forgetBias
        self.batchDict = {}
        self.batchInputTensors = None
        self.batchOutputs = None    # All needed as instance variables?
        self.batchFinalStates = None
        self.batchPredictions = None
        self.batchLoss = None
        self.batchAccuracy = None
        self.initialised = False
        self.session = tf.Session()
        # Take in activation, loss and optimiser FUNCTIONS as args

    def execute(self, command):
        """docstring."""
        return self.session.run(command, self.batchDict)

    def setBatchDict(self, inputs, labels):
        """docstring."""
        self.batchDict = {self.batchInputs: inputs, self.batchLabels: labels}
        self.batchInputTensors = tf.unstack(self.batchInputs, axis=1)

    # def predict(self):
    #     """docstring."""
    #     self.batchOutputs, self.batchFinalState = tf.nn.static_rnn(self.cell, self.batchInputTensors, dtype=tf.float32)
    #     if not self.initialised:
    #         self.session.run(tf.global_variables_initializer())
    #         self.initialised = True
    #     self.batchPredictions = self.execute(tf.tanh(tf.add(tf.matmul(self.batchOutputs[-1], self.weightedMatrix), self.biasMatrix)))
    #     return self.batchPredictions
    #
    # def getLoss(self):
    #     """docstring."""
    #     self.batchLoss = self.execute(tf.losses.mean_squared_error(predictions=self.batchPredictions, labels=self.batchLabels))
    #     return self.batchLoss
    #
    # def getAccuracy(self):
    #     """docstring."""
    #     self.batchAccuracy = self.execute(tf.reduce_mean(1 - (tf.abs(self.batchLabels - self.batchPredictions) / DataWorker.labelRange)))
    #     return self.batchAccuracy
    #
    # def optimise(self):
    #     """docstring."""
    #     self.execute(tf.train.AdamOptimizer(self.learningRate).minimize(self.batchLoss))
    #
    # def processBatch(self):
    #     """docstring."""
    #     pred = self.predict()
    #     loss = self.getLoss()
    #     acc = self.getAccuracy()
    #     self.optimise()
    #     return pred, loss, acc

    def runBatch(self):
        """docstring."""
        self.batchOutputs, self.batchFinalState = tf.nn.static_rnn(self.cell, self.batchInputTensors, dtype=tf.float32)
        pred = tf.tanh(tf.add(tf.matmul(self.batchOutputs[-1], self.weightedMatrix), self.biasMatrix))
        mse = tf.losses.mean_squared_error(predictions=pred, labels=self.batchLabels)
        #optimiser = tf.train.GradientDescentOptimizer(self.learningRate).minimize(mse)
        optimiser = tf.train.AdamOptimizer(self.learningRate).minimize(mse)

        if not self.initialised:
            self.session.run(tf.global_variables_initializer())
            self.initialised = True

        self.execute(optimiser)
        self.batchPredictions = self.execute(pred)
        self.batchLoss = self.execute(tf.losses.mean_squared_error(predictions=self.batchPredictions, labels=self.batchLabels))
        self.batchAccuracy = self.execute(tf.reduce_mean(1 - (tf.abs(self.batchLabels - self.batchPredictions) / DataWorker.labelRange)))
        labels = self.execute(self.batchLabels)
        return self.batchPredictions, labels, self.batchLoss, self.batchAccuracy

    def kill(self):
        """docstring."""
        self.session.close()
