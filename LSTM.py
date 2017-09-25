import tensorflow as tf


class LSTM():
    """docstring."""

    def __init__(self, inputDimensionList, outputDimensionList, numLayers=1, numHidden=100, learningRate=0.01, forgetBias=1.0):
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
        # TAKE IN ACTIVATION, LOSS AND OPTIMISER FUNCTION AS ARGS

    def setBatchDict(self, inputs, labels):
        """docstring."""
        self.batchDict = {self.batchInputs: inputs, self.batchLabels: labels}
        return self.batchDict

    def batchPredictions(self):
        """docstring."""
        inputAsTensors = tf.unstack(self.batchInputs, axis=1)
        outputs, finalState = tf.nn.static_rnn(self.cell, inputAsTensors, dtype=tf.float32)
        return tf.tanh(tf.add(tf.matmul(outputs[-1], self.weightedMatrix), self.biasMatrix))

    def batchLoss(self, batchPredictions):
        """docstring."""
        return tf.losses.mean_squared_error(predictions=batchPredictions, labels=self.batchLabels)

    def backPropagation(self, batchLoss):
        """docstring."""
        tf.train.AdamOptimizer(self.learningRate).minimize(batchLoss)

    def processBatch(self):
        """docstring."""
        predictions = self.batchPredictions()
        loss = self.batchLoss(predictions)
        self.backPropagation(loss)
        return predictions, loss
