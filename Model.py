import os

import tensorflow as tf


class LSTM():
    """OOP implementation of an LSTM RNN using TensorFlow."""

    def __init__(
        self,
        numFeatures,        # Number of input features
        numOutputs,         # Number of output predictions
        sequenceLength,     # Number of unrolled time steps
        unitsPerLayer,      # List of number of LSTM units per stacked LSTM cell
        stateful=False,     # Stateful or stateless LSTM
        manyToOne=True,     # Many-To-One or Many-To-Many network
        regularise=False    # Add L2 regularisation to loss calculation
    ):

        #####################################################
        # Boolean Flags
        #####################################################

        self.stateful = stateful        # Pass state from batch to batch, or reset after each batch
        self.manyToOne = manyToOne      # Use outputs at all time steps, or just last
        self.regularise = regularise    # Add L2 regularisation to LSTM weights or not

        #####################################################
        # Batch Placeholders
        #####################################################
        # Variables to be set per batch

        self.batchSize = tf.placeholder(tf.int32, [])
        self.learningRate = tf.placeholder(tf.float32, [])
        self.dropoutRate = tf.placeholder(tf.float32, [])

        # Intuitive shape for inputs and outputs to be fed in as
        # (batchSize, sequenceLength, numFeatures)
        self.rawInputs = tf.placeholder(
            tf.float32, [None, sequenceLength, numFeatures]
        )
        # (batchSize, sequenceLength, numOutputs)
        self.rawLabels = tf.placeholder(
            tf.float32, [None, sequenceLength, numOutputs]
        )

        # Reshaped for better management using TensorFlow
        # [sequenceLength * (batchSize, numFeatures)]
        self.inputs = tf.unstack(self.rawInputs, axis=1)
        # [sequenceLength * (batchSize, numOutputs)]
        self.labels = tf.unstack(self.rawLabels, axis=1)

        # If Many-To-One network, only last output is required
        # (batchSize, numOutputs)
        if self.manyToOne:
            self.labels = self.labels[-1]

        #####################################################
        # Dense Output Layer Weights & Biases
        #####################################################

        # Weights in fully connected final output layer
        # (numUnitsInFinalHiddenLayer, numOutputs)
        self.denseOutputLayerWeights = tf.Variable(
            tf.random_normal([unitsPerLayer[-1], numOutputs])
        )

        # Biases in fully connected final output layer
        # (numOutputs)
        self.denseOutputLayerWBiases = tf.Variable(
            tf.random_normal([numOutputs])
        )

        #####################################################
        # Network of Stacked LSTM Layers
        #####################################################

        self.network = self.__buildStackedLayers(
            unitsPerLayer, self.dropoutRate
        )

        #####################################################
        # TensorFlow Graph Operations
        #####################################################

        self.session = tf.Session()                             # TensorFlow Session for dynamic runtime
        self.SGD = tf.train.AdamOptimizer(self.learningRate)    # Stochastic Gradient Descent algorithm
        self.__buildTensorFlowGraph()                           # Builds static graph prior to dynamic runtime
        self.saver = tf.train.Saver()                           # Used to save/restore saved trained models
        self.session.run(tf.global_variables_initializer())     # Initialises all variables prior to dynamic runtime

    #########################################################
    # Private Methods
    #########################################################

    def __buildStackedLayers(self, unitsPerLayer, dropoutRate):
        """Stacks several LSTM layers, and adds dropout to each."""
        layers = []
        for i in range(len(unitsPerLayer)):
            cell = tf.contrib.rnn.BasicLSTMCell(unitsPerLayer[i])
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=(1.0 - dropoutRate)
            )
            layers.append(cell)
        stackedLayers = tf.contrib.rnn.MultiRNNCell(layers)
        return stackedLayers

    def __buildTensorFlowGraph(self):
        """Defines all the TensorFlow operations to process a batch of data."""
        # Initialise state to zero_state
        self.resetState()

        # Produces output at final LSTM cell, prior to dense output layer
        # [sequenceLength * (batchSize, numCellsInFinalHiddenLayer)]
        LSTMOutputs, self.state = tf.nn.static_rnn(
            self.network,
            self.inputs,
            initial_state=self.state,
            dtype=tf.float32
        )

        # Processes LSTM outputs throguh final dense output layer
        # (batchSize, numOutputs) OR [sequenceLength * (batchSize, numOutputs)]
        self.outputs = self.__feedThroughDenseOutputLayer(LSTMOutputs)

        # Activate outputs to obtain final predictions
        # Stores their rounded values to compare with labels
        # (batchSize, numOutputs) OR [sequenceLength * (batchSize, numOutputs)]
        self.predictions = self.__activateOutputs()
        self.roundedPredictions = tf.round(self.predictions)

        # Calculates loss value of batch
        self.loss = self.__calculateLoss()
        # Adds L2 regularisation to loss
        if self.regularise:
            self.loss += self.__regularisation()
        # Performs backpropagation
        self.lossMinimiser = self.SGD.minimize(self.loss)

        # Calculates accuracy value
        self.accuracy = self.__calculateAccuracy()

    def __feedThroughDenseOutputLayer(self, LSTMOutputs):
        """Dense fully-connected layer between final LSTM layer and network outputs."""
        if self.manyToOne:
            # (batchSize, numUnitsInFinalHiddenLayer)
            finalSequenceOutput = LSTMOutputs[-1]
            # (batchSize, numOutputs)
            finalSequenceOutput = tf.matmul(finalSequenceOutput, self.denseOutputLayerWeights) + self.denseOutputLayerWBiases
            return finalSequenceOutput
        else:
            # [sequenceLength * (batchSize, numOutputs)]
            sequenceOutputs = [
                tf.matmul(output, self.denseOutputLayerWeights) + self.denseOutputLayerWBiases
                for output in LSTMOutputs
            ]
            return sequenceOutputs

    def __activateOutputs(self):
        """Applies activation function to output neurons."""
        activations = tf.nn.sigmoid(self.outputs)
        if self.manyToOne:
            # (batchSize, numOutputs)
            return activations
        else:
            # [sequenceLength * (batchSize, numOutputs)]
            return tf.unstack(activations)

    def __calculateLoss(self):
        """Calculates the loss between the final activated output and the labels."""
        mse = tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions)
        return mse

    def __regularisation(self):
        """Adds L2 regularisation on LSTM weights to loss."""
        L2 = 0.0005 * sum(
            tf.nn.l2_loss(weight)
            for weight in tf.trainable_variables()
            if not ("noreg" in weight.name or "Bias" in weight.name)
        )
        return L2

    def __calculateAccuracy(self):
        """Calculates percentage of correct predictions in the batch."""
        correctPredictions = tf.equal(self.labels, self.roundedPredictions)
        accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))
        return accuracy

    #########################################################
    # Public Methods
    #########################################################

    def resetState(self):
        """Resets the hidden state to zero-state."""
        self.state = self.network.zero_state(self.batchSize, tf.float32)

    def setBatch(self, inputs, labels, learningRate=0.0, dropoutRate=0.0):
        """Data to be used for current batch."""
        self.batchDict = {
            self.batchSize: len(inputs),
            self.rawInputs: inputs,
            self.rawLabels: labels,
            self.learningRate: learningRate,
            self.dropoutRate: dropoutRate
        }

        # Reset state each batch if stateless LSTM
        if not self.stateful:
            self.resetState()

    def train(self):
        """Performs entire forward feed through and backpropagation."""
        return self.session.run(self.lossMinimiser, self.batchDict)

    def get(self, operations):
        """Returns results of inputted list of operations."""
        ops = []
        for op in operations:
            if op == 'labels':
                ops.append(self.labels)
            if op == 'outputs':
                ops.append(self.outputs)
            if op == 'predictions':
                ops.append(self.predictions)
            if op == 'roundedPredictions':
                ops.append(self.roundedPredictions)
            if op == 'loss':
                ops.append(self.loss)
            if op == 'accuracy':
                ops.append(self.accuracy)
        return self.session.run(ops, self.batchDict)

    def save(self, modelName="LSTM"):
        """Saves entire model."""
        modelName += '.ckpt'
        dir = os.path.dirname(os.path.realpath(__file__)) + '/SavedModels/'
        self.saver.save(self.session, dir + modelName)

    def restore(self, modelName="LSTM"):
        """Restores previously saved  model."""
        modelName += '.ckpt'
        dir = os.path.dirname(os.path.realpath(__file__)) + '/SavedModels/'
        self.saver.restore(self.session, dir + modelName)

    def kill(self):
        """Ends TensorFlow Session."""
        self.session.close()
