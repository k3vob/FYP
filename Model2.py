import tensorflow as tf

import Constants


class LSTM():

    def __init__(
            self,
            numFeatures,
            numOutputs,
            sequenceLength=Constants.sequenceLength,
            numLayers=Constants.numLayers,
            numUnits=Constants.numHidden,  # LSTM units per cell per time step
            dropout=Constants.dropoutRate
    ):

        #####################################################
        # Batch Placeholders
        #####################################################

        self.learningRate = tf.placeholder(tf.float32, [])
        self.batchSize = tf.placeholder(tf.int32, [])
        self.inputs = tf.placeholder(
            tf.float32, [None, sequenceLength, numFeatures]
        )
        self.labels = tf.placeholder(
            tf.float32, [None, sequenceLength, numOutputs]
        )
        self.inputsUnrolled = tf.unstack(self.inputs, axis=1)
        self.labelsUnrolled = tf.unstack(self.labels, axis=1)

        #####################################################
        # Weights & Biases
        #####################################################

        self.weights = tf.Variable(
            tf.random_normal([numUnits, numOutputs])
        )
        self.biases = tf.Variable(
            tf.random_normal([numOutputs])
        )

        #####################################################
        # Network of Stacked Layers
        #####################################################

        self.network = self.__buildStackedLayers(
            numUnits, numLayers, dropout
        )

        #####################################################
        # TensorFlow Graph Operations
        #####################################################

        self.gradientDescentOptimiser = tf.train.AdamOptimizer(
            self.learningRate)

        self.state = None
        self.batchDict = None
        self.predictions = None
        self.loss = None
        self.lossMinimiser = None
        self.session = tf.Session()
        self.__buildTensorFlowGraph()
        self.session.run(tf.global_variables_initializer())

    def __buildStackedLayers(self, numUnits, numLayers, dropout):
        """Stacks layers of LSTM cells, and adds dropout."""
        layers = []
        for _ in range(numLayers):
            layer = tf.contrib.rnn.BasicLSTMCell(numUnits)
            layer = tf.contrib.rnn.DropoutWrapper(
                layer, output_keep_prob=(1.0 - dropout)
            )
            layers.append(layer)
        stackedLayers = tf.contrib.rnn.MultiRNNCell(layers)
        return stackedLayers

    def __buildTensorFlowGraph(self):
        """Initialises all TensorFlow graph operations."""
        self.resetState()
        outputs, self.state = tf.nn.static_rnn(
            self.network,
            self.inputsUnrolled,
            initial_state=self.state,
            dtype=tf.float32
        )
        outputs = [
            tf.add(tf.matmul(output, self.weights), self.biases)
            for output in outputs
        ]
        self.predictions = self.__activate(outputs)
        self.loss = self.__calculateLoss()
        self.lossMinimiser = self.gradientDescentOptimiser.minimize(self.loss)

    def __activate(self, outputs):
        """Applies activation function to all ouputs of the network."""
        # predictions = tf.minimum(tf.maximum(outputs, 0), 1)
        predictions = tf.nn.relu(outputs)
        predictions = tf.unstack(predictions, axis=0)
        return predictions

    def __calculateLoss(self):
        """Calculates loss between predictions and labels."""
        loss = tf.losses.mean_squared_error(
            labels=self.labelsUnrolled, predictions=self.predictions
        )
        return loss

    def resetState(self):
        """Resets memory state of LSTM."""
        self.state = self.network.zero_state(self.batchSize, tf.float32)

    def setBatch(self, learningRate, inputs, labels):
        """Sets TensorFlow Session's 'feed_dict' before each batch."""
        self.batchDict = {
            self.learningRate: learningRate,
            self.batchSize: len(inputs),
            self.inputs: inputs,
            self.labels: labels
        }

    def getBatchLabels(self):
        """Returns unrolled list of labels of shape [batchSize, numOutputs]."""
        return self.session.run(self.labelsUnrolled, self.batchDict)

    def getBatchPredictions(self):
        """Returns unrolled list of predictions of shape [batchSize, numOutputs]."""
        return self.session.run(self.predictions, self.batchDict)

    def getBatchLoss(self):
        """Returns value of loss for batch."""
        return self.session.run(self.loss, self.batchDict)

    def train(self):
        """Executes full forward and backpropagation of network."""
        return self.session.run(self.lossMinimiser, self.batchDict)

    def kill(self):
        """Ends TensorFlow Session."""
        self.session.close()

    def test(self):
        return self.session.run(self.state, self.batchDict)
