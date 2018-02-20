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
        self.batchSize = tf.placeholder(tf.int32, [])
        self.learningRate = tf.placeholder(tf.float32, [])
        # self.learningRate = Constants.learningRate
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

        self.gradientDescentOptimiser = tf.train.AdamOptimizer(self.learningRate)

        self.state = None
        self.batchDict = None
        self.outputs = None
        self.predictions = None
        self.loss = None
        self.accuracy = None
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
        self.outputs, self.state = tf.nn.static_rnn(
            self.network,
            self.inputsUnrolled,
            initial_state=self.state,
            dtype=tf.float32
        )
        # self.outputs = [
        #     tf.add(tf.matmul(output, self.weights), self.biases)
        #     for output in self.outputs
        # ]
        self.outputs = tf.add(tf.matmul(self.outputs[-1], self.weights), self.biases)   # #####################
        self.predictions = self.__activateOutputs()
        self.loss = self.__calculateLoss()
        self.accuracy = self.__calculateAccuracy()
        self.lossMinimiser = self.gradientDescentOptimiser.minimize(self.loss)

    def __activateOutputs(self):
        """Applies activation function to all ouputs of the network."""
        predictions = tf.nn.softmax(self.outputs)
        #predictions = tf.unstack(predictions, axis=0)                              # ######################
        return predictions

    def __calculateLoss(self):
        """Calculates batch loss between predictions and labels."""                 # ######################
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.labelsUnrolled[-1], logits=self.outputs
            )
        )
        return loss

        # loss = tf.losses.mean_squared_error(
        #     labels=self.labelsUnrolled, predictions=self.predictions
        # )
        # return tf.sqrt(loss)    # RMSE

    def __calculateAccuracy(self):
        """Calculates batch accuracy of predictions against labels."""
        correctPredictions = tf.equal(tf.argmax(self.predictions, axis=-1),
                                      tf.argmax(self.labelsUnrolled[-1], axis=-1))   # ##################
        accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))
        return accuracy

    def resetState(self):
        """Resets memory state of LSTM."""
        self.state = self.network.zero_state(self.batchSize, tf.float32)

    def setBatch(self, inputs, labels, learningRate):
        """Sets TensorFlow Session's 'feed_dict' before each batch."""
        self.batchDict = {
            self.batchSize: len(inputs),
            self.inputs: inputs,
            self.labels: labels,
            self.learningRate: learningRate
        }

    def train(self):
        """Executes full forward and backpropagation of network."""
        return self.session.run(self.lossMinimiser, self.batchDict)

    def get(self, operations):
        """Returns a tuple of the requested operations."""
        ops = []
        for op in operations:
            if op == 'labels':
                ops.append(self.labelsUnrolled)
            if op == 'predictions':
                ops.append(self.predictions)
            if op == 'loss':
                ops.append(self.loss)
            if op == 'accuracy':
                ops.append(self.accuracy)

        return self.session.run(ops, self.batchDict)

    def kill(self):
        """Ends TensorFlow Session."""
        self.session.close()
