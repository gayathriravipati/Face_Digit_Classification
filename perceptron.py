import util
PRINT = True


class Perceptron:
    def __init__(self, labels, max_iterations):
        self.labels = labels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in labels:
            # this is the data-structure you should use
            self.weights[label] = util.Counter()

    def setWeights(self, weights):
        assert len(weights) == len(self.labels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.
        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        iter = 0
        while iter < self.max_iterations:
            for x in range(0, len(trainingData)):
                i, j = trainingLabels[x], self.classify([trainingData[x]])[0]
                
                if i != j:
                    self.weights[i] += trainingData[x]
                    
                    self.weights[j] -= trainingData[x]

            iter += 1

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        predictions = []

        for datum in data:
            array = util.Counter()

            for i in self.labels:
                array[i] = self.weights[i] * datum

            predictions.append(array.argMax())

        return predictions
