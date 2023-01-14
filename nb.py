# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
import math
import copy


class NaiveBayesClassifier:
    """
    See the project description for the specifications of the Naive Bayes classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.automaticTuning = False

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(
            set([f for datum in trainingData for f in list(datum.keys())]))

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels,
                          validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.
        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.
        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        # To get all the frequency counts from the training data
        # Frequency(Count) of each label
        CountOfEachLabel = util.Counter()
        # Count of Each feature with a value of 1 given a label
        countFeatureVal1 = util.Counter()
        # Count of Each feature with a given a label, irrespective of its' value
        countOfEachFeature = util.Counter()

        for x in range(len(trainingData)):
            l = trainingLabels[x]
            CountOfEachLabel[l] += 1

            image = trainingData[x]
            for coordinate, value in list(image.items()):
                countOfEachFeature[(coordinate, l)] += 1

                if value > 0:
                    countFeatureVal1[(coordinate, l)] += 1

        # base accuracy value for the validation data, setting to -infinity
        baseAccuracyValue = float('-inf')

        for k in kgrid:

            # getting the temporary copies of the frequencies for each k value
            TempCountOfEachLabel = copy.copy(CountOfEachLabel)
            TempcountFeatureVal1 = copy.copy(countFeatureVal1)
            TempcountOfEachFeature = copy.copy(countOfEachFeature)

            TempCountOfEachLabel.normalize()  # normalize

            # Smoothing: adding the value of k to both the numerator and denominator in the conditional probability
            # formula
            for l in self.legalLabels:
                for feature in self.features:
                    TempcountFeatureVal1[(feature, l)] += k
                    TempcountOfEachFeature[(feature, l)] += (k+k)

            # Calculating the Conditional Probability
            for i, counts in list(TempcountFeatureVal1.items()):
                TempcountFeatureVal1[i] = counts * \
                    1.0 / TempcountOfEachFeature[i]

            self.TempCountOfEachLabel = TempCountOfEachLabel
            self.TempcountFeatureVal1 = TempcountFeatureVal1

            # Calculating the accuracy on validation data set
            guesses = self.classify(validationData)
            cnt = 0
            for i in range(len(validationLabels)):
                if guesses[i] == validationLabels[i]:
                    cnt += 1

            if cnt > baseAccuracyValue:
                OptimalCases = (TempCountOfEachLabel, TempcountFeatureVal1, k)
                baseAccuracyValue = cnt

        self.TempCountOfEachLabel = OptimalCases[0]
        self.TempcountFeatureVal1 = OptimalCases[1]
        self.k = OptimalCases[2]

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        # Log posteriors are stored for later data analysis (autograder).
        self.posteriors = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logOfJointProb = util.Counter()

        for l in self.legalLabels:
            logOfJointProb[l] = math.log(self.TempCountOfEachLabel[l])

            for feature, count in list(datum.items()):

                if count > 0:
                    logOfJointProb[l] += math.log(
                        self.TempcountFeatureVal1[feature, l])
                else:
                    logOfJointProb[l] += math.log(1 -
                                                  self.TempcountFeatureVal1[feature, l])

        return logOfJointProb
