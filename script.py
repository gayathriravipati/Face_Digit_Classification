import util, perceptron, nb, samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print("===================================")
          print("Mistake on example %d" % i) 
          print("Predicted %d; truth is %d" % (prediction, truth))
          print("Image: ")
          print(rawTestData[i])
          break

def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

ITERATIONS = 1
DATASET = 'digits' # 'digits' or 'faces'
TRAIN_PERCENT = 10
TEST_PERCENT = 10
CLASSIFIER = 'Perceptron' # 'Perceptron' or 'NaiveBayes'


classifiers = {'Perceptron': perceptron.PerceptronClassifier,
               'NaiveBayes': nb.NaiveBayesClassifier}

features = {'digits': basicFeatureExtractorDigit, 'faces': basicFeatureExtractorFace}

DATUM_WIDTH = DIGIT_DATUM_WIDTH if DATASET == 'digits' else FACE_DATUM_WIDTH
DATUM_HEIGHT = DIGIT_DATUM_HEIGHT if DATASET == 'digits' else FACE_DATUM_HEIGHT
legalLabels = list(range(10)) if DATASET == 'digits' else list(range(2))
numTraining = 5000 if DATASET == 'digits' else 451
numTest = 1000 if DATASET == 'digits' else 150
getFeatures = features[DATASET]

DATUM_WIDTH

if(DATASET=="faces"):
    rawTrainingData = samples.loadDataFile("data/facedata/facedatatrain", numTraining,DATUM_WIDTH,DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("data/facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("data/facedata/facedatatrain", numTest,DATUM_WIDTH,DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("data/facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("data/facedata/facedatatest", numTest,DATUM_WIDTH,DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("data/facedata/facedatatestlabels", numTest)
else:
    rawTrainingData = samples.loadDataFile("data/digitdata/trainingimages", numTraining,DATUM_WIDTH,DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("data/digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("data/digitdata/validationimages", numTest,DATUM_WIDTH,DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("data/digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("data/digitdata/testimages", numTest,DATUM_WIDTH,DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("data/digitdata/testlabels", numTest)

printImage = util.ImagePrinter(DATUM_WIDTH, DATUM_HEIGHT).printImage

printImage

trainingData = list(map(getFeatures, rawTrainingData))
validationData = list(map(getFeatures, rawValidationData))
testData = list(map(getFeatures, rawTestData))
print(len(trainingData))
classifier = classifiers[CLASSIFIER](legalLabels, max_iterations=ITERATIONS)

import time
st = time.time()
classifier.train(trainingData, trainingLabels, validationData, validationLabels)
print("Training time: %0.3fs" % (time.time() - st))
guesses = classifier.classify(testData)

correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)

print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))

analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)




