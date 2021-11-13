import scipy.linalg
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


#import from custom library

from lib import LLSClassifier, oneVersusOneClassifier, oneVersusAllClassifier #import classifier building block classe and multiclassifiers
from lib import randomFeatureMap #import feature mapper class
from lib import pinv, ReLU, sign #import math functions
from lib import generatePairs, evaluateClassifier, addRandomNoise, mapImages #import utility functions


#ECE 174 mini project 1 Fall 2021
#Hao Le A15547504



#main code



#loading mnist data into arrays

mnist = scipy.io.loadmat('mnist.mat')

#specify the length of the subsets from training and testing sets



trainExamples = 600 #reduced to increase running time for demo purposes; original is 60k
testExamples = 100 #original is 10k


trainX = mnist['trainX'][0:trainExamples] / 255 #divide by 255 to normalize pixel values
trainY = mnist['trainY'][0][0:trainExamples].astype('int32') #cast to int32 so labels can be changed to negative values

testX = mnist['testX'][0:testExamples] / 255
testY = mnist['testY'][0][0:testExamples].astype('int32')





#problem 2

print("problem 2")



OneVsOneClassifier = oneVersusOneClassifier(trainX, trainY, True) #enable sign function within the binary classifiers of 1v1 multiclassifer
OneVsAllClassifier = oneVersusAllClassifier(trainX, trainY) #instantiate one of each multiclass classifiers, then train them with clean

#evaluate the training and testing performance of the classifiers

OneVsOneTrainingConfusionMatrix, OneVsOneTrainingError = evaluateClassifier(OneVsOneClassifier, trainX, trainY, "1 v 1 training")
OneVsOneTestingConfusionMatrix, OneVsOneTestingError = evaluateClassifier(OneVsOneClassifier, testX, testY, "1 v 1 testing")

OneVsAllTrainingConfusionMatrix, OneVsAllTrainingError = evaluateClassifier(OneVsAllClassifier, trainX, trainY, "1 v all training")
OneVsAllTestingConfusionMatrix, OneVsAllTestingError = evaluateClassifier(OneVsAllClassifier, testX, testY, "1 v all testing") 






#problem 3d

print("problem 3d")



#chosen to use one vs all classifier for experiment


featureNum = 100 #original is 1000

#create instances of the four different mappers

identityRandomMapper = randomFeatureMap(featureNum, len(testX[0]), "identity")
sigmoidRandomMapper = randomFeatureMap(featureNum, len(testX[0]), "sigmoid")
sineRandomMapper = randomFeatureMap(featureNum, len(testX[0]), "sine")
reluRandomMapper = randomFeatureMap(featureNum, len(testX[0]), "relu")

randomMappers = [identityRandomMapper, sigmoidRandomMapper, sineRandomMapper, reluRandomMapper] #put mappers in iterable list

for randomMapper in randomMappers: #go through each mapper for experiment
    mappedTrainX = mapImages(trainX, randomMapper)
    mappedTestX = mapImages(testX, randomMapper) #map both train and test images to feature space

    OneVsAllClassifier = oneVersusAllClassifier(mappedTrainX, trainY) #train classifier with mapped images

    #evaluate training and testing performance of feature-trained classifier

    OneVsAllTrainingConfusionMatrix, OneVsAllTrainingError = evaluateClassifier(OneVsAllClassifier, mappedTrainX, trainY, "1 v all training with random mapping + " + randomMapper.nonlinearFunction)
    OneVsAllTestingConfusionMatrix, OneVsAllTestingError = evaluateClassifier(OneVsAllClassifier, mappedTestX, testY, "1 v all testing with random mapping + " + randomMapper.nonlinearFunction) #specify the type of non linearity in the title 












#problem 3e

print("problem 3e")

#using 1 v all classifier for this experiment

#chosen to keep identity function to allow comparision with base classifier


#create empty lists to add data points to as number of features varies

featureNum = []
trainingError = []
testingError = []

#specify range and increment of increasing number of features in experiment

minFeatureNum = 1
maxFeatureNum = 100 #original is 1000
featureNumIncrement = 50


for featureNum_ in range(minFeatureNum, maxFeatureNum, featureNumIncrement):

    print("testing " + str(featureNum_) + " number of features")   

    #create identity mapper, then map training and testing images

    reluRandomMapper = randomFeatureMap(featureNum_, len(testX[0]), "identity")
    remappedTrainX = mapImages(trainX, reluRandomMapper)
    remappedTestX = mapImages(testX, reluRandomMapper)

    #train classifier with mapped images
    
    OneVsAllClassifier = oneVersusAllClassifier(remappedTrainX, trainY)

    #evaluate classifier with mapped images

    OneVsAllTrainingConfusionMatrix, OneVsAllTrainingError = evaluateClassifier(OneVsAllClassifier, remappedTrainX, trainY, "1 v all training with random mapping + " + reluRandomMapper.nonlinearFunction, False)
    OneVsAllTestingConfusionMatrix, OneVsAllTestingError = evaluateClassifier(OneVsAllClassifier, remappedTestX, testY, "1 v all testing with random mapping +" + reluRandomMapper.nonlinearFunction, False) #False since we don't need the pop up confusion matrix window

    featureNum.append(featureNum_)
    trainingError.append(OneVsAllTrainingError)
    testingError.append(OneVsAllTestingError) #add experiment data points to the lists


#plot data

plt.plot(featureNum, testingError, label = "testing error")
plt.plot(featureNum, trainingError, label = "training error")
plt.title("Error rate vs. number of features")
plt.xlabel('Number of features')
plt.ylabel('Error rate')
plt.legend()
plt.show()








#problem 3f

print("problem 3f")

featureNum = 100 #original is 1000

randomMapper = randomFeatureMap(featureNum, len(testX[0]), "relu") #change the type of linearity for experiment by specifying in last argument

mappedTrainX = mapImages(trainX, randomMapper)
mappedTestX = mapImages(testX, randomMapper)

OneVsAllClassifier = oneVersusAllClassifier(mappedTrainX, trainY) #chosen one v all classifier; changed input from clean trainX to mappedTrainX for mapped images


#lists for data points
#in this experiment, only testing error is reported

noiseAmount = []
testingError = []

minNoiseAmount = 0 #no noise for base
maxNoiseAmount = 30
noiseAmountIncrement = 1

for noiseAmount_ in range(minNoiseAmount,maxNoiseAmount,noiseAmountIncrement): 

    noiseAmount_ = noiseAmount_ / 10 #divide by 10 for finer increments

    print("testing noise amount: " + str(noiseAmount_))

    OneVsAllTestingConfusionMatrix, OneVsAllTestingError = evaluateClassifier(OneVsAllClassifier, addRandomNoise(mappedTestX, noiseAmount_), testY, "1 v all testing on noisy images", False) #evaluate the classifier on a noisy test image

    noiseAmount.append(noiseAmount_)
    testingError.append(OneVsAllTestingError) #add data points to lists


#plot data

plt.plot(noiseAmount, testingError, label = randomMapper.nonlinearFunction) #specify non linear function in label
plt.title("Testing error rate vs. noise amount in test images")
plt.xlabel('Noise amount')
plt.ylabel('Testing error rate')
plt.legend()
plt.show()









#problem 3g

print("problem 3g")


#arbitrary selection of fixed testing data point

fixedTestImage = np.array([testX[12]])
fixedTestLabel = testY[12]


classifier = LLSClassifier(trainX, trainY, fixedTestLabel, False) #instance of binary classifier that has the fixed data label as the target label; False means the classifier outputs raw non-binary prediction instead of {-1, 1}

#vary noise experiment

noiseAmount = []
prediction = []

minNoiseAmount = 0 #no noise for base
maxNoiseAmount = 50
noiseAmountIncrement = 1

for noiseAmount_ in range(minNoiseAmount,maxNoiseAmount,noiseAmountIncrement):

    noiseAmount_ = noiseAmount_ / 10
    print("testing noise amount: " + str(noiseAmount_))

    prediction_ = classifier.predict(addRandomNoise(fixedTestImage, noiseAmount_)[0]) #run prediction on noisy test image

    #add data to lists

    noiseAmount.append(noiseAmount_)
    prediction.append(prediction_)


plt.plot(noiseAmount, prediction)
plt.title("Binary classifier raw prediction output vs. noise amount in fixed test image")
plt.xlabel('Noise amount')
plt.ylabel('Raw prediction')
plt.show()

