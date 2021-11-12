import scipy.linalg
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


#ECE 174 mini project 1 Fall 2021
#Hao Le A15547504



#utility functions


def pinv(A): #find pseudo-inverse of input matrix A
    U, s, V_transpose = scipy.linalg.svd(A) #use scipy SVD function to decompose A; s is a 1D array of singular values, NOT sigma

    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)

    m = A.shape[0]
    n = A.shape[1]

    sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        sigma[i, i] = s[i] #reconstruct sigma matrix using given singular values

    sigma_inverse = np.zeros((n,m)) #inverse of sigma is the reciprical of its elements; it is "trying its best" to get an identity matrix when multiplied with sigma
    for i in range(min(m, n)):
        if sigma[i, i] > 0: #check for non zero to avoid divide by zero error
            sigma_inverse[i, i] = 1 / sigma[i,i]

    A_pinv = np.matmul(V, sigma_inverse)
    A_pinv = np.matmul(A_pinv, U_transpose) #pseudo inverse of A is the inverse of its SVD, which is V * Sigma^-1 * U^T

    return A_pinv

def ReLU(x): #implementation of rectified linear unit function
    if x > 0:
        return x
    else:
        return 0
    
def sign(x): #sign function returns 1 if input is nonzero, else returns -1
    if x > 0:
        return 1
    else:
        return -1


def generatePairs(inputList): #given list of unique numbers, generate unique pairs where order does not matter. essentially n (number of elements in list) choose 2

    possiblePairs = [] #tracker for all pairs

    for i in range(len(inputList)):
        for j in range(len(inputList)):
            first = inputList[i]
            second = inputList[j]

            if first == second: #check if numbers are equal, if so do not make them a pair
                continue

            if [first, second] not in possiblePairs and [second, first] not in possiblePairs: #check if pair, ordered both ways, exist in pair tracker. if not, add pair (only first ordering)
                possiblePairs.append([first, second])

    return possiblePairs

def evaluateClassifier(classifier, evaluationImages, evaluationLabels, evaluationName, displayConfusionMatrix = True):

    #run classifier on input evaluation images. Checks against ground truth evaluation labels.
    #calculates error rate, and generates confusion matrix.
    #optional display of confusion matrix in pop up window; must close for program to continue.

    confusionMatrix = np.zeros((10,10)) #10 classes gives a 10 x 10 confusion matrix

    errorCount = 0

    for i in range(len(evaluationImages)):
        groundTruth = evaluationLabels[i] #get corresponding label
        prediction = classifier.predict(evaluationImages[i]) #pass in image to classifer, get out class prediction
        confusionMatrix[groundTruth,prediction] = confusionMatrix[groundTruth,prediction] + 1 #increment corresponding element in confusion matrix based on label (y axis) and prediction (x axis); if label equals prediction this corresponds to the diagonal of the confusion matrix, where there are true positives
        if groundTruth != prediction:
            errorCount = errorCount + 1 #if the prediction does not match the label, this is a false positive, an error; plot this on the confusion matrix, which lies outside of the diagonal
        

    errorRate = errorCount / len(evaluationImages) #error rate is error count divided by number of total examples seen by classifier which is number of input evaluation images

    if displayConfusionMatrix: #check if confusion matrix visualization is wanted
        fig = plt.figure
        plt.title(str(evaluationName + " confusion matrix"))
        plt.suptitle("Error rate: " + str(errorRate))
        plt.xlabel('prediction')
        plt.ylabel('ground truth')
        plt.imshow(confusionMatrix, cmap='viridis')
        plt.show()

    return confusionMatrix, errorRate

def addRandomNoise(inputImages, noiseAmount):

    #add noise to image vector by adding randomized vector of equivalent dimension
    #norm of noise vector will never be greater than specified noise amount

    imageLength = len(inputImages[0])
    noiseVector = np.zeros(imageLength)

    noiseLimit = noiseAmount / np.sqrt(imageLength)  #this ensures noise amount specified is not reach by the noise vector norm. 

    for i in range(imageLength):
        noiseVector[i] = np.random.uniform(-noiseLimit, noiseLimit) #generate noise vector elements randomly through a uniform distribution with limits.

    noisyImages = []

    for inputImage in inputImages: #iterate through input images and add noise vector to them
        noisyImage = inputImage + noiseVector
        noisyImages.append(noisyImage)

    return noisyImages

def mapImages(inputImages, mapper): #utility function to map a collection of images to a collection of features, ready for training/testing

    mappedInputImages = []
    for i in range(len(inputImages)):
        mappedInputImages.append(mapper.map(inputImages[i]))

    return mappedInputImages


#building block class for LLS binary classifier

class LLSClassifier:

    #base class for LLS classifier
    #class constructor takes in training images and labels, and solves normal equation to get weights that minimize linear regression error of mapping all training images to their labels, stored in alpha and beta
    #targetLabel is the class that is cared about; ideally we want the classifier to output 1 for that class, and -1 for everything else
    #binary option toggles whether or not the output of the classifier prediction is converted to {-1, 1} or left as the raw output

    def __init__(self, trainImages, trainLabels, targetLabel, binary = False):

        self.trainImages = np.copy(trainImages)
        self.trainLabels = np.copy(trainLabels) #copy train arrays to avoid passing them in by reference - messing around with labels later will not affect original label array
        self.binary = binary


        for i in range(len(self.trainLabels)): #convert labels so 1 for class we care about, targetLabel, and -1 for everything else
            if self.trainLabels[i] != targetLabel:
                self.trainLabels[i] = -1
            else:
                self.trainLabels[i] = 1

 

        A = np.column_stack((self.trainImages, np.full(len(self.trainImages), 1))) #add a column of 1's to the image data matrix ie every image vector now has a 1 appended to the end

        #normal equation: A^T * y = A^T * A * B     y - groundtruth labels | A - train image row vectors + 1 to end, stacked on top of each other | B - weights vector
        #to solve for weights B, B = (A^T * A)^-1 *  A^T * y
        #since A^T * A is not always invertible, we use pseudoinverse

        A_transpose = np.transpose(A)

        A_A_transpose = np.matmul(A_transpose, A)

        A_A_tranpose_pinv = pinv(A_A_transpose) 

        beta = np.matmul(A_A_tranpose_pinv, A_transpose)
        beta = np.matmul(beta, np.transpose(self.trainLabels))

        self.alpha = beta[len(beta) - 1]
        self.beta = beta[0:len(beta) - 1] #beta has alpha at the end that is constant offset
    
    def predict(self, inputImage):
        prediction = np.matmul(np.transpose(self.beta), inputImage) + self.alpha #with the calculated weights, apply them to an image vector to map it to a label
        if self.binary == False: #toggle whether or not output prediction is {1, -1} or raw output
            return prediction
        else:
            return sign(prediction)


#different implementations of multi-class classifiers that uses the LLS binary classifier

class oneVersusAllClassifier:
    #predict based on the outputs of 10 different LLS classifiers, all trained to care about different classes from 0-9
    #we want the classifiers to output a raw prediction so we can judge the class based on the highest number, essentially the closest to 1 is more confident

    def __init__(self, trainImages, trainLabels):
        self.trainImages = np.copy(trainImages)
        self.trainLabels = np.copy(trainLabels)
        self.classifiers = []

        for i in range(10): #create a list of binary classifiers for 0-9
            print("training binary classifier for: " + str(i))
            classifier = LLSClassifier(self.trainImages, self.trainLabels, i)
            self.classifiers.append(classifier)
        
    def predict(self, inputImage):
        scoreArray = []
        for classifier in self.classifiers:
            predictionScore = classifier.predict(inputImage)
            scoreArray.append(predictionScore) #pass in image to each classifier and tally the scores
        
        return np.argmax(scoreArray) #find the index of the highest score in the tally, since the scores were taken in order of the classes from 0-9. the index is the predicted class

class oneVersusOneClassifier:

    #classify based on pairwise comparison of classifiers
    #higher prediction of the pair gets the vote to its corresponding target class
    #ideally the groundtruth class should get 9 votes, the highest amount, since it will be judge against the other 9 classes in which it should all win

    def __init__(self, trainImages, trainLabels, binary = False):
        self.trainImages = np.copy(trainImages)
        self.trainLabels = np.copy(trainLabels)

        self.pairs = generatePairs(np.array([0,1,2,3,4,5,6,7,8,9])) #10 choose 2 pairs

        self.pairsClassifiers = []

        for pair in self.pairs: #iterate through all possible pairs and train the relevant classifier

            print("training classifier for pair " + str(pair))


            #data with only the two classes in the pair

            relevantTrainImages = []
            relevantTrainLabels = []
            

            for i in range(len(self.trainLabels)): #filter out training set to only include images/labels with either class in pair
                if self.trainLabels[i] == pair[0] or self.trainLabels[i] == pair[1]:
                    relevantTrainImages.append(self.trainImages[i])
                    relevantTrainLabels.append(self.trainLabels[i])

            relevantClassifier = LLSClassifier(relevantTrainImages, relevantTrainLabels, pair[0], binary = True) #train relevant binary classifier for pair; the target label is the 1st in the pair, so output will be 1 if 1st class is detected, else -1

            self.pairsClassifiers.append(relevantClassifier) #add classifier to the classifier list with matching order to the generated pairs list


    def predict(self, inputImage):

        voteArray = np.zeros(10) #tally to keep track of votes; index is representative of the label class

        for i in range(len(self.pairs)): #iterate through pairs to get 45 votes in total

            prediction = self.pairsClassifiers[i].predict(inputImage)

            if (prediction == 1): #check if prediction is 1, if so then the first class gets the vote
                voteArray[self.pairs[i][0]] = voteArray[self.pairs[i][0]] + 1 #increment tally based on index of class label
            else:
                voteArray[self.pairs[i][1]] = voteArray[self.pairs[i][1]] + 1

        return np.argmax(voteArray) #prediction is whichever class gets most votes; ties are settled randomly in numpy's argmax function




#random feature mapper that stores its own mapping matrix, offset vector, and non-linearity function
#has utility function for mapping images to features

class randomFeatureMap:

    #class for generating a random mapping from image pixel space to a feature space
    #mapping is done by multiplying image vector with random matrix, then adding a random vector, then applying a nonlinearity

    def __init__(self, featureNum, inputImageLength, nonlinearFunction): #specifiy how many features does the mapping take image vectors to and the nonlinear function
        self.nonlinearFunction = nonlinearFunction
        self.W = np.zeros((featureNum, inputImageLength)) #randomly generate the matrix that would resize the image vector to the number of features
        for i in range(featureNum):
            for j in range(inputImageLength):
                self.W[i, j] = np.random.normal(0, 1) #elements are generated through gaussian distribution

        self.b = np.zeros(featureNum) #generate gaussian random vector to offset the feature vector
        for i in range(featureNum):
            self.b[i] = np.random.normal(0, 1)

    def map(self, inputImage): #apply mapping to image vector to bring into feature space

        h = np.matmul(self.W, inputImage)
        h = h + self.b #h is the output feature vector before non linearity

        if self.nonlinearFunction == "identity":
            return h #identity is no linearity, so return h as is
        

        #for non linearity functions, go through each element of the feature vector, then apply the non linearity to the element

        if self.nonlinearFunction == "sigmoid": 

            for i in range(h.shape[0]):
                h[i] = 1 / (1 + np.exp(h[i]))

            return h

        if self.nonlinearFunction == "sine":

            for i in range(h.shape[0]):
                h[i] = np.sin(h[i])

            return h

        if self.nonlinearFunction == "relu":

            for i in range(h.shape[0]):
                h[i] = ReLU(h[i])

            return h



#main code



#loading mnist data into arrays

mnist = scipy.io.loadmat('mnist.mat')

#specify the length of the subsets from training and testing sets

trainExamples = 60000
testExamples = 10000


trainX = mnist['trainX'][0:trainExamples] / 255 #divide by 255 to normalize pixel values
trainY = mnist['trainY'][0][0:trainExamples].astype('int32') #cast to int32 so labels can be changed to negative values

testX = mnist['testX'][0:testExamples] / 255
testY = mnist['testY'][0][0:testExamples].astype('int32')



'''

#problem 2



OneVsOneClassifier = oneVersusOneClassifier(trainX, trainY, True) #enable sign function within the binary classifiers of 1v1 multiclassifer
OneVsAllClassifier = oneVersusAllClassifier(trainX, trainY) #instantiate one of each multiclass classifiers, then train them with clean

#evaluate the training and testing performance of the classifiers

OneVsOneTrainingConfusionMatrix, OneVsOneTrainingError = evaluateClassifier(OneVsOneClassifier, trainX, trainY, "1 v 1 training")
OneVsOneTestingConfusionMatrix, OneVsOneTestingError = evaluateClassifier(OneVsOneClassifier, testX, testY, "1 v 1 testing")

OneVsAllTrainingConfusionMatrix, OneVsAllTrainingError = evaluateClassifier(OneVsAllClassifier, trainX, trainY, "1 v all training")
OneVsAllTestingConfusionMatrix, OneVsAllTestingError = evaluateClassifier(OneVsAllClassifier, testX, testY, "1 v all testing") 


'''


'''


#problem 3d



#chosen to use one vs all classifier for experiment


featureNum = 1000

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





'''



'''


#problem 3e

#using 1 v all classifier for this experiment

#chosen to keep identity function to allow comparision with base classifier


#create empty lists to add data points to as number of features varies

featureNum = []
trainingError = []
testingError = []

#specify range and increment of increasing number of features in experiment

minFeatureNum = 1
maxFeatureNum = 1000
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


'''


'''


#problem 3f

featureNum = 200

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




'''



'''




#problem 3g


#arbitrary selection of fixed testing data point

fixedTestImage = np.array([testX[123]])
fixedTestLabel = testY[123]


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

'''