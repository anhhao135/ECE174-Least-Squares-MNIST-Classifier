import scipy.linalg
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

#contains all functions and classes used in main.py script




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