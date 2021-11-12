import scipy.linalg
import scipy.io
import matplotlib.pyplot as plt
import numpy as np








def pinv(A):
    U, s, V_transpose = scipy.linalg.svd(A)

    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)

    m = A.shape[0]
    n = A.shape[1]

    sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        sigma[i, i] = s[i]

    sigma_inverse = np.zeros((m,n))
    for i in range(min(m, n)):
        if sigma[i, i] > 0:
            sigma_inverse[i, i] = 1 / sigma[i,i]

    A_pinv = np.matmul(V, sigma_inverse)
    A_pinv = np.matmul(A_pinv, U_transpose)

    return A_pinv


def generatePairs(inputList):
    possiblePairs = []
    

    
    for i in range(len(inputList)):
        for j in range(len(inputList)):
            first = inputList[i]
            second = inputList[j]

            if first == second:
                continue

            if [first, second] not in possiblePairs and [second, first] not in possiblePairs:
                possiblePairs.append([first, second])

    return possiblePairs

    

def sign(x):
    if x > 0:
        return 1
    else:
        return -1




def evaluateClassifier(classifier, evaluationImages, evaluationLabels):

    confusionMatrix = np.zeros((10,10))
    for i in range(len(evaluationImages)):
        groundTruth = evaluationLabels[i]
        prediction = classifier.predict(evaluationImages[i])
        confusionMatrix[groundTruth,prediction] = confusionMatrix[groundTruth,prediction] + 1


    return confusionMatrix


class LLSClassifier:
    def __init__(self, trainImages, trainLabels, targetLabel, binary = False):

        self.trainImages = np.copy(trainImages)
        self.trainLabels = np.copy(trainLabels)
        self.binary = binary


        for i in range(len(self.trainLabels)):
            if self.trainLabels[i] != targetLabel:
                self.trainLabels[i] = -1
            else:
                self.trainLabels[i] = 1

 

        A = np.column_stack((self.trainImages, np.full(len(self.trainImages), 1))) #A matrix

        A_transpose = np.transpose(A)

        A_A_transpose = np.matmul(A_transpose, A)

        A_A_tranpose_pinv = pinv(A_A_transpose)

        beta = np.matmul(A_A_tranpose_pinv, A_transpose)
        beta = np.matmul(beta, np.transpose(self.trainLabels))

        self.alpha = beta[len(beta) - 1]
        self.beta = beta[0:len(beta) - 1]
    
    def predict(self, inputImage):
        prediction = np.matmul(np.transpose(self.beta), inputImage) + self.alpha
        if self.binary == False:
            return prediction
        else:
            return sign(prediction)

class oneVersusAllClassifier:
    def __init__(self, trainImages, trainLabels):
        self.trainImages = np.copy(trainImages)
        self.trainLabels = np.copy(trainLabels)
        self.classifiers = []

        for i in range(10):
            print(i)
            classifier = LLSClassifier(self.trainImages, self.trainLabels, i)
            self.classifiers.append(classifier)
        
    def predict(self, inputImage):
        scoreArray = []
        for classifier in self.classifiers:
            predictionScore = classifier.predict(inputImage)
            scoreArray.append(predictionScore)
        
        return np.argmax(scoreArray)


class oneVerusOneClassifier:
    def __init__(self, trainImages, trainLabels, binary = False):
        self.trainImages = np.copy(trainImages)
        self.trainLabels = np.copy(trainLabels)
        self.classifiers = []

        for i in range(10):
            print(i)
            classifier = LLSClassifier(self.trainImages, self.trainLabels, i, binary)
            self.classifiers.append(classifier)

    def predict(self, inputImage):



        voteArray = np.zeros(10)
        pairs = generatePairs(np.array([0,1,2,3,4,5,6,7,8,9]))

        for pair in pairs:
            firstClassPrediction = self.classifiers[pair[0]].predict(inputImage)
            secondClassPrediction = self.classifiers[pair[1]].predict(inputImage)

            if (firstClassPrediction > secondClassPrediction):
                voteArray[pair[0]] = voteArray[pair[0]] + 1
            else:
                voteArray[pair[1]] = voteArray[pair[1]] + 1

        return np.argmax(voteArray)



def ReLU(x):
    if x > 0:
        return x
    else:
        return 0

def addRandomNoise(inputImage, noiseAmount):
    imageLength = len(inputImage)
    noiseVector = np.zeros(imageLength)

    noiseLimit = noiseAmount / np.sqrt(imageLength)

    for i in range(imageLength):
        noiseVector[i] = np.random.uniform(-noiseLimit, noiseLimit)


    noisyImage = inputImage + noiseVector

    return noisyImage


class randomFeatureMap:
    def __init__(self, featureNum, inputImageLength, nonlinearFunction):
        self.nonlinearFunction = nonlinearFunction
        self.W = np.zeros((featureNum, inputImageLength))
        for i in range(featureNum):
            for j in range(inputImageLength):
                self.W[i, j] = np.random.normal(0, 1)

        self.b = np.zeros(featureNum)
        for i in range(featureNum):
            self.b[i] = np.random.normal(0, 1)

    def map(self, inputImage):

        h = np.matmul(self.W, inputImage)
        h = h + self.b

        if self.nonlinearFunction == "identity":
            return h

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

    

mnist = scipy.io.loadmat('mnist.mat')

trainX = mnist['trainX'][1:25000] / 255 #60k train images normalized
trainY = mnist['trainY'][0][1:25000].astype('int32') #60k train labels

testX = mnist['testX'] / 255 #10k test images
testY = mnist['testY'][0].astype('int32') # 10k test labels

randomMap = randomFeatureMap(1000, len(testX[0]), "sigmoid")

remappedTrainX = []
remappedTestX = []

for i in range(len(trainX)):
    remappedTrainX.append(randomMap.map(trainX[i]))

for i in range(len(testX)):
    remappedTestX.append(randomMap.map(testX[i]))

oneVallClassifier = oneVerusOneClassifier(remappedTrainX, trainY)


confusionMatrix = evaluateClassifier(oneVallClassifier, remappedTestX, testY)

print(confusionMatrix)


fig = plt.figure
plt.imshow(confusionMatrix, cmap='gray')
plt.show()


'''


#print(sigma)
#print(sigma_inverse)



a = np.matrix('1 2; 5 10')

b = np.matrix('5 25; 25 125')

b_inv = scipy.linalg.pinv(b)

#print(b_inv)



#b_pinv = svdsolve(b)

#print(b_pinv)

#print(b_inv)



b_inv_a = np.matmul(b_inv, a)

result = np.matmul(b_inv_a, np.matrix('1; 5'))

#print(result)


ex = np.matrix('-3 1; 5 0')

print(np.matmul(ex, pinv(ex)))






mnist = scipy.io.loadmat('mnist.mat')


#for key, value in mnist.items() :
    #print(key)

#X are images
#Y are labels

trainX = mnist['trainX'] / 255 #60k train images normalized
trainY = mnist['trainY'][0].astype('int32') #60k train labels

for i in range(len(trainY)):
    if trainY[i] != 8:
        trainY[i] = -1
    else:
        trainY[i] = 1



testX = mnist['testX'] #10k test images
testY = mnist['testY'][0].astype('int32') # 10k test labels

#print(len(mnist['testX'])) #10k test images
#print(len(mnist['testY'][0])) # 10k test labels
#print(len(mnist['trainX'])) #60k train images
#print(len(mnist['trainY'][0])) #60k train labels


#print(trainX[0])


#print(trainY[1030])

#fig = plt.figure
#plt.imshow(np.reshape(trainX[1030], (28,28)), cmap='gray')
#plt.show()

A = np.column_stack((trainX, np.full(len(trainX), 1))) #A matrix

A_transpose = np.transpose(A)

A_A_transpose = np.matmul(A_transpose, A)

A_A_tranpose_pinv = pinv(A_A_transpose)

beta = np.matmul(A_A_tranpose_pinv, A_transpose)
beta = np.matmul(beta, np.transpose(trainY))
alpha = beta[len(beta) - 1]
beta = beta[0:len(beta) - 1]

for i in range(200):

    prediction = np.matmul(np.transpose(beta), testX[i]) + alpha
    print(prediction)
    print(testY[i])




'''