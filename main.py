import scipy.linalg
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
mnist = scipy.io.loadmat('mnist.mat')


#for key, value in mnist.items() :
    #print(key)

#X are images
#Y are labels

trainX = mnist['trainX'] #60k train images
trainY = mnist['trainY'][0].astype('int32') #60k train labels

for i in range(len(trainY)):
    if trainY[i] != 5:
        trainY[i] = -1
    else:
        trainY[i] = 1

print(trainY)



testX = mnist['testX'] #10k test images
testY = mnist['testY'][0].astype('int32') # 10k test labels

#print(len(mnist['testX'])) #10k test images
#print(len(mnist['testY'][0])) # 10k test labels
#print(len(mnist['trainX'])) #60k train images
#print(len(mnist['trainY'][0])) #60k train labels

flatLength = len(trainX[0])
print(flatLength)

#print(trainX[0])


#print(trainY[1030])

#fig = plt.figure
#plt.imshow(np.reshape(trainX[1030], (28,28)), cmap='gray')
#plt.show()

new = np.column_stack((trainX, np.full(len(trainX), 1)))

basis_new = scipy.linalg.orth(new)

print(len(basis_new[0]))

beta = np.matmul(np.transpose(basis_new), np.transpose(trainY))

betaX = np.matmul(basis_new, beta)

print(len(betaX))


