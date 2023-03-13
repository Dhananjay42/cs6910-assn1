import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from neuralnetwork import NeuralNetwork

network = NeuralNetwork(optimizer = 'adam')
n_epochs = 5    


(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

trainX = trainX/255
testX = testX/255

trainX = trainX
trainy = trainy

x_train = []
x_test = []
for i in range(0, len(trainX)):
  x_train.append(trainX[i,:,:].flatten())
for i in range(0, len(testX)):
  x_test.append(testX[i,:,:].flatten())


trainy_encoded = to_categorical(trainy)
testy_encoded = to_categorical(testy)
#trainX = trainX
#trainy_encoded = trainy_encoded[:100]
losses = []
print('Training process started.')
for i in range(0, n_epochs):
    loss = network.epoch(x_train, trainy_encoded, x_test, testy_encoded, i)
    losses.append(loss)