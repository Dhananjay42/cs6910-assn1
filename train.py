from neuralnetwork import NeuralNetwork
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

network = NeuralNetwork()
n_epochs = 10

def epoch(network, x_train, y_train):
    loss = 0
    for (x, y) in zip(x_train, y_train):
        loss = loss + network.step(x, y)
    network.optimize()
    network.reset_updates()
    return loss

(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
trainy_encoded = to_categorical(trainy)
testy_encoded = to_categorical(testy)
trainX = trainX[:100]
trainy_encoded = trainy_encoded[:100]
losses = []
print('Training process started.')
for i in range(0, n_epochs):
    loss = epoch(network, trainX, trainy_encoded)
    test_loss, test_acc = network.inference(testX, testy_encoded)
    train_loss, train_acc = network.inference(trainX, trainy_encoded)
    print(f'Epoch {i+1} completed. The train loss is {loss} and train acc is {train_acc}. The test loss is {test_loss}, and the test accuracy is {test_acc}.')
    losses.append(loss)

plt.plot(losses)
plt.show()

