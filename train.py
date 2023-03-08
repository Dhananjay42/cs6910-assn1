from neuralnetwork import NeuralNetwork
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

network = NeuralNetwork()
n_epochs = 100

def epoch(network, x_train, y_train):
    loss = 0
    for (x, y) in zip(x_train, y_train):
        loss = loss + network.step(x, y)
    network.optimize()
    network.reset_updates()
    return loss

(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
# label = np.array([0]*10)
# label[0] = 1
# image = cv2.imread('media_images_examples_0_0.png',0)
# x_train = [image]
# y_train = [label]

losses = []
for i in range(0, n_epochs):
    loss = epoch(network, trainX, trainy)
    print(f'Epoch {i+1} completed. The loss is {loss}.')
    losses.append(loss)

plt.plot(losses)
plt.show()