from neuralnetwork import NeuralNetwork
import numpy as np
import cv2

network = NeuralNetwork()
n_epochs = 100

label = np.array([0]*10)
label[0] = 1
image = cv2.imread('media_images_examples_0_0.png',0)

for i in range(0, n_epochs):
    network.step(image, label)
    print(network.feedforward(image))