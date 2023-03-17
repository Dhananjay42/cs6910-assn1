import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from neuralnetwork import NeuralNetwork
import sys

def main(argv):
  opts = []
  args = []

  project_name = ''
  entity_name = ''
  dataset = "fashion_mnist"
  epochs = 10
  batch_size = 128
  loss_type = 'cross_entropy'
  optimizer = 'adam'
  lr = 0.0001
  momentum = 0.9
  beta = 0.9
  beta1 = 0.9
  beta2 = 0.99
  lamda = 0
  epsilon = 0.000001
  weight_init = 'Xavier'
  n_hl = 5
  hl_neurons = 128
  activation = 'ReLU'

  for i in range(0, len(argv)-1):
    opts.append(argv[i])
    args.append(argv[i+1])
  
  for opt, arg in zip(opts, args):
    if opt == '-wp' or opt == '--wandb_project':
      project_name = arg
    elif opt == '-we' or opt == '--wandb_entity':
      entity_name = arg
    elif opt == '-d' or opt == '--dataset':
      dataset = arg
    elif opt == '-e' or opt == '--epochs':
      n_epochs = arg
    elif opt == '-b' or opt == '--batch_size':
      batch_size = arg
    elif opt == '-l' or opt == '--loss':
      loss_type = arg
    elif opt == '-o' or opt == '--optimizer':
      optimizer = arg
    elif opt == '-lr' or opt == '--learning_rate':
      lr = arg
    elif opt == '-m' or opt == '--momentum':
      momentum = arg
    elif opt == '-beta' or opt == '--beta':
      beta = arg
    elif opt == '-beta1' or opt == '--beta1':
      beta1 = arg
    elif opt == '-beta2' or opt == '--beta2':
      beta2 = arg
    elif opt == '-w_d' or opt == '--weight_decay':
      lamda = arg
    elif opt == '-w_i' or opt == '--weight_init':
      weight_init = arg
    elif opt == '-nhl' or opt == '--num_layers':
      n_hl = arg
    elif opt == '-sz' or opt == '--hidden_size':
      hl_neurons = arg
    elif opt == '-a' or opt == '--activation':
      activation = arg

network = NeuralNetwork(optimizer = 'adam')
n_epochs = 5    


(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

trainX = trainX/255
testX = testX/255

trainX = trainX
trainy = trainy

# x_train = []
# x_test = []
# for i in range(0, len(trainX)):
#   x_train.append(trainX[i,:,:].flatten())
# for i in range(0, len(testX)):
#   x_test.append(testX[i,:,:].flatten())


# trainy_encoded = to_categorical(trainy)
# testy_encoded = to_categorical(testy)
# #trainX = trainX
# #trainy_encoded = trainy_encoded[:100]
# losses = []
# print('Training process started with '+network.optimizer+' optimizer.')
# for i in range(0, n_epochs):
#     loss = network.epoch(x_train, trainy_encoded, x_test, testy_encoded, i)
#     losses.append(loss)

if __name__ == "__main__":
   main(sys.argv[1:])