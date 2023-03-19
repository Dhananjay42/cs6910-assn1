import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist
from keras.utils import to_categorical
from neuralnetwork import NeuralNetwork
from sklearn.model_selection import train_test_split
import sys

def main(argv):
  opts = []
  args = []

  params = {}

  project_name = ''
  entity_name = ''
  dataset = 'fashion_mnist'
  n_epochs = 10
  
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
      params['batch_size'] = arg
    elif opt == '-l' or opt == '--loss':
      params['loss_type'] = arg
    elif opt == '-o' or opt == '--optimizer':
      params['optimizer'] = arg
    elif opt == '-lr' or opt == '--learning_rate':
      params['lr'] = arg
    elif opt == '-m' or opt == '--momentum':
      params['m_factor'] = arg
    elif opt == '-beta' or opt == '--beta':
      params['beta'] = arg
    elif opt == '-beta1' or opt == '--beta1':
      params['beta1'] = arg
    elif opt == '-beta2' or opt == '--beta2':
      params['beta2'] = arg
    elif opt == '-w_d' or opt == '--weight_decay':
      params['lamda'] = arg
    elif opt == '-w_i' or opt == '--weight_init':
      params['weight_init'] = arg
    elif opt == '-nhl' or opt == '--num_layers':
      params['n_hidden'] = arg
    elif opt == '-sz' or opt == '--hidden_size':
      params['hl_size'] = arg
    elif opt == '-a' or opt == '--activation':
      params['activation_fn'] = arg
    else:
      print('Follow the format to run the script.')
      sys.exit()

  network = NeuralNetwork(**params)

  if dataset == 'fashion_mnist':
    (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
  
  elif dataset == 'mnist':
    (trainX, trainy), (testX, testy) = mnist.load_data()
  
  else:
    print('Wrong Dataset.')
    sys.exit()
  
  trainX = trainX/255
  testX = testX/255

  trainX, valX, trainy, valy = train_test_split(trainX, trainy, test_size=0.1, random_state=40)

  x_train = []
  x_test = []
  x_val = []
  for i in range(0, len(trainX)):
    x_train.append(trainX[i,:,:].flatten())
  for i in range(0, len(testX)):
    x_test.append(testX[i,:,:].flatten())
  for i in range(0, len(valX)):
    x_val.append(valX[i,:,:].flatten())

  y_train = to_categorical(trainy)
  y_test = to_categorical(testy)
  y_val = to_categorical(valy)

  print('Training process started with '+network.optimizer+' optimizer.')

  for i in range(0, n_epochs):
    train_loss,train_acc,test_acc,test_loss = network.epoch(x_train, y_train, x_val, y_val, i)
  
  test_accuracy = network.inference(x_test, y_test, loss_flag = False)
  print(f'The test accuracy is {test_accuracy.}')

        
if __name__ == "__main__":
   main(sys.argv[1:])