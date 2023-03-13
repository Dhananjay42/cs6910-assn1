import numpy as np
from sklearn.metrics import accuracy_score
import cv2
import copy 

class NeuralNetwork:
    def __init__(self, n_hidden = 3, hl_size = 128, batch_size = 16, weight_init = 'Xavier', activation_fn = 'ReLU', \
                 optimizer = 'sgd', lr = 0.001, n_input = 28*28, n_output = 10, loss_type = 'cross_entropy', lamda = 0.0005, \
                 m_factor = 0.5, beta = 0.5, epsilon = 0.000001, beta1 = 0.9, beta2 = 0.99):
        self.n_hidden = n_hidden #number of hidden layers
        self.hl_size = hl_size #size of each hidden layer
        self.batch_size = batch_size 
        self.activation_fn = activation_fn
        self.lr = lr
        self.weight_init = weight_init
        self.n_input = n_input
        self.n_output = n_output
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.lamda = lamda
        self.m_factor = m_factor
        self.beta = beta
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

        self.weights = []
        self.biases = []
        
        self.a_list = []
        self.h_list = []
        self.weight_gradients = []
        self.bias_gradients = []
        self.weight_updates = []
        self.bias_updates = []
        self.loss = 0

        self.weight_history = []
        self.bias_history = []
        self.rmsprop_weight = []
        self.rmsprop_bias = []

        self.adam_weight = []
        self.adam_bias = []
        self.adam_iter = 1

        self.layer_sizes = [n_input]
        for i in range(n_hidden):
          self.layer_sizes.append(hl_size)
        self.layer_sizes.append(n_output)

        # self.updates = []
        
        # self.activated_layers = []
        # self.batch_count = 0
        self.init_weights()

    def init_weights(self):
        if self.weight_init == 'Xavier':
          for i in range(0,len(self.layer_sizes)-1):
            a = np.sqrt(6/(self.layer_sizes[i] + self.layer_sizes[i+1]))
            w_arr = np.random.uniform(low = -a, high = a, size = (self.layer_sizes[i], self.layer_sizes[i+1]))
            self.weights.append(w_arr)
            b_arr = np.zeros((self.layer_sizes[i+1], 1))
            self.biases.append(b_arr)
        
        elif self.weight_init == 'random':
          for i in range(0,len(self.layer_sizes)-1):
            w_arr = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1])
            self.weights.append(w_arr)
            b_arr = np.random.randn(self.layer_sizes[i+1], 1)
            self.biases.append(b_arr)

        else:
            print('Choose correct initialization.')
            quit()
    
    def activation(self, a):
        if self.activation_fn == 'sigmoid':
          return (1.0/(1.0+np.exp(-a)))
        elif self.activation_fn == 'tanh':
          return np.tanh(a)
        elif self.activation_fn == 'identity':
          return a
        else:
          return np.maximum(0,a)
    
    def calculate_grad(self, a):
        if self.activation_fn == 'sigmoid':
            return self.activation(a)*(1-self.activation(a))
        elif self.activation_fn == 'tanh':
            return 1.0 - self.activation(a)**2
        elif self.activation_fn == 'identity':
            return 1
        else:
            return 1*(a > 0)
    
    def softmax_grad(self, a):
      return self.softmax(a)*(1-self.softmax(a))
    
    def loss_fn(self, predicted, labels):
      error = 0
      if self.loss_type == 'cross_entropy':
        #labels are one-hot encoded.
        error = error - np.sum( np.multiply(labels , np.log(predicted)))/len(labels)
      else:
        #labels are not one-hot encoded.
        error = error + np.sum(((predicted - labels)**2) / (2 * len(labels)))
      
      for i in range(0, len(self.weights)):
        reg_error = np.sum(self.weights[i]*self.weights[i])
        error = error + (self.lamda/(2*len(labels)))*reg_error
      
      return error

    def softmax(self, output):
        out = [np.exp(x) for x in output]
        out = out/np.sum((out))
        return out
      
    def feedforward(self, x):
        self.activated_layers = []
        output = copy.deepcopy(x)
        
        self.a_list = [output]
        self.h_list = [output]

        for i in range(0, self.n_hidden):
            output = np.transpose(self.weights[i])@output.reshape(-1,1) + self.biases[i]
            self.a_list.append(output)
            output = self.activation(output)
            self.h_list.append(output)
        
        output = np.transpose(self.weights[-1])@output + self.biases[-1]
        output = self.softmax(output)
        return output
    
    def backprop(self, y_pred, y_true):
        self.weight_gradients = []
        self.bias_gradients = []

        grad_a = y_pred - y_true.reshape(-1, 1) #y_pred and y_true are one-hot encoded when its cross-entropy and just scalars when its MSE. The expression does not change.
        #dimensions: (n_out x 1)
        for i in range(1,self.n_hidden + 2):
          grad_w = np.dot(self.h_list[-i].reshape(-1,1), np.transpose(grad_a))
          grad_b = copy.deepcopy(grad_a)
          self.weight_gradients.append(grad_w)
          self.bias_gradients.append(grad_b)
          grad_h_prev = np.dot(self.weights[-i], grad_a)
          derivative = self.calculate_grad(self.a_list[-i])
          grad_a = np.multiply(grad_h_prev, derivative)

        self.weight_gradients.reverse()
        self.bias_gradients.reverse()
    
    def batchwise_gradient(self, X, y):
      self.weight_updates = []
      self.bias_updates = []
      for i in range(0, len(X)):
        y_pred = self.feedforward(X[i])
        self.backprop(y_pred, y[i])
        #self.loss = self.loss + self.loss_fn(y_pred, y[i].reshape(-1, 1))
        if i==0:
          self.weight_updates = copy.deepcopy(self.weight_gradients)
          self.bias_updates = copy.deepcopy(self.bias_gradients)
        else:
          for i in range(0, len(self.weight_gradients)):
            self.weight_updates[i] = self.weight_updates[i] + self.weight_gradients[i]
            self.bias_updates[i] = self.bias_updates[i] + self.bias_gradients[i]
      
    def step(self, X, y):
      self.batchwise_gradient(X, y)
      reg_factor = (1 - ((self.lr*self.lamda)/self.batch_size))
      if self.optimizer == 'sgd':
        for i in range(0, len(self.weights)):
          self.weights[i] = reg_factor*self.weights[i] - self.lr*self.weight_updates[i]
          self.biases[i] = self.biases[i] - self.lr*self.bias_updates[i]
      
      elif self.optimizer == 'momentum':
        if len(self.weight_history) == 0:
          self.weight_history = copy.deepcopy(self.weight_updates)
          self.bias_history = copy.deepcopy(self.bias_updates)
        else:
          for i in range(0, len(self.weight_history)):
            self.weight_history[i] = self.m_factor*self.weight_history[i] + self.weight_updates[i]
            self.bias_history[i] = self.m_factor*self.bias_history[i] + self.bias_updates[i]
        
        for i in range(0, len(self.weights)):
          self.weights[i] = reg_factor*self.weights[i] - self.lr*self.weight_history[i]
          self.biases[i] = self.biases[i] - self.lr*self.bias_history[i]

      elif self.optimizer == 'nesterov':
        temp_weights = copy.deepcopy(self.weights)
        temp_biases = copy.deepcopy(self.biases)
        
        if len(self.weight_history) == 0:
          self.weight_history = [self.lr*weight for weight in self.weight_updates]
          self.bias_history = [self.lr*bias for bias in self.bias_updates]
        else:
          for i in range(0, len(self.weight_history)):
            self.weights[i] = self.weights[i] - self.m_factor*self.weight_history[i]
            self.biases[i] = self.biases[i] - self.m_factor*self.bias_history[i]

          self.batchwise_gradient(X, y)

          for i in range(0, len(self.weight_history)):
            self.weight_history[i] = self.m_factor*self.weight_history[i] + self.lr*self.weight_updates[i]
            self.bias_history[i] = self.m_factor*self.bias_history[i] + self.lr*self.bias_updates[i]
        
          self.weights = copy.deepcopy(temp_weights)
          self.biases = copy.deepcopy(temp_biases)
        
        for i in range(0, len(self.weights)):
          self.weights[i] = reg_factor*self.weights[i] - self.weight_history[i]
          self.biases[i] = self.biases[i] - self.bias_history[i]
      
      elif self.optimizer == 'rmsprop':
        if len(self.rmsprop_weight) == 0:
          self.rmsprop_weight = [grad**2 for grad in self.weight_updates]
          self.rmsprop_bias = [grad**2 for grad in self.bias_updates]
        else:
          for i in range(0, len(self.rmsprop_weight)):
            self.rmsprop_weight[i] = self.beta*self.rmsprop_weight[i] + (1-self.beta)*self.weight_updates[i]**2
            self.rmsprop_bias[i] = self.beta*self.rmsprop_bias[i] + (1-self.beta)*self.bias_updates[i]**2
        
        for i in range(0, len(self.weights)):
          self.weights[i] = reg_factor*self.weights[i] - self.lr*(self.weight_updates[i]/(np.sqrt(self.rmsprop_weight[i] + self.epsilon)))
          self.biases[i] = self.biases[i] - self.lr*(self.bias_updates[i]/(np.sqrt(self.rmsprop_bias[i] + self.epsilon)))
      
      elif self.optimizer == 'adam':
        if len(self.weight_history) == 0:
          self.weight_history = [(1-self.beta1)*grad for grad in self.weight_updates]
          self.bias_history = [(1-self.beta1)*grad for grad in self.bias_updates]
          self.adam_weight = [(1-self.beta2)*grad**2 for grad in self.weight_updates]
          self.adam_bias = [(1-self.beta2)*grad**2 for grad in self.bias_updates]
        else:
          for i in range(0, len(self.adam_weight)):
            self.weight_history[i] = self.beta1*self.weight_history[i] + (1 - self.beta1)*self.weight_updates[i]
            self.bias_history[i] = self.beta1*self.bias_history[i] + (1 - self.beta1)*self.bias_updates[i]
            self.adam_weight[i] = self.beta2*self.adam_weight[i] + (1 - self.beta2)*self.weight_updates[i]**2
            self.adam_bias[i] =  self.beta2*self.adam_bias[i] + (1 - self.beta2)*self.bias_updates[i]**2
        
        self.adam_grad_weight = [grad/(1-((self.beta1)**self.adam_iter)) for grad in self.weight_history]
        self.adam_grad_bias = [grad/(1-((self.beta1)**self.adam_iter)) for grad in self.bias_history]

        self.adam_lr_weight = [update/(1-((self.beta2)**self.adam_iter)) for update in self.adam_weight]
        self.adam_lr_bias = [update/(1-((self.beta2)**self.adam_iter)) for update in self.adam_bias]

        for i in range(0, len(self.weights)):
          self.weights[i] = reg_factor*self.weights[i] - self.lr*(self.adam_grad_weight[i]/(np.sqrt(self.adam_lr_weight[i] + self.epsilon)))
          self.biases[i] = self.biases[i] - self.lr*(self.adam_grad_bias[i]/(np.sqrt(self.adam_lr_bias[i] + self.epsilon)))
        
        self.adam_iter = self.adam_iter + 1
      
      elif self.optimizer == 'nadam':
        if len(self.weight_history) == 0:
          self.weight_history = [(1-self.beta1)*grad for grad in self.weight_updates]
          self.bias_history = [(1-self.beta1)*grad for grad in self.bias_updates]
          self.adam_weight = [(1-self.beta2)*grad**2 for grad in self.weight_updates]
          self.adam_bias = [(1-self.beta2)*grad**2 for grad in self.bias_updates]
        else:
          for i in range(0, len(self.adam_weight)):
            self.weight_history[i] = self.beta1*self.weight_history[i] + (1 - self.beta1)*self.weight_updates[i]
            self.bias_history[i] = self.beta1*self.bias_history[i] + (1 - self.beta1)*self.bias_updates[i]
            self.adam_weight[i] = self.beta2*self.adam_weight[i] + (1 - self.beta2)*self.weight_updates[i]**2
            self.adam_bias[i] =  self.beta2*self.adam_bias[i] + (1 - self.beta2)*self.bias_updates[i]**2
        
        self.adam_grad_weight = [grad/(1-((self.beta1)**self.adam_iter)) for grad in self.weight_history]
        self.adam_grad_bias = [grad/(1-((self.beta1)**self.adam_iter)) for grad in self.bias_history]

        self.adam_lr_weight = [update/(1-((self.beta2)**self.adam_iter)) for update in self.adam_weight]
        self.adam_lr_bias = [update/(1-((self.beta2)**self.adam_iter)) for update in self.adam_bias]

        factor = (1 - self.beta1)/(1 - (self.beta1**self.adam_iter))

        for i in range(0, len(self.weights)):
          self.weights[i] = reg_factor*self.weights[i] - self.lr*((self.beta1*self.adam_grad_weight[i] + (factor*self.weight_updates[i]))/(np.sqrt(self.adam_lr_weight[i] + self.epsilon)))
          self.biases[i] = self.biases[i] - self.lr*((self.beta1*self.adam_grad_bias[i] + (factor*self.bias_updates[i]))/(np.sqrt(self.adam_lr_bias[i] + self.epsilon)))
        
        self.adam_iter = self.adam_iter + 1
      
      else:
        print('Error in optimizer name.')
      
    def reset(self):
      self.loss = 0
      self.weight_history = []
      self.bias_history = []
      self.rmsprop_weight = []
      self.rmsprop_bias = []
      self.adam_iter = 1
      self.adam_weight = []
      self.adam_bias = []

    
    def inference(self, x_test, y_test):
      predictions = []
      loss = 0
      y_label = copy.deepcopy(y_test)

      for (x,y) in zip(x_test,y_test):
        pred = self.feedforward(x)
        #loss = loss + self.loss_fn(pred, y)
        predictions.append(pred)
      
      if self.loss_type == 'cross_entropy':
        y_pred = [np.argmax(predictions[i]) for i in range(0,len(predictions))]
        y_label = [np.argmax(y_test[i]) for i in range(0,len(y_test))]
      
      else:
        y_pred = np.round(y_pred)
        y_pred[y_pred<0] = 0
        y_pred[y_pred>=self.n_output-1] = self.n_output - 1
      
      acc = accuracy_score(y_label, y_pred)
        
      return loss, acc
    
    def epoch(self, x_train, y_train, x_test, y_test, index):
      for i in range(0,len(x_train),self.batch_size):
        if i+self.batch_size < len(x_train):
          xbatch = x_train[i:i+self.batch_size]
          ybatch = y_train[i:i+self.batch_size]
        else:
          xbatch = x_train[i:]
          ybatch = y_train[i:]

        self.step(xbatch, ybatch)
        
  
      test_loss, test_acc = self.inference(x_test, y_test)
      loss_ret = self.loss
      print(f'Epoch {index+1} completed. Training Loss is {self.loss}. The test loss is {test_loss} and the test accuracy is {test_acc}.')
      self.reset()
      return loss_ret