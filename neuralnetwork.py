import numpy as np

import cv2

class NeuralNetwork:
    def __init__(self, n_hidden = 3, hl_size = 10, batch_size = 4, weight_init = 'random', activation_fn = 'sigmoid', lr = 0.001, n_input = 28*28, n_output = 10):
        self.n_hidden = n_hidden #number of hidden layers
        self.hl_size = hl_size #size of each hidden layer
        self.batch_size = batch_size 
        self.activation_fn = activation_fn
        self.lr = lr
        self.weight_init = weight_init
        self.n_input = n_input
        self.n_output = n_output
        self.weights_list = []
        self.out_weights = []
        self.updates = []
        self.init_weights()

    def init_weights(self):
        #self.weights_list = []
        #self.out_weights = []
        if self.weight_init == 'Xavier':
            a = np.sqrt(6/(self.n_input + self.hl_size))
            arr = np.random.uniform(low = -a, high = a, size = (self.n_input + 1, self.hl_size))
            self.weights_list.append(arr)

            for i in range(self.n_hidden - 1):
                a = np.sqrt(3/self.hl_size)
                arr = np.random.uniform(low = -a, high = a, size = (self.hl_size+1, self.hl_size))
                self.weights_list.append(arr)
            
            a = np.sqrt(6/(self.hl_size + self.n_output))
            arr = np.random.uniform(low = -a, high = a, size = (self.hl_size+1, self.n_output))
            self.out_weights = arr
        
            self.updates = self.weights.copy()

        elif self.weight_init == 'random':
            arr = np.random.randn(self.n_input + 1, self.hl_size)
            self.weights_list.append(arr)

            for i in range(self.n_hidden - 1):
                arr = np.random.randn(self.hl_size+1, self.hl_size)
                self.weights_list.append(arr)
            
            arr = np.random.randn(self.hl_size+1, self.n_output)
            self.out_weights = arr

        else:
            print('Choose correct initialization.')
            quit()
    
    def activation(self, x, grad = False):
        if self.activation_fn == 'sigmoid':
            if grad == False:
                return np.array([1/(1+ np.exp(-a)) for a in x])
            else:
                return [np.exp(-a)/((1 + np.exp(-a))**2) for a in x]

        elif self.activation_fn == 'tanh':
            if grad == False:
                return [np.tanh(a) for a in x]
            else:
                return [1 - np.tanh(a)**2 for a in x]
        else:
            if grad == False:
                return x*(x > 0)
            else:
                return 1*(x >= 0)

    def softmax(self, output):
        out = [np.exp(x) for x in output]
        out = out/np.sum((out))
        return out
    
    def feedforward(self, image):
        output = image.flatten()

        for i in range(0, self.n_hidden):
            weight_array = self.weights_list[i]
            output = np.append(output, 1)
            output = np.matmul(np.transpose(weight_array), output)
            output = self.activation(output)
            
        
        output = np.append(output, 1)
        output = np.matmul(np.transpose(self.out_weights), output)
        print(output)
        output = self.softmax(output)

        return output
    
    def backprop(self, y_pred, y_true):
        out_error = y_pred - y_true
        pass


network = NeuralNetwork()
image  = cv2.imread('media_images_examples_0_0.png',0)
print(np.shape(network.weights_list[0]), np.shape(network.weights_list[1]), np.shape(network.weights_list[2]))
output = network.feedforward(image)
print(output)

    
        
            



