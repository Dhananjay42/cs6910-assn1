import numpy as np

class NeuralNetwork:
    def __init__(self, n_hidden, hl_size, batch_size, weight_init, activation_fn, lr, n_input, n_output = 10):
        self.n_hidden = n_hidden #number of hidden layers
        self.hl_size = hl_size #size of each hidden layer
        self.batch_size = batch_size 
        self.activation_fn = activation_fn
        self.lr = lr
        self.weight_init = weight_init
        self.n_input = n_input
        self.n_output = n_output
        self.weights_list = []

    def init_weights(self):
        self.weights_list = []
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
            self.weights_list.append(arr)

        elif self.weight_init == 'random':
            arr = np.random.randn(self.n_input + 1, self.hl_size)
            self.weights_list.append(arr)

            for i in range(self.n_hidden - 1):
                arr = np.random.randn(self.hl_size+1, self.hl_size)
                self.weights_list.append(arr)
            
            arr = np.random.randn(self.hl_size+1, self.n_output)
            self.weights_list.append(arr)

        else:
            print('Choose correct initialization.')
            quit()
    
    def activation(self, x, grad = False):
        if self.activation_fn == 'sigmoid':
            if grad == False:
                return 1/(1+ np.exp(-x))
            else:
                return np.exp(-x)/((1 + np.exp(-x))**2)

        elif self.activation_fn == 'tanh':
            if grad == False:
                return np.tanh(x)
            else:
                1 - np.multiply(np.tanh(x), np.tanh(x))
        else:
            if grad == False:
                return x*(x > 0)
            else:
                return 1*(x >= 0)

    def softmax(output):
        return [np.exp(x)/sum(np.exp(x)) for x in output]
    
    def feedforward(self, image):
        output = image.flatten()

        for i in range(0, 2 + self.n_hidden):
            weight_array = self.weights_list[i]
            output.append(1)
            output = np.matmul(np.transpose(weight_array), output)
            if i!= len(self.weights_list[i]) - 1:
                output = self.activation(output)
            else:
                output = self.softmax(output)
    
        
            



