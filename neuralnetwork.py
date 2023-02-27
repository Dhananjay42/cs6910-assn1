import numpy as np

class NeuralNetwork:
    def __init__(self, n_hidden, hl_size, batch_size, weight_init, activation, lr, n_input, n_output = 10):
        self.n_hidden = n_hidden #number of hidden layers
        self.hl_size = hl_size #size of each hidden layer
        self.batch_size = batch_size 
        self.activation = activation 
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
    
    def feedforward(self, image):
        output = image.flatten()

        for weight_array in self.weights_list:
            output.append(1)



