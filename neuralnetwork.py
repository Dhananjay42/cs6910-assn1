import numpy as np

import cv2

class NeuralNetwork:
    def __init__(self, n_hidden = 3, hl_size = 2, batch_size = 4, weight_init = 'random', activation_fn = 'sigmoid', lr = 0.001, n_input = 2*2, n_output = 4):
        self.n_hidden = n_hidden #number of hidden layers
        self.hl_size = hl_size #size of each hidden layer
        self.batch_size = batch_size 
        self.activation_fn = activation_fn
        self.lr = lr
        self.weight_init = weight_init
        self.n_input = n_input
        self.n_output = n_output
        self.weights_list = []
        self.updates = []
        self.layer_outputs = []
        self.init_weights()

    def init_weights(self):
        if self.weight_init == 'Xavier':
            a = np.sqrt(6/(self.n_input + self.hl_size))
            arr = np.random.uniform(low = -a, high = a, size = (self.n_input + 1, self.hl_size))
            self.weights_list.append(arr)
            self.updates.append(np.zeros(np.shape(arr)))

            for i in range(self.n_hidden - 1):
                a = np.sqrt(3/self.hl_size)
                arr = np.random.uniform(low = -a, high = a, size = (self.hl_size+1, self.hl_size))
                self.weights_list.append(arr)
                self.updates.append(np.zeros(np.shape(arr)))
            
            a = np.sqrt(6/(self.hl_size + self.n_output))
            arr = np.random.uniform(low = -a, high = a, size = (self.hl_size+1, self.n_output))
            self.weights_list.append(arr)
            self.updates.append(np.zeros(np.shape(arr)))
        
        elif self.weight_init == 'random':
            arr = np.random.randn(self.n_input + 1, self.hl_size)
            self.weights_list.append(arr)
            self.updates.append(np.zeros(np.shape(arr)))

            for i in range(self.n_hidden - 1):
                arr = np.random.randn(self.hl_size+1, self.hl_size)
                self.weights_list.append(arr)
                self.updates.append(np.zeros(np.shape(arr)))
            
            arr = np.random.randn(self.hl_size+1, self.n_output)
            self.weights_list.append(arr)
            self.updates.append(np.zeros(np.shape(arr)))
        
            print('WEIGHTS ARE:', self.weights_list)
            input('Press Enter to continue...')

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
        self.layer_outputs = []
        output = image.flatten()
        self.layer_outputs.append(output)

        for i in range(0, self.n_hidden):
            output = np.append(output, 1)
            output = np.transpose(self.weights_list[i])@output
            self.layer_outputs.append(output)
            output = self.activation(output)
        
        output = np.append(output, 1)
        output = np.transpose(self.weights_list[-1])@output
        #self.layer_outputs.append(output)
        output = self.softmax(output)
        print('OUTPUTS ARE:', self.layer_outputs)
        input('Press Enter to continue...')
        return output
    
    def backprop(self, y_pred, y_true):
        out_err = y_pred - y_true #n_out x 1
        self.updates = []
        count = 0
        for i in range(1,self.n_hidden + 2):
            #self.updates.append()
            #self.updates[-i][:-1,:] = np.outer(self.layer_outputs[-i],out_err) #weight update
            #self.updates[-i][-1,:] = out_err
            weight_update = np.outer(self.layer_outputs[-i],out_err) #weight update
            print('hi',np.shape(self.layer_outputs[-i]), np.shape(out_err))
            bias_update = out_err #bias update
            update = np.vstack([weight_update, bias_update])
            self.updates.append(update)
            del_prev = self.weights_list[-i][:-1,:]@out_err
            grad_prev = self.activation(self.layer_outputs[-i], grad = True)
            out_err = np.multiply(del_prev, grad_prev)
            #print(del_prev, grad_prev, out_err)
            #print('shape of UPDATES ARE:', np.shape(self.updates[-i]))
            #input('Press Enter to continue...')
            #print('UPDATES ARE:',self.updates[count])
            #count = count + 1
            #input('Press Enter to Continue')

        self.updates.reverse()
        print('UPDATES ARE:',self.updates)
        input('Press Enter to Continue')

network = NeuralNetwork()
image  = np.array([[1,1],[1,1]])
output = network.feedforward(image)
print(output)
actual = np.zeros([4])
actual[0] = 1
network.backprop(output, actual)
print(network.weights_list)
    
        
            



