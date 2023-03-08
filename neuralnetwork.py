import numpy as np
from sklearn.metrics import accuracy_score
import cv2

class NeuralNetwork:
    def __init__(self, n_hidden = 3, hl_size = 32, batch_size = 16, weight_init = 'Xavier', activation_fn = 'sigmoid', optimizer = 'sgd', lr = 0.01, n_input = 28*28, n_output = 10):
        self.n_hidden = n_hidden #number of hidden layers
        self.hl_size = hl_size #size of each hidden layer
        self.batch_size = batch_size 
        self.activation_fn = activation_fn
        self.lr = lr
        self.weight_init = weight_init
        self.n_input = n_input
        self.n_output = n_output
        self.optimizer = optimizer
        self.weights_list = []
        self.updates = []
        self.gradients = []
        self.activated_layers = []
        self.batch_count = 0
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

        else:
            print('Choose correct initialization.')
            quit()
    
    def activation(self, x):
        if self.activation_fn == 'sigmoid':
            return np.array([1/(1+ np.exp(-a)) for a in x])
        elif self.activation_fn == 'tanh':
            return [np.tanh(a) for a in x]
        elif self.activation_fn == 'identity':
            return x
        else:
            return x*(x > 0)
    
    def calculate_grad(self, x):
        if self.activation_fn == 'sigmoid':
            return [a*(1-a) for a in x]
        elif self.activation_fn == 'tanh':
            return [1-a**2 for a in x]
        elif self.activation_fn == 'identity':
            return 1
        else:
            return 1*(x > 0)
    
    def loss_fn(self, predicted, label):
        return np.sum(-label*np.log(predicted))

    def softmax(self, output):
        out = [np.exp(x) for x in output]
        out = out/np.sum((out))
        return out
    
    def feedforward(self, image):
        self.activated_layers = []
        output = image.flatten()
        self.activated_layers.append(output)

        for i in range(0, self.n_hidden):
            output = np.append(output, 1)
            output = np.transpose(self.weights_list[i])@output
            output = self.activation(output)
            self.activated_layers.append(output)
        
        output = np.append(output, 1)
        output = np.transpose(self.weights_list[-1])@output
        output = self.softmax(output)
        return output
    
    def reset_updates(self):
        self.updates = []
        for i in range(0, len(self.weights_list)):
            arr = np.zeros(np.shape(self.weights_list[i]))
            self.updates.append(arr)

    
    def update(self):
        if len(self.gradients) == 0:
            print('Error in backprop.')
            exit()
        else:
            for i in range(0, len(self.updates)):
                self.updates[i] = self.updates[i] + self.gradients[i]

    def backprop(self, y_pred, y_true):
        out_err = y_pred - y_true #n_out x 1
        self.gradients = []
        for i in range(1,self.n_hidden + 2):
            weight_update = np.outer(self.activated_layers[-i],out_err) #weight update
            bias_update = out_err #bias update
            update = np.vstack([weight_update, bias_update])
            self.gradients.append(update)
            del_prev = self.weights_list[-i][:-1,:]@out_err
            grad_prev = self.calculate_grad(self.activated_layers[-i])
            out_err = np.multiply(del_prev, grad_prev)
        self.gradients.reverse()
        self.update()
    
    def optimize(self):
        if self.optimizer=='sgd':
            for i in range(0,len(self.weights_list)):
                self.weights_list[i] = self.weights_list[i] - self.lr*self.updates[i]
    
    def step(self, image, label):
        predicted = self.feedforward(image)
        self.backprop(predicted, label)
        self.batch_count = self.batch_count + 1
        if self.batch_count%self.batch_size==0:
            self.optimize()
            self.reset_updates()
        
        return self.loss_fn(predicted, label)
    
    def inference(self, x_test, y_test):
        predictions = []
        loss = 0
        for (x,y) in zip(x_test,y_test):
            pred = self.feedforward(x)
            loss = loss + self.loss_fn(pred, y_test)
            predicted_label = np.argmax(pred)
            predictions.append(predicted_label)
        
        y_label = [np.argmax(y_test[i]) for i in range(0,len(y_test))]
        
        return loss, accuracy_score(y_label, predictions)




    




        
            



