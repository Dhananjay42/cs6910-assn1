import numpy as np

def sigmoid(x):
    return np.array([1/(1+ np.exp(-a)) for a in x])

def sigmoid_grad(x):
    return [np.exp(-a)/((1 + np.exp(-a))**2) for a in x]

def softmax(output):
    out = [np.exp(x) for x in output]
    out = out/np.sum((out))
    return out

matrix = np.array([[1. , 0.5],
       [1. , 0.5],
       [1. , 0.5],
       [1. , 0.5],
       [1. , 0.5]])

inp = np.array([1]*4)
inp = np.append(inp, 1)
out1 = np.transpose(matrix)@inp
#out1 = sigmoid(out1)
#print(out1)

matrix2 = np.array([[-1, -2],
       [-1, -2],
       [-1, -2]])
inp2 = np.append(out1, 1)
out2 = np.transpose(matrix2)@inp2
#out2 = sigmoid(out2)
#print(out2)

matrix3 = np.array([[-1, -2],
       [-1, -2],
       [-1, -2]])
inp3 = np.append(out2, 1)
out3 = np.transpose(matrix3)@inp3
#out3 = sigmoid(out3)
#print(out3)

matrix4 = np.array([[ 1. ,  0.5,  1. , -1. ],
       [ 1. ,  0.5,  1. , -1. ],
       [ 1. ,  0.5,  1. , -1. ]])
inp4 = np.append(out3, 1)
out4 = np.transpose(matrix4)@inp4
print(out4)
out4 = softmax(out4)
print(out4)

actual = np.zeros([4])
actual[0] = 1
out_err = out4 - actual
#print(actual, out4, out_err)
weight_grad = np.outer(out3,out_err)
bias_grad = out_err
update = np.vstack([weight_grad, bias_grad])
#print(update)

del_3 = matrix4[:-1,:]@out_err
#grad_prev = sigmoid_grad()
arr = [[1,0.5]]*4
#print(np.shape(arr))