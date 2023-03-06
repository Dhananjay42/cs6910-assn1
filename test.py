import numpy as np

def sigmoid(x):
    return np.array([1/(1+ np.exp(-a)) for a in x])

def sigmoid_grad(x):
    return [np.exp(-a)/((1 + np.exp(-a))**2) for a in x]

def softmax(output):
    out = [np.exp(x) for x in output]
    out = out/np.sum((out))
    return out

matrix = np.array([[ 0.27107295, -0.00996949],
       [ 1.51507033,  0.70715213],
       [ 0.58185761,  1.27452456],
       [-0.05122307, -0.19037129],
       [-1.4405707 , -1.6585063 ]])

inp = np.array([1]*4)
inp = np.append(inp, 1)
out1 = np.transpose(matrix)@inp
out1 = sigmoid(out1)
#print(out1)

matrix2 = np.array([[-0.75966085,  0.78500062],
       [-0.49926614,  1.51791477],
       [-0.66171071, -0.40754262]])
inp2 = np.append(out1, 1)
out2 = np.transpose(matrix2)@inp2
out2 = sigmoid(out2)
#print(out2)

matrix3 = np.array([[ 0.86014048, -0.80100188],
       [-0.04659072, -1.82801432],
       [-0.64376175, -0.41756069]])
inp3 = np.append(out2, 1)
out3 = np.transpose(matrix3)@inp3
out3 = sigmoid(out3)
#print(out3)

matrix4 = np.array([[-0.29938719,  0.0238128 , -0.55814893,  1.3474676 ],
       [ 0.40173125,  1.55397761, -1.02134338, -0.54521326],
       [-1.59002541,  1.5098208 , -1.00553582, -0.78297386]])
inp4 = np.append(out3, 1)
out4 = np.transpose(matrix4)@inp4
out4 = softmax(out4)
#print(out4)

actual = np.zeros([4])
actual[0] = 1
out_err = out4 - actual
#print(actual, out4, out_err)
weight_grad = np.outer(out3,out_err)
bias_grad = out_err
update = np.vstack([weight_grad, bias_grad])
print(update)

del_3 = matrix4[:-1,:]@out_err
grad_prev = sigmoid_grad()