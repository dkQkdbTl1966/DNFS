import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exps = np.exp(x)
    return exps / (np.sum(exps).reshape(-1,1))

def relu(x):
    return 1.0*(x>0)

def leaky_relu(x, leaky_slope):
    d=np.zeros_like(x)
    d[x<=0]= leaky_slope
    d[x>0]=1
    return d

#Defining dummy values of x 
x = np.linspace(-np.pi, np.pi, 12)

#Finding the Activation Function Outputs
sigmoid_output = sigmoid(x)
tanh_output = tanh(x)
softmax_output = softmax(x)
relu_output = relu(x)
leaky_relu_output = leaky_relu(x,1)

#Printing the Outputs
print(sigmoid_output)
print(tanh_output)
print(softmax_output)
print(relu_output)
print(leaky_relu_output)