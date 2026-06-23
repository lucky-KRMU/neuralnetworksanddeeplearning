'''
This is a program to create a neural network that would be capable of recognizing hand written digits.
Further details about the program will be given in the following code only.  
'''

#importing the necessary modules/libraries


import numpy as np

np.random.seed(42)


#creating the neural network

class Network:
    # Defining the constructor
    def __init__(self, net_size: list):
        self.net_size: list = net_size
        self.layer_num: int = len(net_size)
        
        self.biases = [ np.random.randn(y, 1) for y in (net_size[1:]) ] # Defining the biases of the entire network\
        self.weights = [ np.random.randn(y,x) for y,x in zip(net_size[ 1: ], net_size[ :-1 ]) ] # Definging the weights for each layer of the entire network
        
    # The sigmoid/activatoin function
    def sigmoid(self, z: float) -> float:
        return 1/( 1 + np.exp(-z) )
    
    # Defining the feedforward function which will also be used to train the certain data (later)
    def feedforward(self, a: list) -> list:
        '''
        For this specific network, we know that:
        the weight matricies are [784,50] and [50,10] (dimensions/sizes)
        and the bias matrices are [50,1] and [10,1] (dimensions/sizez)
        1. The first layer(hidden layer) consists of 50 neurons and each neuron has a single bias and 784 weights.
        2. The second layer(output layer) consists of 10 neurons and each neuron has a single bias and 50 weights.
        total parameters of this Neural Network would turn out ot be 784x50 + 50 + 50x10 + 10 = 39,760
        Hence, this model would have 39,760 parameters.
        '''
        a = a/255 # -> this is used for normalizing the value of each pixel. pixel activation lies in [0,255]
        io_layer = [a] # This is actually the list that would consists of all the input/output of the network for each layer.
        # printing certain values for the sake of debugging.
        # print(type(io_layer[0]))
        # print(io_layer[0].shape)
        # print(self.weights[0].shape)
        # print(self.layer_num-1)
        # print([x for x in range(self.layer_num - 1)])
        for i in range(self.layer_num - 1):
            # Defining certain required entities
            w = self.weights[i]
            b = self.biases[i]
            x = io_layer[i]
            # printing certain values for the sake of debugging.
            # print(w,b,act)
            # print(i+1," loop working till here!")
            # print(w.shape)
            # print(b.shape)
            # print(x.shape)
            z = w @ x + b
            a = self.sigmoid(z)
            # @ decorator is used in python numpy to do matrix multiplication.
            # print(i+1, " loop worked")
            io_layer.append(a)
        return a
            
        

# Loading the MNIST Dataset for the sake of testing a single prediction out of the network (without training)
test_data = np.loadtxt(
    "data/test.csv",
    delimiter=',',
    skiprows=1
)
# print(type(test_data))
# print(len(test_data))
# print(test_data.shape)
a = np.array([test_data[0]]).T
# print(type(a))
# print(len(a))
# print(a.shape)
        
    
N = Network([784, 50, 10])
        
# a = np.random.randn(784,1)
output = N.feedforward(a)
print(output)