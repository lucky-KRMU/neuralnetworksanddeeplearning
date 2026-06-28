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
    
    # The sigmoid prime function is the derivative of sigmoid function
    def sigmoid_prime(self, a: float) -> float:
        '''
        a is already the final activation(sigmoid) 
        '''
        
        return a * ( 1 - a )
    
    # Defining the feedforward function which will also be used to train the certain data (later)
    def feedforward(self, a: np.ndarray) -> np.ndarray:
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
        self.io_layer = [a] # This is actually the list that would consists of all the input/output of the network for each layer.
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
            x = self.io_layer[i]
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
            self.io_layer.append(a)
        return a
    
    def cost(self, a:float, y: int) -> float:
        return .5 * ( a - y ) ** 2
    
    def vectorize(self, label: int) -> list:
        vec = np.zeros((10,1))
        vec[label] = 1
        return vec
    
    
        
    
    def train(self, training_data: list, lr: float = .01, epochs: int = 1):
        '''
        Information about train function.
        1. About the training_data:
        In this function training data will be given as whole. The training data consists of numpy arrays. it has over 28,000 numpy arrays inside of it. 
        Whole is packed as a single numpy array. creating a 2D matrix.
        each row has 785 columns, in which the first column consists of the label and the other are the actual numbers.
        '''
        
        # necessary variables for SGD
        batch_size = 64
        if (len(training_data)%batch_size == 0):
            training_data_size = len(training_data)//batch_size 
        else:
            training_data_size = len(training_data)//batch_size + 1
        
        # This is the loop for back propagation (investigating the weights)
        
        
        for i in range(epochs):
            
            # applying SGD (Stochastic gradient descent)
            np.random.shuffle(training_data)
            # SGD_lists = [ training_data[:10000], training_data[10000:20000], training_data[20000:30000], training_data[30000:40000], training_data[40000:] ]
            
            
            layer_error = [] # This is for storing δ
            layer_error_gradient = [] # This is for storing the values of updated gradients of weights
            layer_bias_gradient = [] # This is for storing the values of updated gradients of biases
            
            old_index = 0
            
            for j in range(training_data_size):
                SGD_batch = training_data[old_index:old_index+batch_size]
                old_index += batch_size
            
            
                for data in SGD_batch:
                    
                    y = data[0]
                    # here actually the training inputs are just a 1D vector with (784,) shape, but we need a 2D matrix, hence we will change it
                    # train_inputs = data[1:]
                    train_inputs = np.array(data[1:]).reshape(784,1)
                    # print(train_inputs.shape) 
                    
                    self.feedforward(train_inputs)
                    
                    # Back propagation code strats from here
                    gradient = (self.io_layer[self.io_layer[0]-1] - self.vectorize(y)) * self.sigmoid_prime(self.io_layer[0]-1)
                    
                    for k in range(self.layer_num-1, 0, -1):
                        if k == self.layer_num - 1:
                            gradient_weight = gradient * (self.io_layer[k-1].T)
                        else: 
                            gradient_weight = gradient * self.weights[k] * (self.io_layer[k-1].T)
                        gradient_bias = gradient
                        gradient = gradient_weight
                        
                        # appending to the lists
                        layer_error_gradient.append(gradient_bias)
                        layer_bias_gradient.append(gradient_weight)
                        
                # Gradient Descent starts here
                for k in range(self.io_layer-1, 0, -1):
                    self.weights[k] -= lr * layer_error_gradient[k]
                    self.biases[k] -= lr * layer_bias_gradient[k]
                
        

# Loading the MNIST Dataset for the sake of testing a single prediction out of the network (without training)
test_data = np.loadtxt(
    "data/train.csv",
    delimiter=',',
    skiprows=1
)
# print(type(test_data))
# print(len(test_data))
# print(test_data.shape)
a = np.array([test_data[1][1:]]).T
# print(type(a))
# print(len(a))
# print(a.shape)
        
    
N = Network([784, 50, 10])
        
# a = np.random.randn(784,1)
output = N.feedforward(a)
# print(output)
N.train(test_data[1:])

# test_vec = N.vectorize(3)
# print(test_vec)