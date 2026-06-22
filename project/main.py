'''
This is a program to create a neural network that would be capable of recognizing hand written digits.
Further details about the program will be given in the following code only.  
'''

#importing the necessary modules/libraries

import numpy as np


#creating the neural network

class Network:
    # Defining the constructor
    def __init__(self, net_size: list):
        self.net_size: list = net_size
        self.layer_num: int = len(net_size)
        
        self.biases = [ np.random.randn(y, 1) for y in range(net_size[1:]) ] # Defining the biases of the entire network
        self.weights = [ np.random.randn(y,x) for y,x in zip(net_size[ 1: ], net_size[ :-1 ]) ] # Definging the weights for each layer of the entire network
        