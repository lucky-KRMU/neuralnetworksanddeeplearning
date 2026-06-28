'''
This is a python file which is made to run small experimental code before getting implemented in the origin code for project
'''


import numpy as np

# def sigmoid(x): return 1/(1+np.exp(-x))

# array = np.array([784, 50, 10])

# W = [np.random.randn(y,x) for x,y in zip(array[1:], array[:-1])]
# X = [np.random.randn(y,x) for x,y in zip(array[1:], array[:-1])]
# B = [np.random.randn(x) for x in range(len(array[1:]))]

# for i in range(len(W)):
#     print(W[i].shape)
# print('\n')
# for i in range(len(X)):
#     print(X[i].shape)
 
# y = []
    
# for i in range(len(W)):
#     a = np.dot(W[i], X[i])
#     y.append(a)

# print(y)    

# test_data = np.loadtxt(
#     "data/dummy_test_train.csv",
#     delimiter=',',
#     skiprows=1
# )
# a = test_data[3]
# print(type(test_data))
# print(test_data.shape)
# print(type(a))
# print(a.shape)
# print(a[0])
# b = a[1:]
# print(type(b))
# print(b.shape)
# print(b.reshape(28,28))
# print(a)


class Network:
    def __init__(self, list):
        self.layers = list
        self.num_layers = len(list)
        self.weights = [ np.random.randn(x,y) for x,y in zip(self.layers[1:], self.layers[:-1]) ]
        self.biases = [ np.random.randn(y, 1) for y in  self.layers[1:] ]
        
    def info(self):
        print("Information about the weights: ")
        for layer in self.weights:
            print(layer.shape)
        print("Information about biases: ")
        for layer in self.biases:
            print(layer.shape)
        
Network = Network([784, 50, 10])
Network.info()

# a = np.loadtxt('data/dummy_test_train.csv', skiprows=1, delimiter=',')
# print("loaded")
# print(type(a))
# print(a.shape)
# train_inputs = np.array(a[1][1:]).reshape(784,1)
# print("loaded")
# print(type(train_inputs))
# print(train_inputs.shape)

a = [1,2,3,4,5,6,7,8]
print(a[2:20])