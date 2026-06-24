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

test_data = np.loadtxt(
    "data/dummy_test_train.csv",
    delimiter=',',
    skiprows=1
)
a = test_data[3]
# print(type(test_data))
# print(test_data.shape)
print(type(a))
print(a.shape)
print(a[0])
b = a[1:]
print(type(b))
print(b.shape)
# print(b.reshape(28,28))
# print(a)
