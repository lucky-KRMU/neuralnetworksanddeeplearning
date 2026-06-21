'''
This is a python file to perform temporary calculations
'''

import random

sum = 0

for i in range(120000):
    exp = random.random()
    pre = random.random()
    
    cost = (exp - pre) ** 2
    sum+=cost

print(sum/120000)

sumI = 0

for j in range(10000):
    exp = random.random()
    pre = random.random()
    
    cost = (exp - pre) ** 2
    sumI+=cost

print(sumI/10000)