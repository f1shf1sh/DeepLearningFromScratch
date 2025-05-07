import numpy as np
from common.gradient import numerical_gradient
from common.functions import *

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) #使用高斯分布初始化
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss

def func_2(x):
    return x**2
    
if __name__ == "__main__":
    x = np.array([0.6, 0.9])
    simple = simpleNet()
    p = simple.predict(x)
    print(simple.W) 
    print(func_2(simple.W))