import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import numerical_gradient
from common.functions import *

class simpleNet:
    def __init__(self):
        self.W = np.array([[ 0.47355232,0.9977393, 0.84668094],
                           [ 0.85557411, 0.03563661, 0.69422093]])
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss

if __name__ == "__main__":
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    simple_net = simpleNet()
    y = simple_net.predict(x)
    print("simple_net.W:\n", simple_net.W)
    print("test simple_net.loss:", simple_net.loss(x, t))
    
    # def f(W):
        # W是伪参数，没有实际用处，局部变量x, t直接被使用
        # return simple_net.loss(x, t) 
    f = lambda W: simple_net.loss(x, t)
    dW = numerical_gradient(f, simple_net.W)
    print(dW)