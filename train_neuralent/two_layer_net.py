import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import numerical_gradient
from common.functions import *
from common.layers import *
from collections import OrderedDict

class TwoLayerNet:
    """
        两层基本神经网络
        W1: 784 x 100
        W2: 100 x 10
        b1: 1 x 100
        b2: 1 x 10
    """
    def __init__(self, input_size, hidden_size, output_size, wight_init_std = 0.01):
        self.params = {}
        self.params["W1"] = wight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = wight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        
        # 生成层
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftMaxWithLoss() 
         
    def predict(self, x):
        """
        Args:
            x (np.array):
        Return:
             
        """
        # version 1.0
        # W1, W2 = self.params["W1"], self.params["W2"]
        # b1, b2 = self.params["b1"], self.params["b2"]

        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)
        
        # version 2.0
        for layer in self.layers.values():
            x = layer.forward(x) 
        return x
    
    
    def loss(self, x, t):
        """
        使用交叉熵损失
        Args:
            x (np.array): 输入数据
            t (np.array): 监督数据
        Returns:
            误差函数 
        """
        y = self.predict(x) 
        # y为100 x 10
        # x为100 x 10
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        """ 准确率计算, accuracy = success_data / all_data
        Args:
            x (_type_): _description_
            t (_type_): _description_

        Returns:
            返回识别准确率
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1) # axis=1表示按照列进行方向进行压缩，列求和
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
         
        accuracy = np.sum(y == t) / float(x.shape[0])
        
        return accuracy 
    
    def numerical_gradient(self, x, t):
        """
            loss为损失函数,目的是求出损失函数最小,找到最优权值
        Args:
            x (): 输入数据
            t (): 监督数据
        Returns:

        """
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
        
        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 更新 
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        
        return grads 
        
        
if __name__ == "__main__":
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params["W1"])
    