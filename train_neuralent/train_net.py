import numpy as np
import time
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 执行耗时: {end - start:.6f} 秒")
        return result
    return wrapper

@timer
def train_network():
    network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    
    iters_num = 10000
    train_size = x_train.shape[0] # 60000
    
    batch_size = 100
    learning_rate = 0.1
    loss_train_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1) # 600
    
    for i in range(iters_num): # get mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # cal grads
        grad = network.gradient(x_batch, t_batch)
        
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key] 

        loss = network.loss(x_batch, t_batch)
        loss_train_list.append(loss)
        if i % iter_per_epoch:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
    print(train_acc_list, test_acc_list)
  
if __name__ == "__main__":
    train_network()