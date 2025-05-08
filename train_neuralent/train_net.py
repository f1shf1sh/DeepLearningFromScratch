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
    (train_img, train_label), (_, _) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    
    iters_num = 1000
    train_size = train_img.shape[0]
    
    batch_size = 100
    learning_rate = 0.1
    loss_train_list = [] 
    for i in range(iters_num): # get mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_img[batch_mask]
        t_batch = train_label[batch_mask]
        
        # cal grads
        grad = network.numerical_gradient(x_batch, t_batch)
        
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key] 
        
        loss = network.loss(x_batch, t_batch)
        loss_train_list.append(loss)
    
    plt.plot(loss_train_list)
    plt.title("Loss List")
    plt.xlabel("iters")
    plt.ylabel("loss value")
    plt.grid(True)
    plt.savefig("iter_1000.png")

if __name__ == "__main__":
    train_network()