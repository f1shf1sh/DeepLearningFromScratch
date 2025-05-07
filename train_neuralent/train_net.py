import numpy as np
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist

if __name__ == "__main__":
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    load_mnist(normlize=True, flatten=True, one_hot_label=False)