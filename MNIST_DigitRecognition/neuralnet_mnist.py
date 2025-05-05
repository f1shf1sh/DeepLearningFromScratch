import numpy as np
import pickle
from common.mnist import load_mnist

w_file_path = "./sample_weight.pkl"

def get_data():
    (train_img, train_label), (test_img, test_label) = load_mnist()
    
    return train_img, train_label
    
def init_network():
    with open(w_file_path, "rb") as f:
        network = pickle.load(f)
     
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    
    return exp_a / sum_exp_a 
    
def predict(network, X):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(X, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z3 = softmax(a3) 
    y = z3
    
    return y


if __name__ == "__main__":
    test_img, test_label = get_data()
    network = init_network()

    accuracy_count = 0
    for i in range(len(test_img)):
        y = predict(network, test_img[i])
        p = np.argmax(y)
        if p == test_label[i]:
            accuracy_count += 1
    
    print("Accuracy:", accuracy_count/len(test_img)) 