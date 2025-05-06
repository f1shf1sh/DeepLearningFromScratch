import numpy as np
import pickle
import gzip
from pathlib import Path

pickle_path = "./"
pickle_name = "dataset.pkl"
data_dir = "./dataset/mnist_test_data"
IMG_SIZE = 784

file = {
    "train_img":"train-images-idx3-ubyte.gz",
    "train_label":"train-labels-idx1-ubyte.gz",
    "test_img":"t10k-images-idx3-ubyte.gz",
    "test_label":"t10k-labels-idx1-ubyte.gz"
}

def _load_img(file_name):
    folder = Path(data_dir)
    file_path = folder / file_name
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, IMG_SIZE)
    
    return data 

def _load_label(file_name):
    file_path = data_dir + "/" + file_name
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    
    return data 

def _convert_numpy():
    dataset = {}
    dataset["train_img"] = _load_img(file["train_img"])
    dataset["train_label"] = _load_label(file["train_label"])
    dataset["test_img"] = _load_img(file["test_img"])
    dataset["test_label"] = _load_label(file["test_label"])
    
    return dataset

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, arr in enumerate(T):
        arr[X[idx]] = 1
        
    return T

def init_mnist():
    file_path = pickle_path + pickle_name
    dataset = _convert_numpy()
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f, -1) 
    
def load_mnist(normlize = True, flatten = True, one_hot_label = False):
    init_mnist()
    
    file_path = pickle_path + pickle_name
    with open(file_path, "rb") as f:
        dataset = pickle.load(f)

    if normlize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    
    if one_hot_label:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])
    
    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
         
    return (dataset["train_img"], dataset["train_label"]), (dataset["test_img"], dataset["test_label"])
 
if __name__ == "__main__":
    print(load_mnist(flatten=False))