import numpy as np

def cross_entropy_error(y, t):
    """
    E = - (1/N) * sum( t_i * log(y_i) )
    Args:
        y (np.array): 神经网络输出 
        t (np.array): 监督数据的one-hot数组

    Returns:
        np.float: 损失函数总和 
    """
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def sigmoid(x):
    """_summary_
    σ(x) = 1 / (1 + exp(-x))
    Args:
        x (np.array): _description_
    Returns:
        1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """_summary_

    Args:
        x (np.array): 

    Returns:
        _type_: _description_
    """
    c = np.max(x)
    exp_a = np.exp(x -c)
    sum_exp_a = np.sum(exp_a)
    
    return exp_a / sum_exp_a
    