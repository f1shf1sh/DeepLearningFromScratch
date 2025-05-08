import numpy as np
import matplotlib.pylab as plt

def func_2(x):
    return np.sum(x**2)

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    """
    求函数f在点x处的梯度 grad = (∂f/∂x, ∂f/∂y, ...),
    ∂f/∂x = lim(h->0) {f(x+h) - f(x-h)} / 2*h
    Args:
        f (np.array): 求梯度的原函数
        x (np.array): 函数初始参数

    Returns:
        np.array: grad
    """
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.shape[0]):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fx0 = f(x)
        x[idx] = tmp_val - h
        fx1 = f(x)
        
        grad[idx] = (fx0 - fx1) / (2*h)
    
    return grad 

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    """ 
    梯度下降法实现过程，函数实现以下公式
    x_0 = x_0 - lr * ∂f/∂x_0
    x_1 = x_1 - lr * ∂f/∂x_1
    Args:
        f (function): 梯度函数，函数原型 f(x_0, x_1) = x_0 ** 2 + x_1 ** 2
        init_x (np.array): 梯度函数初始化参数
        lr (np.float): 学习率
        step_num (int): 初始梯度数，逐渐下降
    Returns:
        x (np.array): 训练好的参数    
    """ 
    x = init_x
    x_history = [] 
    for _ in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x = x - lr * grad
    x_history = np.array(x_history) 
    return x, x_history

if __name__ == "__main__":
    init_x = np.array([3.0, 4.0])
    x, x_history = gradient_descent(func_2, init_x = init_x, lr = 0.1, step_num = 100)
    print(x_history)    
    plt.plot( [-5, 5], [0,0], '--b')
    plt.plot( [0,0], [-5, 5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show() 