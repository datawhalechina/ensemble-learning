# MCMC方法之Metropolis-Hastings算法

import numpy as np
import random
import matplotlib.pyplot as plt

## 设置参数
mu = 0.5
sigma = 0.1
skip = 700  ## 设置收敛步长
num = 100000  ##采样点数

def Gussian(x,mu,sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.square((x-mu))/(2*np.square(sigma)))

def M_H(num):
    x_0 = 0
    samples = []
    j = 1
    while(len(samples) <= num):
        while True:
            x_1 = random.random()  # 转移函数(转移矩阵)
            q_i = Gussian(x_0,mu,sigma)
            q_j = Gussian(x_1,mu,sigma)
            alpha = min(1,q_i/q_j)
            u = random.random()
            if u <= alpha:
                x_0 = x_1
                if j >= skip:
                    samples.append(x_1)
                j = j + 1
                break
    return samples

norm_samples = M_H(num)

X = np.array(norm_samples)
px = Gussian(X,mu,sigma)
plt.scatter(X,px)
plt.show()