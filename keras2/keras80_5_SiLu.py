# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def silu(x):
    return x * (1 / (1 + np.exp(-x)))

# x 곱하기 sigmoid
#문제점 : relu보다 계산량이 많다 = 모델이 커질수록 부담스러워

y = silu(x)

plt.plot(x,y)
plt.grid()
plt.show()