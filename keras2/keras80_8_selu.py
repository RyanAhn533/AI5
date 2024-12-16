# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def selu(x, lambda_x=1.0507, alpha = 1.67326):
    return np.where(x>0, lambda_x*x, lambda_x*alpha*(np.exp(x)-1))

y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()