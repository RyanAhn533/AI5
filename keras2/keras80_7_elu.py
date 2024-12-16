<<<<<<< HEAD
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def elu(x):
    return np.where(x > 0, x, 1 * (np.exp(x) - 1))

y = elu(x)

plt.plot(x,y)
plt.grid()
=======
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def elu(x):
    return np.where(x > 0, x, 1 * (np.exp(x) - 1))

y = elu(x)

plt.plot(x,y)
plt.grid()
>>>>>>> cd855f8 (message)
plt.show()