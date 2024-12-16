<<<<<<< HEAD
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

y = mish(x)

plt.plot(x,y)
plt.grid()
plt.show()

#7. elu
#8. selu
=======
# 다른말로 swish = SiLu

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

y = mish(x)

plt.plot(x,y)
plt.grid()
plt.show()

#7. elu
#8. selu
>>>>>>> cd855f8 (message)
#9. leaky_relu