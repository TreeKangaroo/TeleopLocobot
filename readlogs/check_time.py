import numpy as np
import matplotlib.pyplot as plt

with open('iktimes.npy', 'rb') as f:
        a = np.load(f)

x=[i for i in range(0, 500)]
print(a[0:5])
plt.plot(x,a)
plt.show()