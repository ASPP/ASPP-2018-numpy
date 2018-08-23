import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided


def sliding_window(Z, size=2):
    n = Z.shape[0]
    s = Z.strides[0]
    return as_strided(Z, shape=(n-size+1, size), strides=(s, s))


Z = np.random.uniform(0.00, 0.05, (50,100))
Z[0,5::10] = 1.5

a = 0.75
for i in range(1, len(Z)):
    Z[i,1:-1] = a*Z[i-1,1:-1] + (1-a)*np.mean(sliding_window(Z[i-1], 3), axis=1)

plt.figure(figsize=(6,3))
plt.subplot(1,1,1,frameon=False)
plt.imshow(Z, vmin=0, vmax=1)
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig("diffusion.png")
plt.show()
