import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

def sliding_window(Z, size=2):
    n, s = Z.shape[0], Z.strides[0]
    return as_strided(Z, shape=(n-size+1, size), strides=(s, s))

# Rule 30  (see https://en.wikipedia.org/wiki/Rule_30)
# 0x000: 0, 0x001: 1, 0x010: 1, 0x011: 1
# 0x100: 1, 0x101: 0, 0x110: 0, 0x111: 0
rule = 30 
R = np.array([int(v) for v in '{0:08b}'.format(rule)])[::-1]

# Initial state
Z = np.zeros((250,501), dtype=int)
Z[0,250] = 1

# Computing some iterations
for i in range(1, len(Z)):
    N = sliding_window(Z[i-1],3) * [1,2,4]
    Z[i,1:-1] = R[N.sum(axis=1)]

# Display
plt.figure(figsize=(6,3))
plt.subplot(1,1,1,frameon=False)
plt.imshow(Z, vmin=0, vmax=1, cmap=plt.cm.gray_r)
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig("automata.png")
plt.show()
