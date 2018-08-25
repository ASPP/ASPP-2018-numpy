# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import as_strided

def sliding_window(Z, size=2):
    n, s = Z.shape[0], Z.strides[0]
    return as_strided(Z, shape=(n-size+1, size), strides=(s, s))


# Initial conditions:
# Domain size is 100 and we'll iterate over 50 time steps
U = np.zeros((50,100))
U[0,5::10] = 1.5

# Actual iteration
F = 0.05
for i in range(1, len(Z)):
    Z[i,1:-1] = Z[i-1,1:-1] + F*(sliding_window(Z[i-1], 3)*[+1,-2,+1]).sum(axis=1)

# Display
plt.figure(figsize=(6,3))
plt.subplot(1,1,1,frameon=False)
plt.imshow(Z, vmin=0, vmax=1)
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig("diffusion.png")
plt.show()
