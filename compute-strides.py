# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

def compute_strides(Z):
    strides = [Z.itemsize]
    for i in range(Z.ndim-1,0,-1):
        strides.append(strides[-1] * Z.shape[i])
    return tuple(strides[::-1])

# This work
Z = np.arange(24).reshape(2,3,4)
print(Z.strides, " – ", compute_strides(Z))

# This does not work
# Z = Z[::2]
# print(Z.strides, " – ", compute_strides(Z))

