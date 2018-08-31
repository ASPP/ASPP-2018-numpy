# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

def strides(Z):
    strides = [Z.itemsize]
    
    # Fotran ordered array
    if np.isfortran(Z):
        for i in range(0, Z.ndim-1):
            strides.append(strides[-1] * Z.shape[i])
        return tuple(strides)
    # C ordered array
    else:
        for i in range(Z.ndim-1, 0, -1):
            strides.append(strides[-1] * Z.shape[i])
        return tuple(strides[::-1])

# This work
Z = np.arange(24).reshape((2,3,4), order="C")
print(Z.strides, " – ", strides(Z))

Z = np.arange(24).reshape((2,3,4), order="F")
print(Z.strides, " – ", strides(Z))

# This does not work
# Z = Z[::2]
# print(Z.strides, " – ", strides(Z))

