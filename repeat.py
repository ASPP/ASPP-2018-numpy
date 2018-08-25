# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from numpy.lib.stride_tricks import as_strided

Z = np.zeros(5)
Z1 = np.tile(Z,(3,1))
Z2 = as_strided(Z, shape=(3,)+Z.shape, strides=(0,)+Z.strides)

# Real repeat (three times the memory)
Z1[0,0] = 1
print(Z1)

# Fake repeat (but less memory)
Z2[0,0] = 1
print(Z2)
