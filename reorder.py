# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import struct
import numpy as np

# Generation of the array
# Z = range(1001, 1009)
# L = np.reshape(Z, (2,2,2), order="F").ravel().astype(">i8").view(np.ubyte)

L = [  0,   0,   0,   0,   0,   0,   3, 233,
       0,   0,   0,   0,   0,   0,   3, 237,
       0,   0,   0,   0,   0,   0,   3, 235,
       0,   0,   0,   0,   0,   0,   3, 239,
       0,   0,   0,   0,   0,   0,   3, 234,
       0,   0,   0,   0,   0,   0,   3, 238,
       0,   0,   0,   0,   0,   0,   3, 236,
       0,   0,   0,   0,   0,   0,   3, 240]

# Automatic (numpy)
Z = np.reshape(np.array(L, dtype=np.ubyte).view(dtype=">i8"), (2,2,2), order="F")
print(Z[1,0,0])

# Manual (brain)
shape = (2,2,2)
itemsize = 8
# We can probably do better
strides = itemsize, itemsize*shape[0], itemsize*shape[0]*shape[1]
index = (1,0,0)
start = sum(i*s for i,s in zip(index,strides))
end = start+itemsize
value = struct.unpack(">Q", bytes(L[start:end]))[0]
print(value)
