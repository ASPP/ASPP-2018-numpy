# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

# Create a vector with values ranging from 10 to 49
Z = np.arange(10,50)

# Create a null vector of size 100 but the fifth value which is 1
Z = np.zeros(100)
Z[4] = 1

# Reverse a vector (first element becomes last)
Z = np.arange(50)[::-1]

# Create a 3x3 matrix with values ranging from 0 to 8
Z = np.arange(9).reshape(3,3)

# Create a 3x3 identity matrix
Z = np.eye(3)

# Create a 2d array with 1 on the border and 0 inside
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0

# Given a 1D array, negate all elements which are between 3 and 8, in place
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
