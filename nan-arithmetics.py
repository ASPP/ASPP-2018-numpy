# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

# Result is NaN
print(0 * np.nan)

# Result is False
print(np.nan == np.nan)

# Result is False
print(np.inf > np.nan)

# Result is NaN
print(np.nan - np.nan)

# Result is False !!!
print(0.3 == 3 * 0.1)
print("0.1 really is {:0.56f}".format(0.1))
