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
