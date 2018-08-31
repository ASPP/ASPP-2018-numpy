# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

# Create our own dtype
dtype = np.dtype([('rank',       'i8'),
                  ('lemma',      'S8'),
                  ('frequency',  'i8'),
                  ('dispersion', 'f8')])

# Load file using our own dtype
data = np.loadtxt('data.txt', comments='%', dtype=dtype)

# Extract words only
print(data["lemma"])

# Extract the 3rd row
print(data[2])

# Print all words with rank < 30
print(data[data["rank"] < 30])

# Sort the data according to frequency (see [np.argsort]()).
sorted = np.sort(data, order="frequency")
print(sorted)

# Save unsorted and sorted array
np.savez("sorted.npz", data=data, sorted=sorted)

# Load saved array
out = np.load("sorted.npz")
print(out["sorted"])
