# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import imageio
import numpy as np
import scipy.spatial

# Number of final colors we want
n = 16

# Original Image
I = imageio.imread("kitten.jpg")
shape = I.shape

# Flattened image
I = I.reshape(shape[0]*shape[1], shape[2])

# Find the unique colors and their frequency (=counts)
colors, counts = np.unique(I, axis=0, return_counts=True)

# Get the n most frequent colors
sorted = np.argsort(counts)[::-1]
C = I[sorted][:n]

# Compute distance to most frequent colors
D = scipy.spatial.distance.cdist(I, C, 'sqeuclidean')

# Replace colors with closest one
Z = (C[D.argmin(axis=1)]).reshape(shape)

# Save result
imageio.imsave("kitten-dithered.jpg", Z)
