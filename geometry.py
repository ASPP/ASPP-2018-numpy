# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

dtype = [("points",    float, (4, 2)),
         ("scale",     float, 1),
         ("translate", float, 2),
         ("rotate",    float, 1)]
S = np.zeros(25, dtype = dtype)
S["points"] = [(-1,-1), (-1,+1), (+1,+1), (+1,-1)]
S["translate"] = (1,0)
S["scale"] = 0.1
S["rotate"] = np.linspace(0, 2*np.pi, len(S), endpoint=False)

P = np.zeros((len(S), 4, 2))
for i in range(len(S)):
    for j in range(4):
        x = S[i]["points"][j,0]
        y = S[i]["points"][j,1]
        tx, ty = S[i]["translate"]
        scale  = S[i]["scale"]
        theta  = S[i]["rotate"]
        x = tx + x*scale
        y = ty + y*scale
        x_ = x*np.cos(theta) - y*np.sin(theta)
        y_ = x*np.sin(theta) + y*np.cos(theta)
        P[i,j] = x_, y_

fig = plt.figure(figsize=(6,6))
ax = plt.subplot(1,1,1, frameon=False)
for i in range(len(P)):
    X = np.r_[P[i,:,0], P[i,0,0]]
    Y = np.r_[P[i,:,1], P[i,0,1]]
    plt.plot(X, Y, color="black")
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig("geometry.png")
plt.show()
