# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import random
import numpy as np
from tools import timeit

def random_walk_slow(n):
    position = 0
    walk = [position]
    for i in range(n):
        position += 2*random.randint(0, 1)-1
        walk.append(position)
    return walk


def random_walk_faster(n=1000):
    from itertools import accumulate
    # Only available from Python 3.6
    steps = random.choices([-1,+1], k=n)
    return [0]+list(accumulate(steps))

def random_walk_fastest(n=1000):
    steps = np.random.choice([-1,+1], n)
    return np.cumsum(steps)


if __name__ == '__main__':

    timeit("random_walk_slow(1000)", globals())
    timeit("random_walk_faster(1000)", globals())
    timeit("random_walk_fastest(1000)", globals())
