# Advanced NumPy

A 3h00 course on advanced numpy techniques  
[Nicolas P. Rougier](http://www.labri.fr/perso/nrougier), [G-Node summer school](https://python.g-node.org/), Camerino, Italy, 2018

> NumPy is a library for the Python programming language, adding support for
> large, multi-dimensional arrays and matrices, along with a large collection of
> high-level mathematical functions to operate on these arrays.
>
> – Wikipedia

**Quicklinks**: [Numpy website](https://www.numpy.org) – [Numpy GitHub](https://github.com/numpy/numpy) – [Numpy documentation](https://www.numpy.org/devdocs/reference/) – [ASPP archives](https://python.g-node.org/wiki/archives)


**Table of Contents**:

* [Introduction](#--introduction)
* [Warmup](#--warmup)
* [Advanced exercises](#--advanced-exercises)
* [References](#--references)

---

## ❶ – Introduction

NumPy is all about vectorization. If you are familiar with Python, this is the
main difficulty you'll face because you'll need to change your way of thinking
and your new friends (among others) are named "vectors", "arrays", "views" or
"ufuncs". Let's take a very simple example: random walk.

One obvious way to write a random walk in Python is:

```
def random_walk_slow(n):
    position = 0
    walk = [position]
    for i in range(n):
        position += 2*random.randint(0, 1)-1
        walk.append(position)
    return walk
walk = random_walk_slow(1000)
```


It works, but it is slow. We can do better using the itertools Python module
that offers a set of functions for creating iterators for efficient looping. If
we observe that a random walk is an accumulation of steps, we can rewrite the
function by first generating all the steps and accumulate them without any
loop:

```
def random_walk_faster(n=1000):
    from itertools import accumulate
    # Only available from Python 3.6
    steps = random.choices([-1,+1], k=n)
    return [0]+list(accumulate(steps))
walk = random_walk_faster(1000)
```

It is better but still, it is slow. A more efficient implementation, taking
full advantage of NumPy, can be written as:

```
def random_walk_fastest(n=1000):
    steps = np.random.choice([-1,+1], n)
    return np.cumsum(steps)
walk = random_walk_fastest(1000)
```

Now, it is amazingly fast !


Before heading to the course, I would like to warn you about a potential
problem you may encounter once you'll have become familiar with NumPy. It is a
very powerful library and you can make wonders with it but, most of the time,
this comes at the price of readability. If you don't comment your code at the
time of writing, you won't be able to tell what a function is doing after a few
weeks (or possibly days). For example, can you tell what the two functions
below are doing?

```
def function_1(seq, sub):
    return [i for i in range(len(seq) - len(sub)) if seq[i:i+len(sub)] == sub]

def function_2(seq, sub):
    target = np.dot(sub, sub)
    candidates = np.where(np.correlate(seq, sub, mode='valid') == target)[0]
    check = candidates[:, np.newaxis] + np.arange(len(sub))
    mask = np.all((np.take(seq, check) == sub), axis=-1)
    return candidates[mask]
```


As you may have guessed, the second function is the
vectorized-optimized-faster-NumPy version of the first function and it runs 10x
faster than the pure Python version. But it is hardly readable.

Last, but not least, you may have noticed the the `random_walk_fast` works but
is not reproducible at all, which is pretty annoying. If you want to know why,
you can have a look at the article [Re-run, Repeat, Reproduce, Reuse,
Replicate: Transforming Code into Scientific
Contributions](https://www.frontiersin.org/articles/10.3389/fninf.2017.00069/full).



## ❷ – Warmup

You're supposed to be already familiar with NumPy. If not, you should read the
[NumPy chapter](http://www.scipy-lectures.org/intro/numpy/index.html) from the
[SciPy Lecture Notes](http://www.scipy-lectures.org/). Before heading to the
more advanced stuff, let's do some warmup exercises (that should pose no
problem):

###  Basic manipulation

• Create a vector with values ranging from 10 to 49  
• Create a null vector of size 100 but the fifth value which is 1  
• Reverse a vector (first element becomes last)  
• Create a 3x3 matrix with values ranging from 0 to 8  
• Create a 3x3 identity matrix  
• Create a 2d array with 1 on the border and 0 inside   
• Given a 1D array, negate all elements which are between 3 and 8, in place  

<details><summary><b>Solution</b> (click to expand)</summary><p>

Sources: [basic-manipulation.py](basic-manipulation.py)

```Python
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
Z[(3 < Z) & (Z <= 8)] *= -1`
```
</p></details>


###  NaN arithmetics

What is the result of the following expression?  
**→ Hints**: [What Every Computer Scientist Should Know About Floating-Point Arithmetic, D. Goldberg, 1991](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)  

```
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)
```


<details><summary><b>Solution</b> (click to expand)</summary><p>

Sources [nan-arithmetics.py](nan-arithmetics.py)

```
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
```

</p></details>

###  Computing strides

Consider an array Z, how to compute Z strides (manually)?  
**→ Hints**: [itemsize](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.itemsize.html) – [shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) – [ndim](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ndim.html)  


```
import numpy as np
Z = np.arange(24).reshape(2,3,4)
print(Z.strides)
```

<details><summary><b>Solution</b> (click to expand)</summary><p>

Sources [compute-strides.py](compute-strides.py)

```
import numpy as np

def compute_strides(Z):
    strides = [Z.itemsize]
    for i in range(Z.ndim-1,0,-1):
        strides.append(strides[-1] * Z.shape[i])
    return tuple(strides[::-1])

# This work
Z = np.arange(24).reshape(2,3,4)
print(Z.strides, " – ", compute_strides(Z))

# This does not work
# Z = Z[::2]
# print(Z.strides, " – ", compute_strides(Z))
```

</p></details>


###  Repeat and repeat

Can you tell the difference?  
**→ Hints**: [tile](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html) – [as_strided](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.lib.stride_tricks.as_strided.html)  

```
import numpy as np
from numpy.lib.stride_tricks import as_strided

Z = np.random.randint(0,10,5)
Z1 = np.tile(Z, (3,1))
Z2 = as_strided(Z, shape=(3,)+Z.shape, strides=(0,)+Z.strides)
```

<details><summary><b>Solution</b> (click to expand)</summary><p>

Sources [repeat.py](repeat.py)

```
import numpy as np
from numpy.lib.stride_tricks import as_strided

Z = np.zeros(5)
Z1 = np.tile(Z,(3,1))
Z2 = as_strided(Z, shape=(3,)+Z.shape, strides=(0,)+Z.strides)

# Real repeat: three times the memory
Z1[0,0] = 1
print(Z1)

# Fake repeat: less memory but not totally equivalent
Z2[0,0] = 1
print(Z2)
```

</p></details>


###  Moving average

Use `as_strided` to produce a sliding-window view of a 1D array and use it to
produce the picture below (by computing mean filtering with a window size of 3).

![](diffusion.png)

<details><summary><b>Solution</b> (click to expand)</summary><p>

Sources [diffusion.py](diffusion.py)

```
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided


def sliding_window(Z, size=2):
    n = Z.shape[0]
    s = Z.strides[0]
    return as_strided(Z, shape=(n-size+1, size), strides=(s, s))


Z = np.random.uniform(0.00, 0.05, (50,100))
Z[0,5::10] = 1.5

a = 0.75
for i in range(1, len(Z)):
    Z[i,1:-1] = a*Z[i-1,1:-1] + (1-a)*np.mean(sliding_window(Z[i-1], 3), axis=1)

plt.figure(figsize=(6,3))
plt.subplot(1,1,1,frameon=False)
plt.imshow(Z, vmin=0, vmax=1)
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig("diffusion.png")
plt.show()
```

</p></details>

## ❸ – Advanced exercises
## ❹ – References

###  Book & tutorials

* [From Python to Numpy](http://www.labri.fr/perso/nrougier/from-python-to-numpy/),
  Nicolas P.Rougier, 2017
* [100 Numpy Exercises](https://github.com/rougier/numpy-100),
  Nicolas P. Rougier, 2017
* [SciPy Lecture Notes](http://www.scipy-lectures.org/),
  Gaël Varoquaux, Emmanuelle Gouillart, Olav Vahtras et al., 2016
* [Elegant SciPy: The Art of Scientific Python](https://github.com/elegant-scipy/elegant-scipy),
  Juan Nunez-Iglesias, Stéfan van der Walt, Harriet Dashnow, 2016
* [Numpy Medkit](http://mentat.za.net/numpy/numpy_advanced_slides),
  Stéfan van der Walt, 2008

###  Archives

You can access all ASPP archives from https://python.g-node.org/wiki/archives

* **2017** (Juan Nunez-Iglesias): [exercises](https://github.com/jni/aspp2017-numpy) –  [solutions](https://github.com/jni/aspp2017-numpy-solutions)
* **2016** (Stéfan van der Walt): [exercises](https://github.com/ASPP/2016_numpy)
* **2015** (Juan Nunez-Iglesias): [exercises](https://github.com/jni/aspp2015/tree/delivered) – [solutions](https://github.com/jni/aspp2015/tree/solved-in-class)
* **2014** (Stéfan van der Walt): [notebooks](https://python.g-node.org/python-summerschool-2014/_media/numpy_advanced.tar.bz2)
* **2013** (Stéfan van der Walt): [slides](https://python.g-node.org/python-summerschool-2013/_media/advanced_numpy/slides/index.html) – [exercises](https://python.g-node.org/python-summerschool-2013/_media/advanced_numpy/problems.html) – [dropbox](https://www.dropbox.com/sh/4esl1ii7cac5xfa/O-CSFKKYvS/assp2013/numpy_problems)
* **2012** (Stéfan van der Walt): [slides](https://python.g-node.org/python-summerschool-2012/_media/wiki/numpy/numpy_kiel2012.pdf) – [exercises](https://python.g-node.org/python-summerschool-2012/_media/wiki/numpy/problems.html)
* **2011** (Pauli Virtanen): [slides](https://python.g-node.org/python-summerschool-2011/_media/materials/numpy/numpy-slides.pdf) – [exercises](https://python.g-node.org/python-summerschool-2011/_media/materials/numpy/numpy-exercises.zip) – [solutions](https://python.g-node.org/python-summerschool-2011/_media/materials/numpy/numpy-solutions.zip)
* **2010.2** (Stéfan van der Walt): [slides](https://python.g-node.org/python-autumnschool-2010/_media/materials/advanced_numpy/numpy_trento2010.pdf) – [exercises](https://python.g-node.org/python-autumnschool-2010/_media/materials/advanced_numpy/problems.html) – [solutions 1](https://python.g-node.org/python-autumnschool-2010/_media/materials/advanced_numpy/array_interface/solution.py) – [solutions 2](https://python.g-node.org/python-autumnschool-2010/_media/materials/advanced_numpy/structured_arrays/load_txt_solution.py)
* **2010.1** (artosz Teleńczuk): [slides](https://python.g-node.org/python-winterschool-2010/_media/scientific_python.pdf) – [exercises](https://python.g-node.org/python-winterschool-2010/_media/python_tools_for_science.pdf)
* **2009** (Jens Kremkow): [slides](https://python.g-node.org/python-summerschool-2009/_media/numpy_scipy_matplotlib_pynn_neurotools.pdf) – [examples](https://python.g-node.org/python-summerschool-2009/_media/examples_numpy.py) – [exercises](https://python.g-node.org/python-summerschool-2009/_media/exercises_day2_numpy.py)
