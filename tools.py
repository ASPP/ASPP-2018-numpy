# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
from imshow import imshow

def sysinfo():
    import sys
    import time
    import numpy as np
    import scipy as sp
    import matplotlib

    print("Date:       %s" % (time.strftime("%D")))
    version = sys.version_info
    major, minor, micro = version.major, version.minor, version.micro
    print("Python:     %d.%d.%d" % (major, minor, micro))
    print("Numpy:     ", np.__version__)
    print("Scipy:     ", sp.__version__)
    print("Matplotlib:", matplotlib.__version__)


def timeit(stmt, globals=globals()):
    import numpy as np
    import timeit as _timeit
    
    # Rough approximation of a 10 runs
    trial = _timeit.timeit(stmt, globals=globals, number=10)/10
    
    # Maximum duration
    duration = 5.0
    
    # Number of repeat
    repeat = 7
    
    # Compute rounded number of trials
    number = max(1,int(10**np.ceil(np.log((duration/repeat)/trial)/np.log(10))))
    
    # Only report best run
    times = _timeit.repeat(stmt, globals=globals, number=number, repeat=repeat)
    times = np.array(times)/number
    mean = np.mean(times)
    std = np.std(times)

    # Display results
    units = {"s":  1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    for key,value in units.items():
        unit, factor = key, 1/value
        if mean > value: break
    mean *= factor
    std *= factor
    
    print("%.3g %s ± %.3g %s per loop (mean ± std. dev. of %d runs, %d loops each)" %
          (mean, unit, std, unit, repeat, number))

    
def info(Z):
    import numpy as np
        
    print("------------------------------")
    print("Interface (item)")
    print("  shape:      ", Z.shape)
    print("  dtype:      ", Z.dtype)
    print("  size:       ", Z.size)
    if np.isfortran(Z):
        print("  order:       ☐ C  ☑ Fortran")
    else:
        print("  order:       ☑ C  ☐ Fortran")
    print("")
    print("Memory (byte)")
    print("  item size:  ", Z.itemsize)
    print("  array size: ", Z.size*Z.itemsize)
    print("  strides:    ", Z.strides)
    print("")
    print("Properties")
    if Z.flags["OWNDATA"]:
        print("  own data:    ☑ Yes  ☐ No")
    else:
        print("  own data:    ☐ Yes  ☑ No")
    if Z.flags["WRITEABLE"]:
        print("  writeable:   ☑ Yes  ☐ No")
    else:
        print("  writeable:   ☐ Yes  ☑ No")
    if np.isfortran(Z) and Z.flags["F_CONTIGUOUS"]:
        print("  contiguous:  ☑ Yes  ☐ No")
    elif not np.isfortran(Z) and Z.flags["C_CONTIGUOUS"]:
        print("  contiguous:  ☑ Yes  ☐ No")
    else:
        print("  contiguous:  ☐ Yes  ☑ No")
    if Z.flags["ALIGNED"]:
        print("  aligned:     ☑ Yes  ☐ No")
    else:
        print("  aligned:     ☐ Yes  ☑ No")
    print("------------------------------")
    print()
