# Terminal visualization of 2D numpy arrays
# Copyright (c) 2009  Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------
""" Terminal visualization of 2D numpy arrays
    Using extended color capability of terminal (256 colors), the imshow function
    renders a 2D numpy array within terminal.
"""
import sys
import numpy as np
from matplotlib.cm import viridis


def imshow (Z, vmin=None, vmax=None, cmap=viridis, show_cmap=True):
    ''' Show a 2D numpy array using terminal colors '''

    if len(Z.shape) != 2:
        print("Cannot display non 2D array")
        return

    vmin = vmin or Z.min()
    vmax = vmax or Z.max()

    # Build initialization string that setup terminal colors
    init = ''
    for i in range(240):
        v = i/240 
        r,g,b,a = cmap(v)
        init += "\x1b]4;%d;rgb:%02x/%02x/%02x\x1b\\" % (16+i, int(r*255),int(g*255),int(b*255))

    # Build array data string
    data = ''
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            c = 16 + int( ((Z[Z.shape[0]-i-1,j]-vmin) / (vmax-vmin))*239)
            if (c < 16):
                c=16
            elif (c > 255):
                c=255
            data += "\x1b[48;5;%dm  " % c
            u = vmax - (i/float(Z.shape[0]-1)) * ((vmax-vmin))
        if show_cmap:
            data += "\x1b[0m  "
            data += "\x1b[48;5;%dm  " % (16 + (1-i/float(Z.shape[0]))*239)
            data += "\x1b[0m %+.2f" % u
        data += "\n"

    sys.stdout.write(init+'\n')
    sys.stdout.write(data+'\n')


if __name__ == '__main__':
    def func3(x,y):
        return (1- x/2 + x**5 + y**3)*np.exp(-x**2-y**2)
    dx, dy = .2, .2
    x = np.arange(-3.0, 3.0, dx)
    y = np.arange(-3.0, 3.0, dy)
    X,Y = np.meshgrid(x, y)
    Z = np.array (func3(X, Y))
    imshow (Z)
