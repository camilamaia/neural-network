#!/usr/bin/env python

import numpy as np
import pandas as pd

from numpy import matrix

A = matrix( [[1,2,3],[11,12,13],[21,22,23]])
x = matrix( [[1],[2],[3]] )
y = matrix( [[1,2,3]] )

A.T                                    # Transpose of A.
A*x                                    # Matrix multiplication of A and x.
A.I                                    # Inverse of A.

df_a = pd.DataFrame(A, columns=['A', 'B', 'C'])
df_x = pd.DataFrame(x, columns=['A'])
df_y = pd.DataFrame(y, columns=['A','B','C'])

print 'A'
print df_a
print 'x'
print df_x
print 'y'
print df_y

print "A.T"
print pd.DataFrame(A.T, columns=['A', 'B', 'C'])

print "A*x"
print pd.DataFrame(A*x, columns=['A'])

print "A.I"
print pd.DataFrame(A.I, columns=['A', 'B', 'C'])

# df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])

def calculate():
  print 'oi'

if __name__ == "__main__":
  calculate()
