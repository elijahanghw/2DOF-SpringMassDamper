import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from control.matlab import *

## Mass, damping and stiffness properties
m1 = 10
m2 = 1

k1 = 100
k2 = 10

c1 = 0
c2 = 0


## Mass, damping and stiffness matrix
M = np.array([[m1, 0],
              [0, m2]])

K = np.array([[k1+k2, -k2],
              [-k2, k2]])

C = np.array([[c1+c2, -c2],
              [-c2, c2]])


## State-space formulation
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [-(k1+k2)/m1, k2/m1, -(c1+c2)/m1, c2/m1],
              [k2/m2, -k2/m2, c2/m2, -c2/m2]])

B = np.array([[0, 0],
              [0, 0],
              [1/m1, 0],
              [0, 1/m2]])


C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

D = np.array([[0, 0],
              [0, 0]])

sys = ss(A, B, C, D)


plt.figure(1)
yout, T = step(sys)
plt.plot(T.T, yout.T[0][0])
plt.plot(T.T, yout.T[0][1])
plt.xlim((0,10))
plt.show()