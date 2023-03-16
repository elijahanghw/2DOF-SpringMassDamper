import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

## Mass, damping and stiffness properties
m1 = 10
m2 = 1

k1 = 100
k2 = 10

c1 = 0
c2 = 0

## Simulation properties
dt = 0.01
timesteps = 1000

## Mass, damping and stiffness matrix
M = np.array([[m1, 0],
              [0, m2]])

K = np.array([[k1+k2, -k2],
              [-k2, k2]])

C = np.array([[c1+c2, -c2],
              [-c2, c2]])


## Newmark-beta
alpha = 0.0
phi = 0.5 + alpha
beta = 0.25*(phi+0.5)**2

S_1 = M + phi*dt*C + dt**2*beta*K
S_2 = (1-phi)*dt*C + dt**2*(0.5-beta)*K

N_11 = np.concatenate((np.zeros_like(M), np.zeros_like(M), S_1), axis=1)
N_12 = np.concatenate((np.zeros_like(M), -np.eye(2), dt*phi*np.eye(2)), axis=1)
N_13 = np.concatenate((-np.eye(2), np.zeros_like(M), dt**2*beta*np.eye(2)), axis=1)

N_1 = np.concatenate((N_11, N_12, N_13), axis = 0)

N_21 = np.concatenate((K, C+K*dt, S_2), axis=1)
N_22 = np.concatenate((np.zeros_like(M), np.eye(2), (1-phi)*dt*np.eye(2)), axis=1)
N_23 = np.concatenate((np.eye(2), dt*np.eye(2), (0.5-beta)*dt**2*np.eye(2)), axis=1)

N_2 = np.concatenate((N_21, N_22, N_23), axis = 0)

## State-space formulation
A = np.matmul(inv(N_1), -N_2)
B = np.array([[1, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]])

B = np.matmul(inv(N_1), B)

C = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0]])

D = np.array([[0, 0],
              [0, 0]])

## Step inputs
U = np.array([[1],
              [1]])

## Time stepping
time = []
output_1 = []
output_2 = []
x_old = np.zeros((6,1))
output_1.append(np.matmul(C, x_old)[0])
output_2.append(np.matmul(C, x_old)[1])

time.append(0)

for t in range(timesteps):
    x_new = np.matmul(A, x_old) + np.matmul(B, U)
    y = np.matmul(C, x_new)
    output_1.append(y[0])
    output_2.append(y[1])
    time.append((t+1)*dt)
    x_old = x_new
 

plt.plot(time, output_1)
plt.plot(time, output_2)
plt.xlim((0,10))

plt.show()
