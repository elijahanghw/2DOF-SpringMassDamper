import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

## Mass, damping and stiffness properties
M = 1


K = 10


C = 1


## Simulation properties
dt = 0.01
timesteps = 1000


## Newmark-beta
alpha = 0.005
phi = 0.5 + alpha
beta = 0.25*(phi+0.5)**2

S_1 = M + phi*dt*C + dt**2*beta*K
S_2 = (1-phi)*dt*C + dt**2*(0.5-beta)*K

N_1 = np.array([[0, 0, S_1],
                [0, -1, dt*phi],
                [-1, 0, dt**2*beta*1]])

N_2 = np.array([[K, C+K*dt, S_2],
                [0, 1, (1-phi)*dt],
                [1, dt, (0.5-beta)*dt**2]])

## State-space formulation
A = np.matmul(inv(N_1), -N_2)
B = np.array([[1],
              [0],
              [0]])
B = np.matmul(inv(N_1), B)

C = np.array([[1, 0, 0]])

D = np.array([[0]])

## Step inputs
U = 1

## Time stepping
time = []
output_1 = []
x_old = np.zeros((3,1))
output_1.append(np.matmul(C, x_old))

time.append(0)

for t in range(timesteps):
    x_new = np.matmul(A, x_old) + B*U
    y = np.matmul(C, x_new)
    output_1.append(y[0])
    time.append((t+1)*dt)
    x_old = x_new
 

plt.plot(time, output_1)

plt.show()
