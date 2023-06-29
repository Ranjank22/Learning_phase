# Groundwater flow in a confined aquifer may be simulated by the following vertically integrated 2-
# dimensional equation:
# 𝑆*(𝜕ℎ/𝜕𝑡) =∂ (𝑇𝑥*𝜕ℎ/𝜕𝑥)/∂𝑥+∂ (𝑇𝑦*𝜕ℎ/𝜕𝑦)/∂𝑦
# where, S is the storage coefficient and, Tx and Ty are the transmissivities. The size of the aquifer is 2 km in each direction. Boundary conditions are: h(0, y) = 0; 
# 𝜕ℎ/𝜕𝑥 (2, y) = 0; 
# h(x,0) = 0, 
# h(x,2) = sin (3π*x/4)
# (a) Develop a computer code for the numerical solution of the steady state problem using 2nd order central difference approximations and delta_x = delta_y= 0.2 km. 
# Assume the aquifer to be homogenous and isotropic with hydraulic conductivity of 5×10^-4 m/s and a uniform thickness of 22 m. 
# Verify your code by comparing your solution at the grid points with the analytical solution given by
# ℎ(𝑥, 𝑦) = sin(3𝜋𝑥/4)*sinh (3𝜋𝑦/4)/ sinh (3𝜋/2)



#%%
import numpy as np

# Aquifer properties
S = 0.2  # storage coefficient
K = 5e-4  # hydraulic conductivity
#-++T = K/0.2  # transmissivity

# Grid parameters
L = 2  # length of aquifer in x-direction
W = 2  # length of aquifer in y-direction
dx = dy = 0.2  # grid spacing

# Create grid
x = np.arange(0, L+dx/2, dx)
y = np.arange(0, W+dy/2, dy)

# x = np.arange(0, L+dx, dx)
# y = np.arange(0, W+dy, dy)
X, Y = np.meshgrid(x, y)

# Calculate the transmissivity in each direction
Tx = 1
Ty = 1

# Initialize solution array
h = np.zeros_like(X)

# Boundary conditions
h[0, :] = 0                                           # h(0, y) = 0
h[:, -1] = np.sin(3*np.pi*np.arange(0, L+dy, dy)/4)   # h(x,2) = sin (3πx/4)
h[:, 0] = 0                                           # h(x,0) = 0
h[-1, :] = h[-2, :]                                   # d/dx (h(x,2)) = 0

print(h)

# Implement the numerical solution using 2nd order central difference approximations
# and delta_x = delta_y = 0.2 km
err = 1
while err > 1e-5:
    hn = np.copy(h)
    for i in range(1, h.shape[1]-1):
        for j in range(1, h.shape[0]-1):
            h[i, j] = h[i+1,j]+h[i-1,j]+h[i,j+1]+h[i,j-1]
    err = np.max(np.abs(h-hn))

# Analytical solution
h_exact = np.sin(3*np.pi*X/4)*np.sinh(3*np.pi*Y/4)/np.sinh(3*np.pi/2)

print("\nvalue of h_exact = ", h_exact)

# Create contour plots of the numerical and analytical solutions
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
cmap = plt.cm.get_cmap('coolwarm')

# Numerical solution
c1 = axs[0].contourf(X, Y, h, cmap=cmap)
axs[0].set_title('Numerical Solution')
axs[0].set_xlabel('x (km)')
axs[0].set_ylabel('y (km)')
plt.colorbar(c1, ax=axs[0])

# Analytical solution
c2 = axs[1].contourf(X, Y, h_exact, cmap=cmap)
axs[1].set_title('Analytical Solution')
axs[1].set_xlabel('x (km)')
axs[1].set_ylabel('y (km)')
plt.colorbar(c2, ax=axs[1])

plt.tight_layout()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

# Aquifer properties
K = 5e-4  # hydraulic conductivity [m/s]
H = 22  # aquifer thickness [m]
S = 1e-4  # storage coefficient

# Grid properties
L = 2000  # length of aquifer in x and y direction [m]
dx = 200  # grid spacing in x direction [m]
dy = 200  # grid spacing in y direction [m]
nx = int(L/dx) + 1  # number of grid points in x direction
ny = int(L/dy) + 1  # number of grid points in y direction
x = np.linspace(0, L, nx)  # x coordinates of grid points
y = np.linspace(0, L, ny)  # y coordinates of grid points
X, Y = np.meshgrid(x, y)  # meshgrid of x and y coordinates

# Time properties
t = 0  # initial time
dt = 50  # time step [s]
tmax = 100000  # maximum simulation time [s]

# Initialize head
h = np.zeros((ny, nx))  # array of head values

# Set boundary conditions
h[0, :] = 0  # h(0, y) = 0
h[:, -1] = np.sin(3*np.pi*np.arange(0, L+dy, dy)/4)   # h(x,2) = sin (3πx/4)

#h[-1, :] = np.sin(3*np.pi*np.arange(0, L+dx, dx)/4)  # h(x,2) = sin (3πx/4)
h[:, 0] = 0  # h(x,0) = 0
h[-1, :] = h[-2, :]  # d/dx (h(x,2)) = 0
# Compute transmissivity
T = 1

# Compute coefficients
a = T/(dx**2)
b = T/(dy**2)
c = 2*S/dt

# Create matrix A
A = np.zeros((ny*nx, ny*nx))

for i in range(1, ny-1):
    for j in range(1, nx-1):
        k = i*nx + j
        A[k, k] = c + 2*a + 2*b
        A[k, k+1] = -a
        A[k, k-1] = -a
        A[k, k+nx] = -b
        A[k, k-nx] = -b

print(A)
# Time loop
while t < tmax:
    # Compute rhs
    rhs = np.zeros(ny*nx)

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            k = i*nx + j
            rhs[k] = h[i, j]*c

    # Solve the linear system
    if np.linalg.det(A) != 0:
        h_new = np.linalg.solve(A, rhs).reshape((ny, nx))
    else:
        h_new = np.linalg.lstsq(A, rhs, rcond=None)[0].reshape((ny, nx))
    # Update time and head
    t += dt
    h = h_new.copy()

# Analytical solution
h_analytical = np.sin(3*np.pi*X/4) * (np.sinh(3*np.pi*Y/4) / np.sinh(3*np.pi/2))

# Compute maximum error
error = np.max(np.abs(h - h_analytical))

# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Numerical solution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.contourf(X, Y, h, levels=50, cmap='jet')

plt.colorbar()
plt.contour(X, Y, h, levels=10, colors='k')
plt.plot([0, 0, 2, 2, 0], [0, 2, 2, 0, 0], 'k--')
plt.plot(X[:, 0], Y[:, 0], 'k-', linewidth=2)
plt.plot(X[:, -1], Y[:, -1], 'k-', linewidth=2)

plt.subplot(1, 2, 2)
plt.title('Analytical solution')
plt.xlabel('x [m]')
plt.contourf(X, Y, h_analytical, levels=50, cmap='jet')
plt.colorbar()
plt.contour(X, Y, h_analytical, levels=10, colors='k')
plt.plot([0, 0, 2, 2, 0], [0, 2, 2, 0, 0], 'k--')
plt.plot(X[:, 0], Y[:, 0], 'k-', linewidth=2)
plt.plot(X[:, -1], Y[:, -1], 'k-', linewidth=2)

plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np

# Aquifer properties
S = 0.2  # storage coefficient
K = 5e-4  # hydraulic conductivity

# Analytical solution
x = np.arange(0, 2.2, 0.2)
y = np.arange(0, 2.2, 0.2)
X, Y = np.meshgrid(x, y)
h_exact = np.sin(3*np.pi*X/4)*np.sinh(3*np.pi*Y/4)/np.sinh(3*np.pi/2)
h_exact = np.nan_to_num(h_exact)

# Grid parameters
L = 2  # length of aquifer in x-direction
W = 2  # length of aquifer in y-direction
dx = dy = 0.2  # grid spacing
N = int(L/dx)  # number of grid blocks in x-direction
M = int(W/dy)  # number of grid blocks in y-direction

# Initial conditions
h = np.zeros((M+1, N+1))
h[0, :] = 10  # top boundary
h[:, 0] = 20  # left boundary
h[:, -1] = 0  # right boundary

# Time parameters
t_end = 5000  # simulation time
dt = 100  # time step size
N_t = int(t_end/dt)  # number of time steps

# Solve for each time step
for i in range(N_t):
    # Compute hydraulic head for each grid block
    for j in range(1, M):
        for k in range(1, N):
            h[j, k] = (1-2*S*K*dt/(dx**2+dy**2))*h[j, k] + (S*K*dt/dx**2)*(h[j, k+1]+h[j, k-1]) + (S*K*dt/dy**2)*(h[j+1, k]+h[j-1, k])
    
    # Update boundary conditions
    h[0, :] = 10  # top boundary
    h[:, 0] = 20  # left boundary
    h[:, -1] = h_exact[:, -1]  # right boundary

# Create contour plots of the numerical and analytical solutions
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
cmap = plt.cm.get_cmap('coolwarm')

# Numerical solution
c1 = axs[0].contourf(X, Y, h, cmap=cmap)
axs[0].set_title('Numerical Solution')
axs[0].set_xlabel('x (km)')
axs[0].set_ylabel('y (km)')
plt.colorbar(c1, ax=axs[0])

# Analytical solution
c2 = axs[1].contourf(X, Y, h_exact, cmap=cmap)
axs[1].set_title('Analytical Solution')
axs[1].set_xlabel('x (km)')
axs[1].set_ylabel('y (km)')
plt.colorbar(c2, ax=axs[1])

plt.tight_layout()
plt.show()
