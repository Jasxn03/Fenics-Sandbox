import numpy as np
import matplotlib.pyplot as plt

# Parameters
U = 1500.0
a = 1.0
mu = 1.0

# Grid
nx, ny = 500, 500
x_vals = np.linspace(-5, 5, nx)
y_vals = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

# Velocity arrays
U_grid = np.zeros_like(X)
V_grid = np.zeros_like(X)
W_grid = np.zeros_like(X)

for i in range(nx):
    for j in range(ny):
        x, y, z = X[j,i], Y[j,i], Z[j,i]
        r = np.sqrt(x**2 + y**2 + z**2)
        if r <= a:
            # inside sphere
            U_grid[j,i] = 0
            V_grid[j,i] = 0
            W_grid[j,i] = 0
        else:
            # Stokes flow past sphere
            u_x = U * (1 - 3*a/(4*r) * (1 + x**2/r**2) + a**3/(4*r**3) * (1 - 3*x**2/r**2)) * y
            u_y = U * (-3*a/(4*r) * (x*y/r**2) - 3*a**3/(4*r**3) * (x*y/r**2))
            u_z = U * (-3*a/(4*r) * (x*z/r**2) - 3*a**3/(4*r**3) * (x*z/r**2))
            U_grid[j,i] = u_x
            V_grid[j,i] = u_y
            W_grid[j,i] = u_z

# Velocity magnitude
speed = np.sqrt(U_grid**2 + V_grid**2 + W_grid**2)

# Plot velocity magnitude
plt.figure(figsize=(6,5))
plt.contourf(X, Y, speed, levels=50, cmap='inferno')
plt.colorbar(label='Velocity magnitude')
#plt.streamplot(X, Y, U_grid, V_grid, color='white', density=1.2)
theta = np.linspace(0, 2*np.pi, 400)
plt.plot(a*np.cos(theta), a*np.sin(theta), 'w', linewidth=2)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stokes Flow Past Sphere (z=0 slice)')
plt.show()

# Function to compute surface normal for sphere
def surface_normal_sphere(x, y, z, a):
    n = np.array([x, y, z]) / a
    return n / np.linalg.norm(n)

# Finite difference to compute velocity gradient
def velocity_gradient_fd(x, y, z, h=1e-6):
    u0 = np.array([U_interp(x, y, z) for U_interp in [U_grid_func, V_grid_func, W_grid_func]])
    grad = np.zeros((3,3))
    for i, (dx, dy, dz) in enumerate([(h,0,0),(0,h,0),(0,0,h)]):
        u_h = np.array([U_interp(x+dx, y+dy, z+dz) for U_interp in [U_grid_func, V_grid_func, W_grid_func]])
        grad[:,i] = (u_h - u0)/h
    return grad

# Interpolation functions for velocity grids
from scipy.interpolate import RegularGridInterpolator
U_grid_func = RegularGridInterpolator((y_vals, x_vals), U_grid)
V_grid_func = RegularGridInterpolator((y_vals, x_vals), V_grid)
W_grid_func = RegularGridInterpolator((y_vals, x_vals), W_grid)

# Compute shear on the sphere surface in xy-plane (z=0)
n_theta = 400
theta = np.linspace(0, 2*np.pi, n_theta)
shear_surface = np.zeros(n_theta)
x_surf = a * np.cos(theta)
y_surf = a * np.sin(theta)
z_surf = np.zeros_like(theta)

for i in range(n_theta):
    x, y, z = x_surf[i], y_surf[i], z_surf[i]
    n_vec = surface_normal_sphere(x, y, z, a)
    grad_u = velocity_gradient_fd(x, y, z)
    sigma = mu * (grad_u + grad_u.T)
    traction = sigma @ n_vec
    shear_vec = traction - np.dot(traction, n_vec) * n_vec
    shear_surface[i] = np.linalg.norm(shear_vec)

# Arc length along circle
ds = 2*np.pi*a / n_theta
s = np.arange(n_theta) * ds

# Plot shear
plt.figure(figsize=(7,4))
plt.plot(s, shear_surface, lw=2)
plt.xlabel('Arc length along circle')
plt.ylabel('Surface shear stress')
plt.title('Surface Shear Stress on Sphere (z=0 plane)')
plt.grid(True)
plt.show()

