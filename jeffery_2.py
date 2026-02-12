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



