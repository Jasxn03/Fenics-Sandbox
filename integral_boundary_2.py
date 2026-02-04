import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Geometry
# ----------------------
L, H = 50.0, 20.0
Nx, Ny = 50, 20  # boundary nodes along top/bottom
tau_wall = 1500.0
mu = 1.0

# Bottom wall: Dirichlet u=0
x_bottom = np.linspace(0, L, Nx)
y_bottom = np.zeros_like(x_bottom)
bottom_nodes = np.stack([x_bottom, y_bottom], axis=1)

# Top wall: Neumann tau = tau_wall (tangential)
x_top = np.linspace(0, L, Nx)
y_top = H*np.ones_like(x_top)
top_nodes = np.stack([x_top, y_top], axis=1)

# Normals
n_bottom = np.tile(np.array([0,1]), (Nx,1))  # pointing up
n_top    = np.tile(np.array([0,-1]), (Nx,1)) # pointing down

# ds (segment lengths)
ds_bottom = np.full(Nx, L/(Nx-1))
ds_top    = np.full(Nx, L/(Nx-1))

Nu = Nx
Nd = Nx
unknowns = 2*(Nu + Nd)

# ----------------------
# BIE functions
# ----------------------
def stokeslet_2d(x, y, mu):
    r = x - y
    r2 = np.dot(r,r) + 1e-12
    I = np.eye(2)
    G = -np.log(np.sqrt(r2)) * I + np.outer(r,r) / r2
    return G / (4*np.pi*mu)

def stresslet_2d(x, y, n):
    r = x - y
    r2 = np.dot(r,r) + 1e-12
    rdotn = np.dot(r,n)
    T = (-1.0/np.pi) * np.outer(r,r) * rdotn / (r2**2)
    return T

# ----------------------
# RHS traction vector (top wall)
# ----------------------
Tvec_top = np.zeros((Nu,2))
t_hat = np.array([1,0])  # tangential along x
for i in range(Nu):
    Tvec_top[i] = tau_wall * t_hat

# ----------------------
# Assemble BIE matrix
# Unknowns: [u_top, f_bottom]
# ----------------------
A = np.zeros((unknowns, unknowns))
b = np.zeros(unknowns)

# --- Top wall (Neumann) ---
for i in range(Nu):
    xi = top_nodes[i]

    # Single-layer: known traction from other top nodes
    for j in range(Nu):
        if j == i:
            continue
        G = stokeslet_2d(xi, top_nodes[j], mu)
        b[2*i:2*i+2] += G @ Tvec_top[j] * ds_top[j]

    # Double-layer: unknown velocities (top)
    for j in range(Nu):
        Tij = stresslet_2d(xi, top_nodes[j], n_top[j])
        if i == j:
            # self-term for Neumann: 0
            continue
        A[2*i:2*i+2, 2*j:2*j+2] -= Tij * ds_top[j]

    # Double-layer: bottom Dirichlet nodes (unknown tractions)
    for j in range(Nd):
        Tij = stresslet_2d(xi, bottom_nodes[j], n_bottom[j])
        A[2*i:2*i+2, 2*Nu + 2*j : 2*Nu + 2*j+2] -= Tij * ds_bottom[j]

# --- Bottom wall (Dirichlet) ---
for i in range(Nd):
    # Self-contribution: double-layer 0.5 * I
    A[2*Nu + 2*i: 2*Nu + 2*i+2, 2*Nu + 2*i: 2*Nu + 2*i+2] = 0.5*np.eye(2)

    # Single-layer: top wall traction
    for j in range(Nu):
        G = stokeslet_2d(bottom_nodes[i], top_nodes[j], mu)
        A[2*Nu + 2*i:2*Nu + 2*i+2, 2*j:2*j+2] += G * ds_top[j]

    # RHS: prescribed velocity = 0
    b[2*Nu + 2*i:2*Nu + 2*i+2] = [0.0, 0.0]

# ----------------------
# Solve
# ----------------------
u_unknown, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

U_top = u_unknown[:2*Nu].reshape(Nu,2)
F_bottom = u_unknown[2*Nu:].reshape(Nd,2)

# ----------------------
# Evaluate velocity on grid
# ----------------------
nx, ny = 50, 20
xs = np.linspace(0, L, nx)
ys = np.linspace(0, H, ny)
Ugrid = np.zeros((ny, nx, 2))

for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        u = np.zeros(2)
        x_point = np.array([x,y])

        # Single-layer: top wall traction
        for k in range(Nu):
            u += stokeslet_2d(x_point, top_nodes[k], mu) @ Tvec_top[k] * ds_top[k]

        # Double-layer: top velocities
        for k in range(Nu):
            u -= stresslet_2d(x_point, top_nodes[k], n_top[k]) @ U_top[k] * ds_top[k]

        # Double-layer: bottom tractions
        for k in range(Nd):
            u -= stresslet_2d(x_point, bottom_nodes[k], n_bottom[k]) @ F_bottom[k] * ds_bottom[k]

        Ugrid[i,j] = u

# ----------------------
# Plot
# ----------------------
Xg, Yg = np.meshgrid(xs, ys)
speed = np.linalg.norm(Ugrid, axis=2)
print("Max velocity magnitude on grid:", np.max(speed))


plt.figure(figsize=(8,3))
plt.contourf(Xg, Yg, speed, levels=30)
plt.colorbar(label="|u|")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Velocity magnitude (rectangular channel)")
plt.axis("equal")
plt.show()
