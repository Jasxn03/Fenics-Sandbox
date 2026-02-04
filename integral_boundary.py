import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx.io import XDMFFile

comm = MPI.COMM_WORLD
mu = 1.0
tau_wall = 1500.0


with XDMFFile(comm, "Biofilm Meshes/uneven_biofilm_mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="mesh")
    ft = xdmf.read_meshtags(mesh, name="Facet markers")
    ct = xdmf.read_meshtags(mesh, name="Cell tags")

inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5

fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, 0)
geometry = mesh.geometry.x

def facet_nodes_coords(facet):
    vertices = mesh.topology.connectivity(fdim, 0).links(facet)
    return geometry[vertices]

boundary_nodes = []
boundary_normals = []
ds = []
facet_id_list = []

for f, val in zip(ft.indices, ft.values):
    if val in [wall_marker, obstacle_marker, inlet_marker, outlet_marker]:
        x0, x1 = facet_nodes_coords(f)
        xm = 0.5 * (x0 +x1)
        t = x1 - x0
        length = np.linalg.norm(t)
        n = np.array([-t[1], t[0]]) / length
        boundary_nodes.append(xm)
        boundary_normals.append(n)
        ds.append(length)
        facet_id_list.append(val)

boundary_nodes = np.array(boundary_nodes)
boundary_nodes = boundary_nodes[:,:2]  # take x and y only
boundary_normals = np.array(boundary_normals)
ds = np.array(ds)
facet_id_list = np.array(facet_id_list)
N = len(boundary_nodes)

top_nodes = np.where(facet_id_list == wall_marker)[0]        # Neumann (shear)
bottom_nodes = np.where(facet_id_list == obstacle_marker)[0] # Dirichlet (no-slip)
left_nodes = np.where(facet_id_list == inlet_marker)[0]
right_nodes = np.where(facet_id_list == outlet_marker)[0]

left_nodes = left_nodes[np.argsort(boundary_nodes[left_nodes,1])]
right_nodes = right_nodes[np.argsort(boundary_nodes[right_nodes,1])]
Np = len(left_nodes)


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


Tvec = np.zeros((N,2))
for i in top_nodes:
    n = boundary_normals[i]
    t_hat = np.array([-n[1], n[0]])
    Tvec[i] = tau_wall * t_hat


Nu = len(top_nodes)
Nd = len(bottom_nodes)
unknowns = 2*(Nu + Nd)
A = np.zeros((unknowns, unknowns))
b = np.zeros(unknowns)
extra_row = np.zeros(A.shape[1])

right_to_left = {r: left_nodes[i] for i,r in enumerate(right_nodes)}


for i_local, i in enumerate(top_nodes):
    xi = boundary_nodes[i]
    # Single-layer: known traction (top wall)
    for j in top_nodes:
        if j == i:
            continue
        b[2*i_local:2*i_local+2] += (
            stokeslet_2d(xi, boundary_nodes[j], mu)
            @ Tvec[j] * ds[j]
        )
    A[2*i_local:2*i_local+2, 2*i_local:2*i_local+2] -= 0.5 * np.eye(2)

        # Double-layer: unknown velocities (top wall)
    for j_local, j in enumerate(top_nodes):
        if j == i:
            continue
        Tij = stresslet_2d(xi, boundary_nodes[j], boundary_normals[j])
        A[2*i_local:2*i_local+2, 2*j_local:2*j_local+2] -= Tij * ds[j]
    # Double-layer: left/right periodic nodes
    # for k_local, k in enumerate(left_nodes):
    #     A[2*i_local:2*i_local+2, 2*(Nu+Nd+k_local):2*(Nu+Nd+k_local)+2] -= stresslet_2d(xi, boundary_nodes[k], boundary_normals[k]) * ds[k]

# --- Dirichlet nodes: bottom wall
for i_local, i in enumerate(bottom_nodes):
    A[2*(Nu+i_local):2*(Nu+i_local)+2, 2*(Nu+i_local):2*(Nu+i_local)+2] = np.eye(2)
    b[2*(Nu+i_local):2*(Nu+i_local)+2] = [0.0, 0.0]


u_unknown, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
# Extract velocities on top wall
U_top = u_unknown[:2*Nu].reshape(Nu,2)
# Tractions on bottom wall
F_bottom = u_unknown[2*Nu:2*(Nu+Nd)].reshape(Nd,2)
# Velocities on left (periodic)
# U_left = u_unknown[2*(Nu+Nd):].reshape(Np,2)


def velocity_at_point(x):
    u = np.zeros(2)

    # Single-layer from top (Neumann)
    for i in top_nodes:
        u += stokeslet_2d(x, boundary_nodes[i], mu) @ Tvec[i] * ds[i]

    # Double-layer from bottom (Dirichlet)
    for i_local, i in enumerate(bottom_nodes):
        u -= stresslet_2d(
            x,
            boundary_nodes[i],
            boundary_normals[i]
        ) @ np.array([0.0, 0.0]) * ds[i]
    
    for i_local, i in enumerate(top_nodes):
        u -= stresslet_2d(
            x,
            boundary_nodes[i],
            boundary_normals[i]
        ) @ U_top[i_local] * ds[i]

    return u


# Grid evaluation
nx, ny = 40, 20
xs = np.linspace(0.05, 50-0.05, nx)
ys = np.linspace(0.05, 20-0.05, ny)
Ugrid = np.zeros((ny, nx, 2))

for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        Ugrid[i,j] = velocity_at_point(np.array([x,y]))

Xg, Yg = np.meshgrid(xs, ys)
speed = np.linalg.norm(Ugrid, axis=2)

# plt.figure(figsize=(6,3))
# plt.quiver(Xg, Yg, Ugrid[:,:,0], Ugrid[:,:,1], angles='xy', scale_units='xy', scale=1)
# plt.xlabel("x"); plt.ylabel("y")
# plt.title("Velocity field (Neumann + Dirichlet + Periodic)")
# plt.axis("equal")
# plt.show()

plt.figure(figsize=(6,3))
plt.contourf(Xg, Yg, speed, levels=30)
plt.colorbar(label="|u|")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Velocity magnitude")
plt.axis("equal")
plt.show()

