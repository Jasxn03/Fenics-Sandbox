import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element

from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
    Expression,
)
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import gmsh as gmshio, XDMFFile
from ufl import (
    FacetNormal,
    TestFunctions,
    TrialFunctions,
    div,
    dx,
    inner,
    grad,
    Identity,
    SpatialCoordinate,
    MixedFunctionSpace,
    TestFunction,
    TrialFunction,
    sym
)
from dolfinx.plot import vtk_mesh
from dolfinx import default_scalar_type
import pyvista

from dolfinx_mpc import (
    MultiPointConstraint,
    LinearProblem
)

import dolfinx.la.petsc
from dolfinx import plot
import typing
from dolfinx.mesh import meshtags

# endregion

L = 50
H = 20
with XDMFFile(MPI.COMM_WORLD, "Biofilm Meshes/uneven_biofilm_mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name = "mesh")
    ft = xdmf.read_meshtags(mesh, name="Facet markers")
    ct = xdmf.read_meshtags(mesh, name="Cell tags")

inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5 # need to write these three lines so it knows what markers are what
fdim = mesh.topology.dim - 1 
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

# region Boundary Conditions 

Q0 = functionspace(mesh, ("DG", 0))
mu_field = Function(Q0)

mu = Constant(mesh, PETSc.ScalarType(0.001))  
mu_bf = Constant(mesh, PETSc.ScalarType(1000))

mu_field.x.array[:] = 1e-3
biofilm_cells = ct.find(6)  
mu_field.x.array[biofilm_cells] = 1000

rho = Constant(mesh, PETSc.ScalarType(1)) 
#f = Constant(mesh, (0.0, -9.81))
f = Constant(mesh, (0.0,0.0))

v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)
Z = MixedFunctionSpace(V,Q)
x = SpatialCoordinate(mesh)
n = FacetNormal(mesh)

fdim = mesh.topology.dim - 1

# shear velocity 
def shear_velocity_f(x):
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0, :] = 1500 
    values[1, :] = 0.0
    return values
u_wall = Function(V)
u_wall.interpolate(shear_velocity_f)
bc_shear_velocity = dirichletbc(
    u_wall, locate_dofs_topological(V, fdim, ft.find(wall_marker))
)

# biofilm
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_obstacle = dirichletbc(
    u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V
)
             
bcu = bc_shear_velocity, bcu_obstacle

coords = mesh.geometry.x
x_target = L / 2
y_target = 0
dist2 = (coords[:, 0] - x_target)**2 + (coords[:, 1] - y_target)**2
dof_p_ref = np.array([np.argmin(dist2)], dtype=np.int32)
bcp_ref = dirichletbc(PETSc.ScalarType(0), dof_p_ref, Q)

bcup = [bcu, bcp_ref]

def periodic_boundary(x):
    return (x[0] == 0.0) | (x[0] == L)

def periodic_relation(x):
    out_x = np.zeros_like(x)
    out_x[0] = 0  
    out_x[1] = x[1] 
    return out_x

mpc_V = MultiPointConstraint(V)
mpc_V.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, bcu)
mpc_V.finalize()
mpc_p = MultiPointConstraint(Q)
mpc_p.create_periodic_constraint_geometrical(Q, periodic_boundary, periodic_relation, [bcp_ref]) 
mpc_p.finalize()
mpc = [mpc_V, mpc_p]

# endregion

# region Solver
(u, p) = TrialFunctions(Z)
(v, q) = TestFunctions(Z)

# a = [[mu * inner(grad(u), grad(v)) * dx, -inner(p, div(v)) * dx],
#     [ inner(div(u), q) * dx, None]]

a = [[
    2 * inner(mu_field * sym(grad(u)), sym(grad(v))) * dx,
    -inner(p, div(v)) * dx
],[
    inner(div(u), q) * dx,
    None
]]


L = [inner(f, v) * dx,
    Constant(mesh, PETSc.ScalarType(0)) * q * dx]

from pathlib import Path

problem = LinearProblem(
    a,
    L,
    bcs=[bc_shear_velocity, bcu_obstacle, bcp_ref],
    mpc=[mpc_V, mpc_p],
    petsc_options={"ksp_type": "gmres", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
)
u_sol, p_sol = problem.solve()

u_sol.x.scatter_forward()
p_sol.x.scatter_forward()

u_full = Function(V)
mpc_V.backsubstitution(u_sol)  

# Pressure
p_full = Function(Q)
mpc_p.backsubstitution(p_sol)

print("Max velocity:", np.max(u_sol.x.array))

Vdim = mesh.geometry.dim
u_vals = u_sol.x.array.reshape((-1, Vdim))
biofilm_dofs = locate_dofs_topological(V, mesh.topology.dim, biofilm_cells)

print("Max |u| in biofilm:",
      np.linalg.norm(u_vals[biofilm_dofs], axis=1).max())

print("simulation finished")

# endregion 

# region Plotting

# u plotting
# topology, cell_types, geometry = vtk_mesh(V)
# grid_u = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# u_2d = u_sol.x.array.reshape((-1, mesh.geometry.dim))
# u_3d = np.zeros((u_2d.shape[0], 3), dtype=u_2d.dtype)
# u_3d[:, :2] = u_2d
# grid_u["u"] = u_3d
# grid_u["|u|"] = np.linalg.norm(u_2d, axis=1)

# plotter = pyvista.Plotter(off_screen=True)
# plotter.add_mesh(grid_u,scalars="|u|",cmap="viridis",show_edges=False)

# glyphs = grid_u.glyph(orient="u",scale=False,factor=0.05)
# plotter.add_mesh(glyphs, color="black")
# plotter.add_title("Velocity field")
# plotter.view_xy()
# folder = "steady_stokes_results"
# os.makedirs(folder, exist_ok=True)
# plotter.screenshot(f"{folder}/uneven_velocity_field.png")
# plotter.close()


# cell_tags = ct.values
# grid_u.cell_data['cell_tags'] = cell_tags
# biofilm_tag = 6

# fluid_grid = grid_u.extract_cells(grid_u.cell_data["cell_tags"] != biofilm_tag)
# biofilm_grid = grid_u.extract_cells(grid_u.cell_data["cell_tags"] == biofilm_tag)
# plotter = pyvista.Plotter(off_screen=True)

# # Fluid: colored by velocity magnitude
# plotter.add_mesh(
#     fluid_grid,
#     scalars="|u|",
#     cmap="viridis",
#     show_edges=False
# )

# # Biofilm: solid white
# plotter.add_mesh(
#     biofilm_grid,
#     color="white",
#     show_edges=True
# )

# # Optional: velocity glyphs only in fluid
# glyphs = fluid_grid.glyph(orient="u", scale=False, factor=0.05)
# plotter.add_mesh(glyphs, color="black")

# plotter.add_title("Velocity field (biofilm masked)")
# plotter.view_xy()
# plotter.screenshot(f"{folder}/uneven_velocity_field_masked.png")
# plotter.close()


# # p plotting
# topology_p, cell_types_p, geometry_p = vtk_mesh(Q)
# grid_p = pyvista.UnstructuredGrid(topology_p, cell_types_p, geometry_p)
# grid_p["p"] = p_sol.x.array

# plotter = pyvista.Plotter(off_screen=True)
# plotter.add_mesh(grid_p,scalars="p",cmap="coolwarm",show_edges=False)
# plotter.add_title("Pressure field")
# plotter.view_xy()
# folder = "steady_stokes_results"
# os.makedirs(folder, exist_ok=True)
# plotter.screenshot(f"{folder}/uneven_pressure_field.png")
# plotter.close()

print("plotting finished")

# endregion

# region stress
dim = mesh.geometry.dim
I = Identity(dim)
strain_rate = sym(grad(u_sol))
sigma_expr = -p_sol*I + 2.0*mu_field*strain_rate

tensor_el = element("Lagrange", mesh.basix_cell(), 1, shape=(dim, dim))
T = functionspace(mesh, tensor_el)
stress = Function(T)

sigma_trial = TrialFunction(T)
w = TestFunction(T)

a_proj = inner(sigma_trial, w) * dx
L_proj = inner(sigma_expr, w) * dx

stress_problem = dolfinx.fem.petsc.LinearProblem(
    a_proj,
    L_proj,
    bcs=[],  # no Dirichlet BCs for stress
    petsc_options={"ksp_type": "gmres", "pc_type": "lu"},
    petsc_options_prefix="solver_"
)
stress = stress_problem.solve()

stress_coords = T.tabulate_dof_coordinates() #this line is new

stress_vals = stress.x.array.reshape((-1, dim, dim))
stress_magnitude = np.linalg.norm(stress_vals, axis=(1,2))

x = stress_coords[:, 0] #these three lines are new
y = stress_coords[:, 1]
s = stress_magnitude

#-----------------------------------------------------------------------------
# i am trying to figure out what is wrong




# Get cell-to-dof connectivity for stress field
dim = mesh.geometry.dim
fdim = dim -1
mesh.topology.create_connectivity(dim, fdim)
mesh.topology.create_connectivity(fdim, dim)
mesh.topology.create_connectivity(dim, 0)
mesh.topology.create_connectivity(0, dim)
cell_to_dof = T.dofmap.cell_dofs
cell_to_facet = mesh.topology.connectivity(dim, fdim)
facet_to_cell = mesh.topology.connectivity(fdim, dim)
cell_to_vertex = mesh.topology.connectivity(dim, 0)
vertex_to_cell = mesh.topology.connectivity(0, dim)

# cells
biofilm_cells = ct.find(6)
fluid_cells = ct.find(1)
neighbouring_biofilm_cells = set()

# Compute average stress magnitude per cell
stress_mag_cell = np.zeros(len(ct.values))
for c in range(len(ct.values)):
    dofs = cell_to_dof(c)
    stress_mag_cell[c] = np.max(stress_magnitude[dofs])

# Threshold
threshold = 0.15
high_stress_biofilm = [c for c in biofilm_cells if stress_mag_cell[c] > threshold]
high_stress_fluid = [c for c in fluid_cells if stress_mag_cell[c] > threshold]


for c in high_stress_fluid:
    vertices = cell_to_vertex.links(c)
    for v in vertices:
        neighbours = vertex_to_cell.links(v)
        for n in neighbours:
            if n in biofilm_cells:
                neighbouring_biofilm_cells.add(n)

neighbour_of_neighbour = set()

for c in neighbouring_biofilm_cells:
    vertices = cell_to_vertex.links(c)
    for v in vertices:
        neighbours = vertex_to_cell.links(v)
        for n in neighbours:
            if n in biofilm_cells:
                neighbour_of_neighbour.add(n)

all_high_stress_biofilm_cells = neighbouring_biofilm_cells.union(neighbour_of_neighbour)

print(f"Number of biofilm cells with stress > {threshold}: {len(high_stress_biofilm)}")
print(f"Number of fluid cells with stress > {threshold}: {len(high_stress_fluid)}")
print(f"Number of biofilm cells next to high-stress fluid cells: {len(neighbouring_biofilm_cells)}")
print(f"All biofilm cells that are high-stress: {len(all_high_stress_biofilm_cells)}")

# --- Prepare cell masks ---
num_cells = len(ct.values)
cell_colors = np.full(num_cells, -1, dtype=np.int32)  # -1 = default (transparent)

# high-stress fluid = 0 (black)
for c in high_stress_fluid:
    cell_colors[c] = 0

# neighboring biofilm cells = 1 (white)
for c in all_high_stress_biofilm_cells:
    cell_colors[c] = 1

# --- Attach cell data to PyVista grid ---
topology_s, cell_types_s, geometry_s = vtk_mesh(T)
grid_mask = pyvista.UnstructuredGrid(topology_s, cell_types_s, geometry_s)
grid_mask.cell_data["mask"] = cell_colors

# --- Plot ---
plotter = pyvista.Plotter(off_screen=True)







# --- Build PyVista grid from velocity solution ---
topology, cell_types, geometry = vtk_mesh(V)
grid_u = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_2d = u_sol.x.array.reshape((-1, mesh.geometry.dim))
u_3d = np.zeros((u_2d.shape[0], 3), dtype=u_2d.dtype)
u_3d[:, :2] = u_2d
grid_u["u"] = u_3d
grid_u["|u|"] = np.linalg.norm(u_2d, axis=1)

# --- Cell tags ---
grid_u.cell_data['cell_tags'] = ct.values  # mesh after removing high-stress biofilm
biofilm_tag = 6

# Extract fluid and remaining biofilm for plotting
fluid_grid = grid_u.extract_cells(grid_u.cell_data["cell_tags"] != biofilm_tag)
biofilm_grid = grid_u.extract_cells(grid_u.cell_data["cell_tags"] == biofilm_tag)

if len(all_high_stress_biofilm_cells) > 0:
    high_stress_cell_ids = np.array(list(all_high_stress_biofilm_cells), dtype=np.int32)
    high_stress_grid = grid_u.extract_cells(high_stress_cell_ids)
else:
    high_stress_grid = None
# --- Plot ---
plotter = pyvista.Plotter()

# Fluid: colored by velocity magnitude
plotter.add_mesh(
    fluid_grid,
    scalars="|u|",
    cmap="viridis",
    show_edges=False
)

# Remaining biofilm: solid white
plotter.add_mesh(
    biofilm_grid,
    color="white",
    show_edges=True
)

# High-stress biofilm (red overlay)
if high_stress_grid is not None:
    plotter.add_mesh(
        high_stress_grid,
        color="red",
        show_edges=True
    )

# Optional: velocity glyphs only in fluid
glyphs = fluid_grid.glyph(orient="u", scale=False, factor=0.05)
plotter.add_mesh(glyphs, color="black")

plotter.add_title("Velocity field (high-stress biofilm in red)")
plotter.view_xy()
plotter.show()

















# --- Step 1: Update cell tags ---
biofilm_tag = 6
fluid_tag = 1

ct_values = ct.values.copy()
for c in all_high_stress_biofilm_cells:
    ct_values[c] = fluid_tag

new_ct = meshtags(mesh, mesh.topology.dim, ct.indices, ct_values)

# --- Step 2: Update viscosity field ---
biofilm_cells_new = np.setdiff1d(biofilm_cells, list(all_high_stress_biofilm_cells))
mu_field.x.array[:] = mu  # default fluid viscosity
mu_field.x.array[biofilm_cells_new] = mu_bf  # biofilm cells


# --- Step 3: Update facet tags ---
# For facets, you need to check if they were previously on the biofilm boundary
# If both cells on a facet are now fluid, remove the "obstacle" tag
fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)
mesh.topology.create_connectivity(mesh.topology.dim, fdim)
cell_to_facet = mesh.topology.connectivity(mesh.topology.dim, fdim)
facet_to_cell = mesh.topology.connectivity(fdim, mesh.topology.dim)

# Find all facets that are now between fluid-fluid (formerly fluid-biofilm)
from dolfinx.mesh import meshtags

# Make a copy of facet values
ft_values = ft.values.copy()
ft_indices = ft.indices

# Suppose you want to change the facets that belonged to biofilm cells
# You need the facets linked to the biofilm cells
biofilm_facets = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim-1)
facets_to_update = set()
for c in all_high_stress_biofilm_cells:
    facets_to_update.update(biofilm_facets.links(c))

# Set new facet tag (e.g., free-slip or fluid boundary)
mask = np.isin(ft_indices, list(facets_to_update))
fluid_facet_tag = 2  # example
ft_values[mask] = fluid_facet_tag

# Create new MeshTags
new_ft = meshtags(mesh, mesh.topology.dim-1, ft.indices, ft_values)


# --- Step 4: Recompute mesh connectivity ---
mesh.topology.create_connectivity(fdim, mesh.topology.dim)
mesh.topology.create_connectivity(mesh.topology.dim, fdim)

problem = LinearProblem(
    a,
    L,
    bcs=[bc_shear_velocity, bcu_obstacle, bcp_ref],
    mpc=[mpc_V, mpc_p],
    petsc_options={"ksp_type": "gmres", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
)
u_sol, p_sol = problem.solve()

u_sol.x.scatter_forward()
p_sol.x.scatter_forward()

u_full = Function(V)
mpc_V.backsubstitution(u_sol)  

# Pressure
p_full = Function(Q)
mpc_p.backsubstitution(p_sol)

print("Max velocity:", np.max(u_sol.x.array))

Vdim = mesh.geometry.dim
u_vals = u_sol.x.array.reshape((-1, Vdim))
biofilm_dofs = locate_dofs_topological(V, mesh.topology.dim, biofilm_cells)

print("Max |u| in biofilm:",
      np.linalg.norm(u_vals[biofilm_dofs], axis=1).max())

# u plotting
topology, cell_types, geometry = vtk_mesh(V)
grid_u = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_2d = u_sol.x.array.reshape((-1, mesh.geometry.dim))
u_3d = np.zeros((u_2d.shape[0], 3), dtype=u_2d.dtype)
u_3d[:, :2] = u_2d
grid_u["u"] = u_3d
grid_u["|u|"] = np.linalg.norm(u_2d, axis=1)

# Assign cell tags
grid_u.cell_data['cell_tags'] = new_ct.values
biofilm_tag = 6

# Extract new fluid and biofilm grids
fluid_grid = grid_u.extract_cells(grid_u.cell_data["cell_tags"] != biofilm_tag)
biofilm_grid = grid_u.extract_cells(grid_u.cell_data["cell_tags"] == biofilm_tag)

plotter = pyvista.Plotter()

# Fluid: colored by velocity magnitude
plotter.add_mesh(
    fluid_grid,
    scalars="|u|",
    cmap="viridis",
    show_edges=False
)

# Remaining biofilm: solid white
plotter.add_mesh(
    biofilm_grid,
    color="white",
    show_edges=True
)

# Optional: velocity glyphs only in fluid
glyphs = fluid_grid.glyph(orient="u", scale=False, factor=0.05)
plotter.add_mesh(glyphs, color="black")

plotter.add_title("Velocity field (updated biofilm → fluid)")
plotter.view_xy()
plotter.show()

# this looks correct now. 
# next step is to plot stress in this new field to see if my boundaries have actually changed
# or maybe can export mesh to gmsh and can see via tools/visibility

dim = mesh.geometry.dim
I = Identity(dim)
strain_rate = sym(grad(u_sol))
sigma_expr = -p_sol*I + 2.0*mu_field*strain_rate

tensor_el = element("Lagrange", mesh.basix_cell(), 1, shape=(dim, dim))
T = functionspace(mesh, tensor_el)
stress = Function(T)

sigma_trial = TrialFunction(T)
w = TestFunction(T)

a_proj = inner(sigma_trial, w) * dx
L_proj = inner(sigma_expr, w) * dx

stress_problem = dolfinx.fem.petsc.LinearProblem(
    a_proj,
    L_proj,
    bcs=[],  # no Dirichlet BCs for stress
    petsc_options={"ksp_type": "gmres", "pc_type": "lu"},
    petsc_options_prefix="solver_"
)
stress = stress_problem.solve()

stress_coords = T.tabulate_dof_coordinates() #this line is new

stress_vals = stress.x.array.reshape((-1, dim, dim))
stress_magnitude = np.linalg.norm(stress_vals, axis=(1,2))

x = stress_coords[:, 0] #these three lines are new
y = stress_coords[:, 1]
s = stress_magnitude

topology_s, cell_types_s, geometry_s = vtk_mesh(T)
grid_s = pyvista.UnstructuredGrid(topology_s, cell_types_s, geometry_s)
grid_s["s_mag"] = stress_magnitude

plotter = pyvista.Plotter()
plotter.add_mesh(grid_s,scalars="s_mag",cmap="coolwarm",show_edges=False)
plotter.add_title("Stress Magnitude field")
plotter.view_xy()
plotter.show()


fname = "eroded_mesh"

for ext in [".xdmf", ".h5"]:
    if os.path.exists(fname + ext):
        os.remove(fname + ext)

with XDMFFile(MPI.COMM_WORLD, fname + ".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    #xdmf.write_meshtags(new_ct, mesh.geometry)
    xdmf.write_meshtags(new_ft, mesh.geometry)

#-----------------------------------------------------------------------------

# fdim = mesh.topology.dim - 1 #this next block is new
# boundary_facets = ft.find(obstacle_marker)
# boundary_dofs = locate_dofs_topological(T, fdim, boundary_facets)
# stress_boundary = stress_vals[boundary_dofs]
# coords_boundary = stress_coords[boundary_dofs]
# stress_mag_boundary = np.linalg.norm(stress_boundary, axis=(1,2))
# idx = np.argsort(coords_boundary[:, 0])
# coords_boundary = coords_boundary[idx]
# stress_mag_boundary = stress_mag_boundary[idx]
# stress_boundary = stress_boundary[idx]
# ds = np.sqrt(np.sum(np.diff(coords_boundary, axis=0)**2, axis=1))
# s = np.insert(np.cumsum(ds), 0, 0.0)
# plt.plot(s, stress_mag_boundary, markersize=3)
# plt.xlabel("Arc length along boundary")
# plt.ylabel("Stress magnitude")
# plt.title("Boundary stress")
# plt.show()

# data = np.column_stack((coords_boundary[:, 0], coords_boundary[:, 1], stress_mag_boundary))
# np.savetxt(f"{folder}/uneven_boundary_stress.csv", data, delimiter=",", header="x,y,stress_mag", comments="")

# topology_s, cell_types_s, geometry_s = vtk_mesh(T)
# grid_s = pyvista.UnstructuredGrid(topology_s, cell_types_s, geometry_s)
# grid_s["s_mag"] = stress_magnitude

# plotter = pyvista.Plotter(off_screen=True)
# plotter.add_mesh(grid_s,scalars="s_mag",cmap="coolwarm",show_edges=False)
# plotter.add_title("Stress Magnitude field")
# plotter.view_xy()

# folder = "steady_stokes_results"
# os.makedirs(folder, exist_ok=True)
# plotter.screenshot(f"{folder}/uneven_stress_field.png")
# plotter.close()

# # --- Boundary stress plot with black line overlay ---
# plt.figure(figsize=(8,4))
# plt.plot(s, stress_mag_boundary, 'o', markersize=3, label="Stress magnitude")
# plt.plot(s, stress_mag_boundary, 'k-', linewidth=1.5, label="Boundary")  # black line along boundary
# plt.xlabel("Arc length along boundary")
# plt.ylabel("Stress magnitude")
# plt.title("Boundary stress with boundary line")
# plt.legend()
# plt.tight_layout()
# plt.show()

# endregion

print("end")