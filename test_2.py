# changed u = 1500 on top boundary to du/dz = 1500. 
# as a consequence, we have neumann type bc on the top boundary
# thus we need to basically add a vector to our weak formulation, so we remove the dirichlet bc and add boundary integral to RHS


# region Imports
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
    sym,
    as_vector,
    Measure
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

# endregion

L = 50
H = 20
with XDMFFile(MPI.COMM_WORLD, "Biofilm Meshes/arc_biofilm_mesh.xdmf", "r") as xdmf:
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
mu_field.x.array[biofilm_cells] = 0.001
gamma_dot = Constant(mesh, PETSc.ScalarType(1500.0))
traction = as_vector([mu * gamma_dot, 0.0])
ds = Measure("ds", domain=mesh, subdomain_data=ft)


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


# Normal no-penetration at top wall: u_y = 0,
uy_zero = Function(Q)
uy_zero.x.array[:] = 0.0

top_facets = ft.find(wall_marker)
top_dofs_y = locate_dofs_topological(
    V.sub(1), fdim, top_facets
)
bc_top_normal = dirichletbc(uy_zero,top_dofs_y)


# biofilm
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_obstacle = dirichletbc(
    u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V
)

# Fix one tangential velocity DOF to kill rigid-body mode
obs_facets = ft.find(obstacle_marker)
dofs_x = locate_dofs_topological(V.sub(0), fdim, obs_facets)

bc_fix_ux = dirichletbc(
    PETSc.ScalarType(0.0),
    dofs_x[:1],   # just ONE dof is enough
    V.sub(0)
)

top_facets = ft.find(wall_marker)
top_dofs_x = locate_dofs_topological(V.sub(0), fdim, top_facets)
bc_top_fix = dirichletbc(PETSc.ScalarType(0.0), top_dofs_x[:10], V.sub(0))

bcu = [bcu_obstacle, bc_top_normal, bc_top_fix]

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

a = [[
    2 * inner(mu_field * sym(grad(u)), sym(grad(v))) * dx,
    -inner(p, div(v)) * dx
],[
    inner(div(u), q) * dx,
    None
]]


L = [inner(f, v) * dx + inner(traction, v) * ds(wall_marker),
    Constant(mesh, PETSc.ScalarType(0)) * q * dx]

from pathlib import Path

problem = LinearProblem(
    a,
    L,
    bcs=[bcu_obstacle, bcp_ref, bc_top_normal, bc_top_fix],
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
# plotter.screenshot(f"{folder}/elliptic_velocity_field.png")
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
# plotter.screenshot(f"{folder}/elliptic_velocity_field_masked.png")
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
# plotter.screenshot(f"{folder}/elliptic_pressure_field.png")
# plotter.close()

# print("plotting finished")

# endregion

# region stress
# dim = mesh.geometry.dim
# I = Identity(dim)
# strain_rate = sym(grad(u_sol))
# sigma_expr = -p_sol*I + 2.0*mu_field*strain_rate

# tensor_el = element("Lagrange", mesh.basix_cell(), 1, shape=(dim, dim))
# T = functionspace(mesh, tensor_el)
# stress = Function(T)

# sigma_trial = TrialFunction(T)
# w = TestFunction(T)

# a_proj = inner(sigma_trial, w) * dx
# L_proj = inner(sigma_expr, w) * dx

# stress_problem = dolfinx.fem.petsc.LinearProblem(
#     a_proj,
#     L_proj,
#     bcs=[],  # no Dirichlet BCs for stress
#     petsc_options={"ksp_type": "gmres", "pc_type": "lu"},
#     petsc_options_prefix="solver_"
# )
# stress = stress_problem.solve()

# stress_coords = T.tabulate_dof_coordinates() #this line is new

# stress_vals = stress.x.array.reshape((-1, dim, dim))
# stress_magnitude = np.linalg.norm(stress_vals, axis=(1,2))

# x = stress_coords[:, 0] #these three lines are new
# y = stress_coords[:, 1]
# s = stress_magnitude

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
# np.savetxt(f"{folder}/elliptic_boundary_stress.csv", data, delimiter=",", header="x,y,stress_mag", comments="")

# topology_s, cell_types_s, geometry_s = vtk_mesh(T)
# grid_s = pyvista.UnstructuredGrid(topology_s, cell_types_s, geometry_s)
# grid_s["s_mag"] = stress_magnitude

# plotter = pyvista.Plotter(off_screen=True)
# plotter.add_mesh(grid_s,scalars="s_mag",cmap="coolwarm",show_edges=False)
# plotter.add_title("Stress Magnitude field")
# plotter.view_xy()

# folder = "steady_stokes_results"
# os.makedirs(folder, exist_ok=True)
# plotter.screenshot(f"{folder}/elliptic_stress_field.png")
# plotter.close()

# endregion

print("end")