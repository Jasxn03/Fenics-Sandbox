# this one is going to be steady, so no time dependence but the same as 5
# this works

# region Imports
import gmsh
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
    assemble_scalar,
    dirichletbc,
    extract_function_spaces,
    form,
    locate_dofs_topological,
    set_bc,
    Expression,
    bcs_by_block
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    create_matrix,
    set_bc
)
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import VTXWriter, gmsh as gmshio, XDMFFile
from ufl import (
    FacetNormal,
    Measure,
    TestFunctions,
    TrialFunctions,
    as_vector,
    div,
    dot,
    dx,
    inner,
    lhs,
    grad,
    nabla_grad,
    rhs,
    sqrt,
    Identity,
    transpose,
    as_tensor,
    SpatialCoordinate,
    MixedFunctionSpace,
    extract_blocks,
    TestFunction,
    TrialFunction,
    sym
)
from dolfinx.plot import vtk_mesh
from dolfinx import default_scalar_type
import pyvista

from dolfinx_mpc import (
    MultiPointConstraint,
    apply_lifting,
    LinearProblem
)
from dolfinx_mpc import assemble_matrix as mpc_assemble_matrix
from dolfinx_mpc import assemble_vector as mpc_assemble_vector
from dolfinx_mpc import assemble_matrix_nest as mpc_assemble_matrix_nest
from dolfinx_mpc import assemble_vector_nest as mpc_assemble_vector_nest
from dolfinx_mpc import create_matrix_nest
from dolfinx_mpc import create_matrix_nest, create_vector_nest
import dolfinx.la.petsc
from dolfinx import plot
import typing


# endregion

# region Mesh Creation

gmsh.initialize()

L = 50
H = 20
r = 1
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    N=1000
    k = 10*np.pi/L
    A = 1
    top_points = []
    for i in range(N):
        x = L * i / (N - 1)
        y = 1 + A * np.sin(k * x)
        top_points.append(
            gmsh.model.occ.addPoint(x, y, 0)
        )
    top_bf = gmsh.model.occ.addSpline(top_points)
    L_bf = L
    H_bf = 1
    p0 = gmsh.model.occ.addPoint(0, 0, 0)
    p1 = gmsh.model.occ.addPoint(L_bf, 0, 0)
    bottom_bf = gmsh.model.occ.addLine(p0, p1)
    left_bf = gmsh.model.occ.addLine(top_points[0], p0)
    right_bf = gmsh.model.occ.addLine(p1, top_points[-1])
    biofilm_boundary = gmsh.model.occ.addCurveLoop([bottom_bf, right_bf, -top_bf, left_bf])
    biofilm_surface = gmsh.model.occ.addPlaneSurface([biofilm_boundary])

if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, biofilm_surface)])
    gmsh.model.occ.synchronize()

fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert len(volumes) == 1
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")


inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        y_min = H_bf - A - 1e-6
        y_max = H_bf + A + 1e-6
        for dim, tag in boundaries:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            if abs(xmin) < 1e-6 and abs(xmax) < 1e-6:
                inflow.append(tag)

            elif abs(xmin - L) < 1e-6 and abs(xmax - L) < 1e-6:
                outflow.append(tag)

            elif abs(ymin) < 1e-6 and abs(ymax) < 1e-6:
                walls.append(tag)

            elif abs(ymin - H) < 1e-6 and abs(ymax - H) < 1e-6:
                walls.append(tag)
            else:
                obstacle.append(tag)
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

res_min = r / 3
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")
    #gmsh.fltk.run()

mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
ft = mesh_data.facet_tags
ft.name = "Facet markers"

# endregion

# with XDMFFile(MPI.COMM_WORLD, "mesh_biofilm.xdmf", "r") as xdmf:
#     mesh = xdmf.read_mesh(name = "mesh")
#     ft = xdmf.read_meshtags(mesh, name="Facet markers")

# inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
# fdim = mesh.topology.dim - 1 
# mesh.topology.create_connectivity(fdim, mesh.topology.dim)

# region Boundary Conditions

# t = 0.0
# T = 20.0  
# dt = 0.01 
# num_steps = int(T / dt)
# k = Constant(mesh, PETSc.ScalarType(dt))    
mu = Constant(mesh, PETSc.ScalarType(0.001))  
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
    values[0, :] = 1.5 
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

# pressure = 0 on top and on bottom (obstacle boundary)
# facets_wall = ft.find(wall_marker)
# facets_obstacle = ft.find(obstacle_marker)
# facets_pressure = np.concatenate([facets_wall, facets_obstacle])
# dofs_pressure = locate_dofs_topological(Q, entity_dim=1, entities=facets_wall) #changing it so pressure is set to 0 on one boundary makes it look better

# bcp = dirichletbc(PETSc.ScalarType(0), dofs_pressure, Q)

coords = mesh.geometry.x
x_target = L / 2
y_target = H
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

a = [[mu * inner(grad(u), grad(v)) * dx, -inner(p, div(v)) * dx],
    [ inner(div(u), q) * dx, None]]


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

print("simulation finished")

# endregion 

# region Plotting

# u plotting
topology, cell_types, geometry = vtk_mesh(V)
grid_u = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_2d = u_sol.x.array.reshape((-1, mesh.geometry.dim))
u_3d = np.zeros((u_2d.shape[0], 3), dtype=u_2d.dtype)
u_3d[:, :2] = u_2d
grid_u["u"] = u_3d
grid_u["|u|"] = np.linalg.norm(u_2d, axis=1)

plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(grid_u,scalars="|u|",cmap="viridis",show_edges=False)

glyphs = grid_u.glyph(orient="u",scale=False,factor=0.05)
plotter.add_mesh(glyphs, color="black")
plotter.add_title("Velocity field")
plotter.view_xy()
folder = "steady_stokes_results"
os.makedirs(folder, exist_ok=True)
plotter.screenshot(f"{folder}/velocity_field.png")
plotter.close()

# p plotting
topology_p, cell_types_p, geometry_p = vtk_mesh(Q)
grid_p = pyvista.UnstructuredGrid(topology_p, cell_types_p, geometry_p)
grid_p["p"] = p_sol.x.array

plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(grid_p,scalars="p",cmap="coolwarm",show_edges=False)
plotter.add_title("Pressure field")
plotter.view_xy()
folder = "steady_stokes_results"
os.makedirs(folder, exist_ok=True)
plotter.screenshot(f"{folder}/pressure_field.png")
plotter.close()

print("plotting finished")

# endregion

# stress
dim = mesh.geometry.dim
I = Identity(dim)
strain_rate = sym(grad(u_sol))
sigma_expr = -p_sol*I + 2.0*mu*strain_rate

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

fdim = mesh.topology.dim - 1 #this next block is new
boundary_facets = ft.find(obstacle_marker)
boundary_dofs = locate_dofs_topological(T, fdim, boundary_facets)
stress_boundary = stress_vals[boundary_dofs]
coords_boundary = stress_coords[boundary_dofs]
stress_mag_boundary = np.linalg.norm(stress_boundary, axis=(1,2))
idx = np.argsort(coords_boundary[:, 0])
coords_boundary = coords_boundary[idx]
stress_mag_boundary = stress_mag_boundary[idx]
stress_boundary = stress_boundary[idx]
ds = np.sqrt(np.sum(np.diff(coords_boundary, axis=0)**2, axis=1))
s = np.insert(np.cumsum(ds), 0, 0.0)
# plt.plot(s, stress_mag_boundary, markersize=3)
# plt.xlabel("Arc length along boundary")
# plt.ylabel("Stress magnitude")
# plt.title("Boundary stress")
# plt.show()

data = np.column_stack((coords_boundary[:, 0], coords_boundary[:, 1], stress_mag_boundary))
np.savetxt(f"{folder}/boundary_stress.csv", data, delimiter=",", header="x,y,stress_mag", comments="")

topology_s, cell_types_s, geometry_s = vtk_mesh(T)
grid_s = pyvista.UnstructuredGrid(topology_s, cell_types_s, geometry_s)
grid_s["s_mag"] = stress_magnitude

plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(grid_s,scalars="s_mag",cmap="coolwarm",show_edges=False)
plotter.add_title("Stress Magnitude field")
plotter.view_xy()
folder = "steady_stokes_results"
os.makedirs(folder, exist_ok=True)
plotter.screenshot(f"{folder}/stress_field.png")
plotter.close()

# fdim = mesh.topology.dim - 1 
# obstacle_facets = ft.find(obstacle_marker)
# obstacle_dofs = locate_dofs_topological(T, fdim, obstacle_facets)
# stress_obstacle = stress_vals[obstacle_dofs, :, :]
# stress_obs_mag = np.linalg.norm(stress_obstacle, axis=(1, 2))
# obstacle_coords = mesh.geometry.x[obstacle_dofs, :]

# x_obs = obstacle_coords[:, 0]

# # sort by x-position
# sorted_idx = np.argsort(x_obs)
# x_obs_sorted = x_obs[sorted_idx]
# stress_obs_mag_sorted = stress_obs_mag[sorted_idx]

# plt.figure(figsize=(6, 4))
# plt.plot(x_obs_sorted, stress_obs_mag_sorted)
# plt.xlabel("x-position along obstacle")
# plt.ylabel("Stress magnitude")
# plt.title("Stress magnitude on obstacle boundary")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# sorted_idx = np.argsort(obstacle_coords[:, 0])
# coords_sorted = obstacle_coords[sorted_idx, :]
# stress_sorted = stress_obs_mag[sorted_idx]

# arc_length = np.zeros(len(coords_sorted))
# for i in range(1, len(arc_length)):
#     arc_length[i] = arc_length[i-1] + np.linalg.norm(coords_sorted[i] - coords_sorted[i-1])

# plt.figure(figsize=(6,4))
# plt.plot(arc_length, stress_sorted)
# plt.xlabel("Arc length along obstacle")
# plt.ylabel("Stress magnitude")
# plt.title("Stress magnitude along obstacle boundary")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


print("end")