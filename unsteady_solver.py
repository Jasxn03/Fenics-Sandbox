# this is unsteady stokes and seems to work

# region Imports
import gmsh
import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook

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

L = 1
H = 1
r = 0.1 
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    N=1000
    k = 10*np.pi/L
    A = 0.1
    top_points = []
    for i in range(N):
        x = L * i / (N - 1)
        y = 0.1 + A * np.sin(k * x)
        top_points.append(
            gmsh.model.occ.addPoint(x, y, 0)
        )
    top_bf = gmsh.model.occ.addSpline(top_points)
    L_bf = L
    H_bf = 0.1
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

# region Boundary Conditions

t = 0.0
T = 5.0  
dt = 0.005 
num_steps = int(T / dt)
k = Constant(mesh, PETSc.ScalarType(dt))    
mu = Constant(mesh, PETSc.ScalarType(1))  
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
    
def shear_velocity_f(x):
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)  # 2 rows for u_x, u_y
    values[0, :] = 1.5  # u_x = 1.5 (shear velocity)
    values[1, :] = 0.0  # u_y = 0
    return values

# shear velocity
u_wall = Function(V)
u_wall.interpolate(shear_velocity_f)
bc_shear_velocity = dirichletbc(
    u_wall, locate_dofs_topological(V, fdim, ft.find(wall_marker))
)

# biofilim
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_obstacle = dirichletbc(
    u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V
)
             
bcu = bc_shear_velocity, bcu_obstacle

# pressure = 0 on top and on bottom (obstacle boundary)
facets_wall = ft.find(wall_marker)
facets_obstacle = ft.find(obstacle_marker)
facets_pressure = np.concatenate([facets_wall, facets_obstacle])
dofs_pressure = locate_dofs_topological(Q, entity_dim=1, entities=facets_wall) #changing it so pressure is set to 0 on one boundary makes it look better

bcp = dirichletbc(PETSc.ScalarType(0), dofs_pressure, Q)

coords = mesh.geometry.x
x_target = L / 2
y_target = H
dist2 = (coords[:, 0] - x_target)**2 + (coords[:, 1] - y_target)**2
dof_p_ref = np.array([np.argmin(dist2)], dtype=np.int32)
bcp_ref = dirichletbc(PETSc.ScalarType(0), dof_p_ref, Q)

bcup = [bcu, bcp_ref]

tol = 1e-7
# def periodic_boundary(x):
#     on_left_or_right = np.isclose(x[0], 0.0, atol=1e-14) | np.isclose(x[0], L, atol=tol)
#     not_on_top = ~np.isclose(x[1], H, atol=1e-14)
#     not_on_bottom = ~np.isclose(x[1], 0.0, atol=tol)
#     return on_left_or_right & not_on_top & not_on_bottom

# def periodic_boundary(x):
#     on_left_or_right = np.isclose(x[0], 0.0, atol=tol) | np.isclose(x[0], L, atol=tol)
#     not_on_top = x[1] < H  #- tol    # exclude top
#     not_on_bottom = x[1] > 0 # + tol # exclude bottom
#     return on_left_or_right & not_on_top & not_on_bottom

# def periodic_boundary(x):
#     on_left_or_right = np.isclose(x[0], 0.0, atol=1e-14) | np.isclose(x[0], L, atol=tol)
#     return on_left_or_right

def periodic_boundary(x):
    return (x[0] == 0.0) | (x[0] == L)


def periodic_relation(x):
    out_x = np.zeros_like(x)
    out_x[0] = 0  
    out_x[1] = x[1] 
    return out_x

mpc_V = MultiPointConstraint(V)
mpc_V.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, bcu)#empty box contained bcu_obstacle
mpc_V.finalize()

mpc_p = MultiPointConstraint(Q)
mpc_p.create_periodic_constraint_geometrical(Q, periodic_boundary, periodic_relation, [bcp_ref]) 
# last argument of above is one that says "these are the dirichlet conditions, i will not impose periodicity here"
mpc_p.finalize()

mpc = [mpc_V, mpc_p]

# endregion

(u, p) = TrialFunctions(Z)
(v, q) = TestFunctions(Z)
u_n = Function(V)
u_n.x.array[:] = 0.0
u_sol = Function(V)
p_sol = Function(Q)

a = [[(rho/dt) * inner(u, v) * dx + mu * inner(grad(u), grad(v)) * dx, -inner(p, div(v)) * dx],
     [inner(div(u), q) * dx, None]]

L = [inner(f, v) * dx + (rho/dt) * inner(u_n, v) * dx, Constant(mesh, PETSc.ScalarType(0)) * q * dx]


from pathlib import Path

folder = Path("results_stokes_2")
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(mesh.comm, folder / "velocity.bp", [u_sol], engine="BP4")
vtx_p = VTXWriter(mesh.comm, folder / "pressure.bp", [p_sol], engine="BP4")
vtx_u.write(t)
vtx_p.write(t)

problem = LinearProblem(
    a,
    L,
    bcs=[bc_shear_velocity, bcu_obstacle, bcp_ref],
    mpc=[mpc_V, mpc_p],
    petsc_options = {
                    "ksp_type": "gmres",
                    "pc_type": "fieldsplit",
                    "pc_fieldsplit_type": "schur",
                    "pc_fieldsplit_schur_fact_type": "FULL", 
                    "fieldsplit_0_ksp_type": "preonly",
                    "fieldsplit_0_pc_type": "ilu",
                    "fieldsplit_1_ksp_type": "preonly",
                    "fieldsplit_1_pc_type": "jacobi"}
)

snapshot = 0

progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
for n in range(num_steps):
    t = (n+1)*dt
    progress.update(1)

    u_sol, p_sol = problem.solve()

    u_sol.x.scatter_forward()
    p_sol.x.scatter_forward()

    mpc_V.backsubstitution(u_sol)
    mpc_p.backsubstitution(p_sol)

    u_n.x.array[:] = u_sol.x.array[:]
    u_n.x.scatter_forward()

    folder = Path("results_stokes_2/vtu_files")
    folder.mkdir(exist_ok=True, parents=True)
    vtx_writer = VTXWriter(mesh.comm, folder / f"velocity_{n:04d}.vtu", [u_sol])
    vtx_writer.write(0.0) 
    vtx_writer.close()
    

    if mesh.comm.rank == 0 and n % 50 == 0:
        num_vertices = mesh.geometry.x.shape[0]

        u_vals = u_sol.x.array.reshape((num_vertices, mesh.geometry.dim))
        u_3d = np.zeros((num_vertices, 3), dtype=np.float64)
        u_3d[:, :mesh.geometry.dim] = u_vals

        topology, cell_types, x = vtk_mesh(V)
        num_vertices = x.shape[0]
        coords_3d = x.copy()
        grid = pyvista.UnstructuredGrid(topology, cell_types, coords_3d)
        grid.point_data["u"] = u_3d

        plotter = pyvista.Plotter(off_screen = True)
        plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, scalars="u", cmap = "viridis")#show_edges = True/False for mesh lines
        plotter.view_xy()
        folder = "results_stokes_2/velocity_pyvista"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/velocity_pyvista_{snapshot:04d}.png"
        plotter.screenshot(filename)
        plotter.close()
        print(f"Saved figure {filename}")

        print("Check",np.linalg.norm(u_sol.x.array - u_n.x.array))

        snapshot += 1

    print("Max velocity:", np.max(u_sol.x.array))

vtx_u.close()
vtx_p.close()

print("u_sol:", u_sol.x.array)
print("u_max:", np.max(u_sol.x.array))

print("simulation finished")

# ffmpeg -r 10 -pattern_type glob -i 'results_stokes_2/velocity_pyvista/velocity_pyvista_*.png' -vcodec libx264 velocity_animation.mp4
