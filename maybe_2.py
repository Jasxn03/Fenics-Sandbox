
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
    Expression,
)
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells, BoundingBoxTree
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
import pyvista as pv

from dolfinx_mpc import (
    MultiPointConstraint,
    LinearProblem
)

import dolfinx.la.petsc
from dolfinx import plot
import typing
from dolfinx.mesh import create_submesh

# endregion

mesh_file = "Biofilm Meshes/uneven_biofilm_mesh.xdmf"
output_folder = "steady_stokes_results"
os.makedirs(output_folder, exist_ok=True)

with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
    mesh = xdmf.read_mesh(name="mesh")
    ft = xdmf.read_meshtags(mesh, name="Facet markers")
    ct = xdmf.read_meshtags(mesh, name ="Cell tags")

L_const, H_const = 50, 20
u_prev = None
p_prev = None
mu_field_prev = None
ct_current = ct
ft_current = ft
max_iterations = 5
stress_threshold = 0.1
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5


# region Functions 
from collections import deque

def plot_stress_pyvista(mesh, stress_cell, step):
    if mesh.comm.rank != 0:
        return

    import pyvista as pv
    import dolfinx.plot
    import numpy as np
    import os

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(
        mesh, mesh.topology.dim, cell_indices
    )

    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    grid.cell_data["StressMagnitude"] = stress_cell.x.array.copy()

    folder = "erosion"
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"stress_iteration_{step+1}.vtu")
    grid.save(filename)

    print(f"Saved stress field for iteration {step+1} → {filename}")


def remove_floating_biofilm(mesh, ct, mu_field, biofilm_marker=6):
    dim = mesh.topology.dim
    mesh.topology.create_connectivity(dim, 0)
    mesh.topology.create_connectivity(0, dim)

    # All biofilm cells
    biofilm_cells = set(ct.find(biofilm_marker))

    # Build clusters (connected via vertex-sharing)
    unvisited = set(biofilm_cells)
    clusters = []
    cell_to_vertex = mesh.topology.connectivity(dim, 0)
    vertex_to_cell = mesh.topology.connectivity(0, dim)

    while unvisited:
        start = unvisited.pop()
        cluster = set([start])
        queue = deque([start])
        while queue:
            c = queue.popleft()
            vertices = cell_to_vertex.links(c)
            for v in vertices:
                neighbors = vertex_to_cell.links(v)
                for n in neighbors:
                    if n in unvisited:
                        unvisited.remove(n)
                        cluster.add(n)
                        queue.append(n)
        clusters.append(cluster)

    # Keep the largest cluster
    # clusters = sorted(clusters, key=len, reverse=True)
    # main_cluster = clusters[0]

    # # Convert all smaller clusters to fluid
    # ct_values = ct.values.copy()
    # for cluster in clusters[1:]:
    #     for c in cluster:
    #         ct_values[c] = 1          # convert to fluid
    #         mu_field.x.array[c] = 1e-3

    MIN_CLUSTER_SIZE = 8  # tune this
    ct_values = ct.values.copy()
    for cluster in clusters:
        if len(cluster) < MIN_CLUSTER_SIZE:
            for c in cluster:
                ct_values[c] = 1
                mu_field.x.array[c] = 1e-3


    # Update MeshTags
    ct_new = dolfinx.mesh.meshtags(mesh, dim, ct.indices, ct_values)
    return ct_new, mu_field

def plot_biofilm_fluid_pyvista(mesh, ct, step):
    # Only plot on rank 0
    if mesh.comm.rank != 0:
        return

    # Get all cell indices
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)

    # Extract the VTK topology for these cells
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim, cell_indices)

    # Create a PyVista UnstructuredGrid
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # Create a cell array for coloring
    biofilm_cells = ct.find(6)   # biofilm tag
    fluid_cells = ct.find(1)     # fluid tag
    cell_data = np.zeros(num_cells, dtype=np.int32)
    cell_data[fluid_cells] = 1
    cell_data[biofilm_cells] = 6

    # Attach cell data to PyVista mesh
    grid.cell_data["BiofilmFluid"] = cell_data

    # Set up nice colors: fluid blue, biofilm red
    # (1 → blue, 6 → red)
    # cmap = {1:"blue", 6:"red"}

    # plotter = pv.Plotter()
    # # values mapped by our custom scalars array
    # plotter.add_mesh(grid, scalars="BiofilmFluid", 
    #                  categories=True, clim=[1,6],
    #                  cmap=["blue","red"], show_edges=True)
    # plotter.add_title(f"Iteration {step+1}: Fluid (blue) vs Biofilm (red)")
    # plotter.view_xy()
    # plotter.show()

    folder = "erosion"
    filename = os.path.join(f"{folder}/biofilm_fluid_iteration_{step+1}.vtu")
    grid.save(filename)
    print(f"Saved iteration {step+1} to {filename} (open in ParaView for high-quality visualization)")


def build_bcs_mpcs(mesh, ft):
    biofilm_marker = 6
    obstacle_marker = 5
    wall_marker = 4
    dim = mesh.topology.dim
    fdim = dim - 1

    v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(dim,))
    s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)
    Z = MixedFunctionSpace(V,Q)

    # BCs
    def shear_velocity_f(x):
        values = np.zeros((dim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0, :] = 1500
        values[1, :] = 0.0
        return values

    u_wall = Function(V)
    u_wall.interpolate(shear_velocity_f)
    bc_shear_velocity = dirichletbc(
        u_wall, locate_dofs_topological(V, fdim, ft.find(wall_marker))
    )

    u_nonslip = np.zeros(dim, dtype=PETSc.ScalarType)
    bcu_obstacle = dirichletbc(
        u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V
    )

    bcu = [bc_shear_velocity, bcu_obstacle]

    # Pressure reference
    coords = mesh.geometry.x
    x_target = 50 / 2
    y_target = 0
    dist2 = (coords[:, 0] - x_target)**2 + (coords[:, 1] - y_target)**2
    dof_p_ref = np.array([np.argmin(dist2)], dtype=np.int32)
    bcp_ref = dirichletbc(PETSc.ScalarType(0), dof_p_ref, Q)

    # MPCs for periodicity
    def periodic_boundary(x):
        return (x[0] == 0.0) | (x[0] == 50)

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

    return V, Q, Z, bcu, bcp_ref, [mpc_V, mpc_p], v_cg2, s_cg1



def remesh_iteration(mesh, ft, ct, u_prev = None, p_prev = None, mu_prev = None):
    biofilm_marker = 6
    obstacle_marker = 5
    wall_marker = 4

    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)

    V, Q, Z, bcu, bcp_ref, mpcs, v_cg2, s_cg1 = build_bcs_mpcs(mesh, ft)
    mpc_V, mpc_p = mpcs

    Q0 = functionspace(mesh, ("DG", 0))
    mu_field = Function(Q0)
    mu = Constant(mesh, PETSc.ScalarType(0.001))  
    mu_bf = Constant(mesh, PETSc.ScalarType(1000))
    mu_field.x.array[:] = 1e-3
    biofilm_cells = ct.find(6)  
    mu_field.x.array[biofilm_cells] = 1000 
    rho = Constant(mesh, PETSc.ScalarType(1))
    f = Constant(mesh, (0.0,0.0))

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

    problem = LinearProblem(
        a, 
        L,
        bcs = bcu + [bcp_ref],
        mpc = [mpc_V, mpc_p],
        petsc_options={"ksp_type": "gmres", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    )
    u_sol, p_sol = problem.solve()

    if u_prev is not None and p_prev is not None:
        u_sol.interpolate(u_prev)
        p_sol.interpolate(p_prev)
        u_sol.x.scatter_forward()
        p_sol.x.scatter_forward()
        mpc_V.backsubstitution(u_sol)
        mpc_p.backsubstitution(p_sol)
    
    u_sol.x.scatter_forward()
    p_sol.x.scatter_forward()
    mpc_V.backsubstitution(u_sol)
    mpc_p.backsubstitution(p_sol)
    
    print("Max velocity:", np.max(u_sol.x.array))
    
    dim = mesh.geometry.dim
    I = Identity(dim)
    strain_rate = sym(grad(u_sol))
    sigma_expr = -p_sol*I + 2.0*mu_field*strain_rate

    tensor_el = element("Lagrange", mesh.basix_cell(), 1, shape=(dim, dim))
    T = functionspace(mesh, tensor_el)
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



    # --- Compute stress magnitude at DOFs ---
    stress_vals = stress.x.array.reshape((-1, dim, dim))
    stress_mag_dofs = np.linalg.norm(stress_vals, axis=(1, 2))

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cell_to_vertex = mesh.topology.connectivity(mesh.topology.dim, 0)
    stress_mag_cell = np.zeros(num_cells)

    for c in range(num_cells):
        vertices = cell_to_vertex.links(c)  # vertices of this cell
        # dofs = []
        # for v in vertices:
        #     # For CG1 tensor, each vertex contributes dim*dim DOFs
        #     dofs.extend([v * dim * dim + i for i in range(dim*dim)])
        stress_mag_cell[c] = np.mean(stress_mag_dofs[vertices])

    # --- Wrap as DG0 Function for plotting ---
    Q0 = functionspace(mesh, ("DG", 0))
    stress_cell = Function(Q0)
    stress_cell.x.array[:] = stress_mag_cell

    mesh.topology.create_connectivity(dim, dim-1)
    mesh.topology.create_connectivity(dim-1, dim)

    fluid_facets = ct.find(1)
    biofilm_facets = ct.find(6)

    facet_to_cell = mesh.topology.connectivity(dim-1, dim)
    cell_to_facet = mesh.topology.connectivity(dim, dim-1)

    # --- Identify all biofilm cells ---
    biofilm_cells_set = set()
    for f in biofilm_facets:
        biofilm_cells_set.update(facet_to_cell.links(f))

    # --- Detect high-stress fluid cells ---
    high_stress_fluid_cells = set()
    boundary_facets = ft.find(obstacle_marker)
    for f in boundary_facets:
            # get the cells that share this facet
        owners = facet_to_cell.links(f)
        # check if any of these cells is a fluid cell and above threshold
        for c in owners:
            if ct.values[c] == 1:  # fluid cell
                if stress_mag_cell[c] > stress_threshold:
                    high_stress_fluid_cells.add(c)

    # neighbours that only share an edge
    neighboring_biofilm_cells = set()
    for c in high_stress_fluid_cells:
        facets = cell_to_facet.links(c)       
        for f in facets:
            neighbor_cells = facet_to_cell.links(f) 
            for n in neighbor_cells:
                if n in biofilm_cells_set:
                    neighboring_biofilm_cells.add(n)
    
    additional_neighbors = set()
    for c in neighboring_biofilm_cells:
        facets = cell_to_facet.links(c)
        for f in facets:
            neighbor_cells = facet_to_cell.links(f)
            for n in neighbor_cells:
                if n in biofilm_cells_set:
                    additional_neighbors.add(n)


    neighboring_biofilm_cells.update(additional_neighbors)

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    all_indices = np.arange(num_cells, dtype=np.int32)
    ct_old_values = ct.values.copy() #this line is new
    ct_values = ct.values.copy()
    for c in neighboring_biofilm_cells:
        ct_values[c] = 1
        mu_field.x.array[c] = 1e-3
    # Replace old MeshTags
    ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, ct.indices, ct_values)
    # ct, mu_field = remove_floating_biofilm(mesh, ct, mu_field, biofilm_marker=6)

    # Update viscosity field
    for c in neighboring_biofilm_cells:
        mu_field.x.array[c] = 1e-3

    dim = mesh.topology.dim
    fdim = dim - 1
    mesh.topology.create_connectivity(dim-1, dim)
    mesh.topology.create_connectivity(dim, dim-1)

    facet_to_cell = mesh.topology.connectivity(dim-1, dim)
    cell_to_facet = mesh.topology.connectivity(dim, dim-1)

    new_ft_values = np.zeros(mesh.topology.index_map(fdim).size_local, dtype=np.int32)

    # Re-mark obstacle facets (boundary of biofilm or original obstacles)
    for f in range(mesh.topology.index_map(fdim).size_local):
        linked_cells = facet_to_cell.links(f)
        if len(linked_cells) == 1:
            # external boundary facet
            new_ft_values[f] = obstacle_marker
        elif len(linked_cells) == 2:
            c1, c2 = linked_cells
            # facet between fluid (1) and biofilm (6)
            if (ct.values[c1] == 1 and ct.values[c2] == 6) or (ct.values[c2] == 1 and ct.values[c1] == 6):
                new_ft_values[f] = obstacle_marker

    ft = dolfinx.mesh.meshtags(mesh, fdim, np.arange(len(new_ft_values), dtype=np.int32), new_ft_values)


    print("High-stress fluid cells:", len(high_stress_fluid_cells))
    print("Biofilm cells to remove:", len(neighboring_biofilm_cells))
    print("Biofilm cells converted to fluid:", len(neighboring_biofilm_cells))
    num_changed = np.sum(ct_old_values != ct_values) #changed ct.indices to ct_old_values
    print(f"Number of biofilm cells actually converted to fluid this iteration: {num_changed}")


    # --- Create submesh without these biofilm cells ---
    if len(neighboring_biofilm_cells) == 0:
        return None, u_sol, p_sol, mu_field, ft , ct, stress_mag_cell  # nothing to remove

    return mesh, u_sol, p_sol, mu_field, ft, ct, stress_mag_cell

#endregion

# Iteration
for step in range(max_iterations):
    print(f"\n=== Step {step+1} ===")
    ct_current, mu_field_prev = remove_floating_biofilm(mesh, ct_current, mu_field_prev, biofilm_marker=6)

    mesh, u_prev, p_prev, mu_field_prev, ft_current, ct_current, stress_mag_cell = remesh_iteration(
        mesh, ft_current, ct_current, u_prev, p_prev, mu_prev = mu_field_prev
    )
    # Re-wrap stress magnitude as DG0 function
    Q0 = functionspace(mesh, ("DG", 0))
    stress_cell = Function(Q0)
    stress_cell.x.array[:] = stress_mag_cell

    plot_stress_pyvista(mesh, stress_cell, step)
    plot_biofilm_fluid_pyvista(mesh, ct_current, step)

    if len(ct_current.find(6)) == 0:
        print("All biofilm removed, stopping simulation.")
        break
