# this is the same as test but i want to loop over different arbitrary biofilms

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
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import gc

# endregion

mesh_folder = "Varying Height Meshes Meshes"
#bump_height = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
bump_height = [15.0]
#bump_length = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
#bump_length = [15.0, 16.0, 17.0]
#bump_length = [18.0]
#bump_length = [19.0, 20.0, 21.0]
#bump_length = [22.0, 23.0, 24.0]
bump_length = [25.0]
mesh_files = [f"{mesh_folder}/biofilm_height_{h:.2f}_length_{l:.2f}.xdmf" for h, l in zip(bump_height, bump_length)]

velocity_values = [10000, 11000, 12000]  # shear velocities to test

if MPI.COMM_WORLD.rank == 0:
    os.makedirs("simulation_results", exist_ok = True)
    index_path = "simulation_results/index.csv"
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("height, length, velocity, file\n")

all_stress =[]
coords_boundary_ref = None
shear_velocities = velocity_values

for height in bump_height:
    for length in bump_length:
        mesh_file = f"{mesh_folder}/biofilm_height_{height:.2f}_length_{length:.2f}.xdmf"
        print(f"running with for bump height {height} and length {length}")
        for shear_val in velocity_values:
            print(f"running with shear velocity {shear_val}")
            L = 50
            H = 20
            with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
                mesh = xdmf.read_mesh(name = "mesh")
                ft = xdmf.read_meshtags(mesh, name="Facet markers")

            inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5 # need to write these three lines so it knows what markers are what
            fdim = mesh.topology.dim - 1 
            mesh.topology.create_connectivity(fdim, mesh.topology.dim)

            # region Boundary Conditions 
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
                values[0, :] = shear_val
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

            dim = mesh.geometry.dim
            I = Identity(dim)
            strain_rate = sym(grad(u_sol))
            sigma_expr = -p_sol*I + 2.0*mu*strain_rate

            # Project stress tensor onto Lagrange P1 tensor space
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

            # Coordinates of stress DOFs
            stress_coords = T.tabulate_dof_coordinates()
            stress_vals = stress.x.array.copy().reshape((-1, dim, dim))
            stress_magnitude = np.linalg.norm(stress_vals, axis=(1,2))

            # Get obstacle (biofilm) boundary
            fdim = mesh.topology.dim - 1
            boundary_facets = ft.find(obstacle_marker)
            boundary_dofs = locate_dofs_topological(T, fdim, boundary_facets)
            stress_boundary = stress_vals[boundary_dofs]
            coords_boundary = stress_coords[boundary_dofs]
            stress_mag_boundary = np.linalg.norm(stress_boundary, axis=(1,2))

            # Sort along boundary (by x or y)
            idx_sort = np.argsort(coords_boundary[:,0])
            coords_boundary = coords_boundary[idx_sort]
            stress_mag_boundary = stress_mag_boundary[idx_sort]

            if mesh.comm.rank ==0:
                os.makedirs("simulation_results/data", exist_ok=True)
                out_name = (f"stress_height{height:.2f}"f"_length{length:.2f}"f"_velocity{shear_val:.2f}")
                out_path = os.path.join("simulation_results/data", out_name)

                csv_data = np.column_stack((coords_boundary[:,0], coords_boundary[:,1], stress_mag_boundary))

                np.savetxt(out_path, csv_data, delimiter=",", header="x,y,stress", comments="")
            if mesh.comm.rank==0:
                with open(index_path, "a") as f:
                    f.write(f"{height}, {length}, {shear_val},{out_name}\n")

            # del problem
            # del u_sol, p_sol
            # del u_full, p_full
            # del V, Q, Z
            # del mpc_V, mpc_p
            # del mesh
            # del ft
            # del stress
            # del stress_vals
            # del stress_mag_boundary
            # del coords_boundary


# endregion

print("end")