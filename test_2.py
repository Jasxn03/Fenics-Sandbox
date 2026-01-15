import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile

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
    gmsh.fltk.run()

mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
ft = mesh_data.facet_tags
ft.name = "Facet markers"

# this is the bit that writes mesh to file

with XDMFFile(mesh_comm, "mesh_biofilm.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)

gmsh.finalize()

print("Mesh generation complete.")