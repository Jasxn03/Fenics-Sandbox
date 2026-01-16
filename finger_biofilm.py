import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
import os

gmsh.initialize()
L =  50.0
H = 20.0
r = 1 
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    n = 10
    bottom_points = []
    for i in range(n+1):
        bottom_points.append(gmsh.model.occ.addPoint(5*i,1,0))
    bottom_center_points = []
    for i in range(n+1):
        bottom_center_points.append(gmsh.model.occ.addPoint(5*i,2.25,0))
    bottom_left_points = []
    for i in range(n):
        bottom_left_points.append(gmsh.model.occ.addPoint(5*i + 1.25,2.25,0))
    bottom_right_points = []
    for i in range(n):
        bottom_right_points.append(gmsh.model.occ.addPoint(5*i + 3.75,2.25,0))
    top_left_points = []
    for i in range(n):
        top_left_points.append(gmsh.model.occ.addPoint(5*i + 1.25,4,0))
    top_right_points = []
    for i in range(n):
        top_right_points.append(gmsh.model.occ.addPoint(5*i + 3.75,4,0))
    top_points = []
    for i in range(n):
        top_points.append(gmsh.model.occ.addPoint(5*i + 2.5,4,0))
    bottom_left_curve = []
    for i in range(n):
        bottom_left_curve.append(gmsh.model.occ.addCircleArc(bottom_left_points[i], bottom_center_points[i], bottom_points[i]))
    bottom_right_curve = []
    for i in range(n):
        bottom_right_curve.append(gmsh.model.occ.addCircleArc(bottom_points[i+1], bottom_center_points[i+1], bottom_right_points[i]))
    left_curve = []
    for i in range(n):
        left_curve.append(gmsh.model.occ.addLine(bottom_left_points[i], top_left_points[i]))
    right_curve = []
    for i in range(n):
        right_curve.append(gmsh.model.occ.addLine(top_right_points[i], bottom_right_points[i]))
    top_curve = []
    for i in range(n):
        top_curve.append(gmsh.model.occ.addCircleArc(top_left_points[i], top_points[i], top_right_points[i]))
    left_line = gmsh.model.occ.addLine(gmsh.model.occ.addPoint(0,0,0), gmsh.model.occ.addPoint(0,1,0))
    right_line = gmsh.model.occ.addLine(gmsh.model.occ.addPoint(L,1,0), gmsh.model.occ.addPoint(L,0,0))
    bottom_line = gmsh.model.occ.addLine(gmsh.model.occ.addPoint(L,0,0), gmsh.model.occ.addPoint(0,0,0))
    biofilm_boundary_curves = []
    for i in range(n):
        biofilm_boundary_curves.append(-bottom_left_curve[i])
        biofilm_boundary_curves.append(left_curve[i])
        biofilm_boundary_curves.append(top_curve[i])
        biofilm_boundary_curves.append(right_curve[i])
        biofilm_boundary_curves.append(-bottom_right_curve[i])
    biofilm_boundary_curves.append(right_line)
    biofilm_boundary_curves.append(bottom_line)
    biofilm_boundary_curves.append(left_line)
    biofilm_boundary = gmsh.model.occ.addCurveLoop(biofilm_boundary_curves)
    biofilm_surface = gmsh.model.occ.addPlaneSurface([biofilm_boundary])
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, biofilm_surface)])
    gmsh.model.occ.synchronize()
fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(gdim)
    assert len(volumes) == 1
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
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
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

res_min = r/3
if mesh_comm.rank == model_rank:   
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25*H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2*H)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm",2)
    gmsh.option.setNumber("Mesh.RecombineAll",1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm",1)
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

folder = "Biofilm Meshes"
file_path = os.path.join(folder, "finger_biofilm_mesh.xdmf")

with XDMFFile(mesh_comm, file_path, "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)

gmsh.finalize()

print("Mesh generation complete.")