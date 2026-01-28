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
    fluid = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    p1 = gmsh.model.occ.addPoint(35,0,0)
    p2 = gmsh.model.occ.addPoint(L/2,0,0)
    p3 = gmsh.model.occ.addPoint(15,0,0)
    arc_loop = gmsh.model.occ.addCircleArc(p3, p2, p1)
    line = gmsh.model.occ.addLine(p1,p3)
    biofilm_boundary = gmsh.model.occ.addCurveLoop([arc_loop, line])
    biofilm_surface = gmsh.model.occ.addPlaneSurface([biofilm_boundary])
    whole_domain = gmsh.model.occ.fragment([(gdim, fluid)], [(gdim, biofilm_surface)])
    gmsh.model.occ.synchronize()
fluid_marker = 1
biofilm_marker = 6
if mesh_comm.rank == model_rank:
    surfaces = gmsh.model.getEntities(2)
    assert len(surfaces) == 2
    areas =[]
    for dim, tag in surfaces:
        area = gmsh.model.occ.getMass(dim, tag)
        areas.append((area, tag))
    areas.sort()
    biofilm_tag = areas[0][1]
    fluid_tag = areas[1][1]

    gmsh.model.addPhysicalGroup(2, [fluid_tag], 1)
    gmsh.model.setPhysicalName(2,1, "Fluid")
    gmsh.model.addPhysicalGroup(2, [biofilm_tag], 6)
    gmsh.model.setPhysicalName(2,6,"Biofilm")

inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []
obstacle_candidates = []
tol = 1e-6

if mesh_comm.rank == model_rank:
    curves = gmsh.model.getEntities(1)
    fluid_boundary = gmsh.model.getBoundary([(2, fluid_tag)], oriented=False)
    for dim, curve in curves:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1,curve)
        left = abs(xmin) < tol and abs(xmax) < tol
        right = abs(xmin-L) < tol and abs(xmax-L) < tol
        top = abs(ymin-H) < tol and abs(ymax-H) < tol
        if left:
            inflow.append(curve)
        elif right:
            outflow.append(curve)
        elif top:
            walls.append(curve)
    for dim, tag in fluid_boundary:
        if dim == 1:
                obstacle_candidates.append(tag)
    used_curves = set(inflow) |set(outflow) |set(walls)
    obstacle = [c for c in obstacle_candidates if c not in used_curves]
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
ct = mesh_data.cell_tags
ct.name = "Cell tags"

# this is the bit that writes mesh to file

folder = "Biofilm Meshes"
file_path = os.path.join(folder, "arc_biofilm_mesh.xdmf")

with XDMFFile(mesh_comm, file_path, "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)
    xdmf.write_meshtags(ct, mesh.geometry)

gmsh.finalize()
print("facet tag 5:", len(ft.find(5)))
print("Mesh generation complete.")