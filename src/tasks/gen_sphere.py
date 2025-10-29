import pyvista as pv

box = pv.Box(bounds=(-1, 1, -1, 1, -1, 1))
box = box.triangulate().clean()

tetra = box.delaunay_3d(alpha=1.0)
tetra.save("box_tetra.vtu")

tetra.plot(show_edges=True)
