import pyvista as pv  
  
# Create rounded cube surface  
rounded_cube_surface = pv.Superquadric(  
    theta_roundness=0.3,  
    phi_roundness=0.3,  
)  
  
# Fill it to create a solid volume  
solid_rounded_cube = rounded_cube_surface.delaunay_3d()

print(type(solid_rounded_cube))
print(isinstance(solid_rounded_cube, pv.UnstructuredGrid))