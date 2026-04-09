"""
Creates multiple worlds using the Newton API. It does this by defining one Model that contains N isolated physics enviroments. Bodies in world 0 don't collide with bodies in world 1. The GPU solves all of this in one pass.
"""

import newton

def allocate_worlds(n, stable_mesh=None):
    """
    Allocates n worlds following a "blueprint" - which is just a stand in for an
    empty world

    Args:
        num_worlds : number of parrallel worlds to create
        stable_mesh : list[StableMeshBody] list of meshes that exist in all scenes

    Returns:
        combines_builder : ModelBuilder with all worlds (not yet finalized)
    """
    
    # Main builder with shared ground plane
    main_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    main_builder.current_world = -1
    main_builder.add_ground_plane()

    # Create n of these worlds
    main_builder.replicate(newton.ModelBuilder(up_axis=newton.Axis.Y), num_worlds=n)

    return main_builder


