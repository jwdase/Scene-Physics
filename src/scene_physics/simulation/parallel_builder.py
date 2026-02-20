"""
Creates multiple worlds using the Newton API. It does this by defining one Model that contains N isolated physics enviroments. Bodies in world 0 don't collide with bodies in world 1. The GPU solves all of this in one pass.
"""

import newton
import warp as wp


def build_parallel_worlds(base_builder_fn, num_worlds):
    """
    The ground plane is shared across all worlds (world=-1). Each world
    gets its own copy of the scene objects, with collision isolated
    between worlds by Newton's world grouping.

    Args:
        base_builder_fn: callable() -> (builder, bodies_dict)
            Must return a fresh ModelBuilder (without ground plane) and a dict
            mapping body names to Body objects. Called once per world.
        num_worlds: number of parallel worlds to create

    Returns:
        combined_builder: ModelBuilder with all worlds (not yet finalized)
        world_bodies: list[dict] — per-world mapping of body names to Body objects
        body_index_map: dict mapping (world_idx, body_name) -> index in state.body_q
    """
    # Main builder with shared ground
    main_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    main_builder.current_world = -1
    main_builder.add_ground_plane()

    world_bodies = []
    body_index_map = {}

    for world_idx in range(num_worlds):
        # Track body count before adding this world's builder
        body_offset = len(main_builder.body_q)

        # Build a fresh single-world scene
        world_builder, bodies_dict = base_builder_fn()

        # Add all bodies from this world's builder into the main builder
        main_builder.add_builder(world_builder, world=world_idx) 
        # TODO - Replace add_builder with  add_world

        # Record index mappings — each body's shape_start_index in the
        # sub-builder becomes body_offset + original_index in the combined builder
        remapped_bodies = {}
        for name, body in bodies_dict.items():
            global_index = body_offset + body.shape_start_index
            # Update the body's shape_start_index to reflect its position
            # in the combined builder
            body.shape_start_index = global_index
            body.builder = main_builder
            remapped_bodies[name] = body
            body_index_map[(world_idx, name)] = global_index

        world_bodies.append(remapped_bodies)

    return main_builder, world_bodies, body_index_map
