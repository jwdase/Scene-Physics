from scene_physics.properties.shapes import Parallel_Mesh, Parallel_Static_Mesh


def build_worlds(worlds, objects):
    """
    Fills each world with meshes depending on whether they're dynamic
    meshes or not, then finalizes the model

    Args:
        worlds: world builder
        stat_obj: objects that exist in all sim and don't move
        dyn_obj_ob: all observable objects
        dyn_obj_un: all unobservable objects

    Returns:
        model: Our finalized model
        objects: list of objects in order which they should be inserted
    """
    all_objects = objects["static"] + objects["observed"] + objects["unobserved"]

    # Insert all static objects
    for obj in objects["static"]:
        print(type(obj))
        assert isinstance(obj, Parallel_Static_Mesh), "Must be static"
        obj.insert_object_static(worlds)

    # Insert all observed dynamic objects
    for i in range(worlds.num_worlds):
        for obj in objects["observed"]:
            assert isinstance(obj, Parallel_Mesh), "Must be dynamic"
            obj.insert_object(worlds, i)

        for obj in objects["unobserved"]:
            assert isinstance(obj, Parallel_Mesh), "Must be dynamic"
            obj.insert_object(worlds, i)

    # Finalize and assign pointers to objects
    model = worlds.finalize()
    for obj in all_objects:
        obj.give_finalized_world(model)
    
    # Take state of correct placement
    for obj in all_objects:
        obj.move_to_target()
    
    target = model.state()

    # Hide all objects that are not static
    for obj in (objects["observed"] + objects["unobserved"]):
        obj.freeze_finalized_body()
    
    return model, target