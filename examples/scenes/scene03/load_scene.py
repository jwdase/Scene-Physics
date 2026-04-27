import warp as wp
import newton
import numpy as np

from dataclasses import dataclass, field

NUM_WORLDS = 5

@dataclass
class Scene_Makeup:
    static : set[str] = field(default_factory=set) 
    observed : set[str] = field(default_factory=set) 
    hidden : set[str] = field(default_factory=set)

    def get_type(self, name):
        if name in self.static:
            return Static
        elif name in self.observed:
            return Observed
        elif name in self.hidden:
            return Hidden
        else:
            return None

    def __contains__(self, name):
        return self.get_type(name) is not None

class Body:
    def __init__(self):
        self.allocs = []

    def add(self, i):
        self.allocs.append(i)

    def finalize(self, model):
        self.allocs = np.array(self.allocs)


class Static(Body):
    pass

class Observed(Body):
    pass

class Hidden(Body):
    pass


@dataclass
class Object_Collection:
    objects : dict[str, Body] = field(default_factory=dict)

    def finalize(self, model):
        for obj in self.objects.values(): obj.finalize(model)

    def __setitem__(self, key, value):
        self.objects[key] = value

    def __contains__(self, value):
        return value in self.objects.keys()

    def __getitem__(self, value):
        return self.objects[value]


scene_usd = ("scene01_physics.usdc")

scene_makeup = Scene_Makeup(
        static=['dining_room_table'],
        observed=['coffee_0023', 'soap_dispenser_01'],
        hidden=['f10_apple_iphone_4'],
        )

blueprint = newton.ModelBuilder(up_axis=newton.Axis.Z)
blueprint.add_usd(scene_usd)

# Replicate Across N Worlds
builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
builder.add_ground_plane()
builder.replicate(blueprint, NUM_WORLDS, spacing=(0.0, 0.0, 0.0))

model = builder.finalize()


def insert_bodies(model, scene_makeup):
    objects = Object_Collection()

    for i, body_name in enumerate(model.body_key):
        name = body_name.split('/')[-1]
        print(name)

        if name not in objects:
            assert name in scene_makeup, f"Specification did not include {name}"
            objects[name] = scene_makeup.get_type(name)()
    
        objects[name].add(i)

    objects.finalize(model)

    return objects

