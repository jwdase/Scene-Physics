import json
from dataclasses import dataclass, field

from scene_physics.properties.shapes import *


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


@dataclass
class Object_Collection:
    objects : dict[str, Body] = field(default_factory=dict)
    dynamic : list[Dynamic] = field(default_factory=list)

    def finalize(self, model):
        for obj in self.objects.values(): obj.finalize(model)

    def assign_priors(self, prior_json : str,  proposer : Proposer, iterations :int, rng : np.random.Generator):
        with open(prior_json, "r") as f:
            priors = json.load(f)

        children = rng.spawn(len(priors))

        for i, (name, prior) in enumerate(priors.items()):
            self[name].set_proposer(children[i], prior, proposer, iterations)

    def assign_correct(self, truth_json : str):
        with open(truth_json, "r") as f:
            truth = json.load(f)

        for name, xform in truth.items():
            if name in self.objects:
                obj = self.objects[name]

                obj.correct = np.array(xform)

    def initialize(self, scene):
        for obj in self.objects.values():

            # NOTE only samples for occluded object
            if isinstance(obj, Hidden):
                scene = obj.initialize(scene)
                self.dynamic.append(obj)

        return scene
    
    def get_random(self):
        return np.random.choice(self.dynamic)

    def gen_plots(self, save_dir):
        for obj in self.dynamic:
            obj.gen_plots(save_dir)

    def __setitem__(self, key, value):
        self.objects[key] = value

    def __contains__(self, value):
        return value in self.objects.keys()

    def __getitem__(self, value):
        return self.objects[value]


def object_collection(model, scene_makeup, num_worlds) -> Object_Collection:
    objects = Object_Collection()

    for i, body_name in enumerate(model.body_key):
        name = body_name.split('/')[-1]

        if name not in objects:
            assert name in scene_makeup, f"Specification did not include {name}"
            objects[name] = scene_makeup.get_type(name)(name, num_worlds)
    
        objects[name].add(i)

    objects.finalize(model)

    return objects

