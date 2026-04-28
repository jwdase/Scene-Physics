import json

import numpy as np

from dataclasses import dataclass, field

from scene_physics.sampling.proposals import Proposer, Prior

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
    def __init__(self, name):
        self.name = name
        self.allocs = []

        self.proposer = None
        self.prior = None

    def add(self, i):
        self.allocs.append(i)

    def finalize(self, model):
        self.allocs = np.array(self.allocs)


class Static(Body):
    pass

class Dynamic(Body):
    def set_proposer(self, rand_seed, prior_dict, proposer):
        self.prior = Prior(prior_dict)
        self.proposer = proposer(rand_seed, self.prior)

class Observed(Dynamic):
    pass

class Hidden(Dynamic):
    pass


@dataclass
class Object_Collection:
    objects : dict[str, Body] = field(default_factory=dict)

    def finalize(self, model):
        for obj in self.objects.values(): obj.finalize(model)

    def assign_priors(self, prior_json : str,  proposer : Proposer, rng : np.random.Generator):
        with open(prior_json, "r") as f:
            priors = json.load(f)

        children = rng.spawn(len(priors))

        for i, (name, prior) in enumerate(priors.items()):
            self[name].set_proposer(children[i], prior, proposer)

    def __setitem__(self, key, value):
        self.objects[key] = value

    def __contains__(self, value):
        return value in self.objects.keys()

    def __getitem__(self, value):
        return self.objects[value]


def object_collection(model, scene_makeup) -> Object_Collection:
    objects = Object_Collection()

    for i, body_name in enumerate(model.body_key):
        name = body_name.split('/')[-1]

        if name not in objects:
            assert name in scene_makeup, f"Specification did not include {name}"
            objects[name] = scene_makeup.get_type(name)(name)
    
        objects[name].add(i)

    objects.finalize(model)

    return objects

