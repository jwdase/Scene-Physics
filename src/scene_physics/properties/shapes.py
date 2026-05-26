import json

import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from dataclasses import dataclass, field

from scene_physics.sampling.proposals import Proposer, Prior

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
    def __init__(self, name, num_worlds):
        self.name = name
        self.allocs = []
        self.num_worlds = num_worlds

        self.correct = None

    def add(self, i):
        self.allocs.append(i)

    def finalize(self, model):
        self.allocs = np.array(self.allocs)

    def __str__(self):
        return f"Body Name: {self.name}"
    

class Static(Body):
    pass

class Dynamic(Body):
    def __init__(self, name, num_worlds):
        super().__init__(name, num_worlds)

        self.proposer = None
        self.prior = None
        self.plots = []


    def set_proposer(self, rand_seed, prior_dict, proposer):
        self.prior = Prior(prior_dict)
        self.proposer = proposer(rand_seed, self.num_worlds, self.prior)

    def initialize(self, scene):
        proposals = self._warp_to_numpy(scene)
        proposals[self.allocs] = self.proposer.initial_proposal()
        scene.body_q = self._numpy_to_warp(proposals)

        return scene
    
    def propose(self, scene, likelihood):
        proposals = self._warp_to_numpy(scene)

        # Index Position, Likelihood, Then propose
        positions, likelihoods = proposals[self.allocs], likelihood
        proposals[self.allocs] = self.proposer.propose(positions, likelihoods)

        # Save a copy of highest likelihood location (allocs[0] = best world's body)
        self.plots.append(proposals[self.allocs].copy())

        scene.body_q = self._numpy_to_warp(proposals)
        return scene
    
    def _warp_to_numpy(self, state):
        return state.body_q.numpy()
    
    def _numpy_to_warp(self, proposals):
        return wp.array(proposals, dtype=wp.transformf)
    
    def gen_plots(self, save_dir):
        self._plot_xy_position(save_dir)

    def _plot_xy_position(self, save_dir):
        assert self.correct is not None, "Correct value must be assigned to generate plots"

        positions = np.array(self.plots)
        n, nw, size = positions.shape

        flat = positions.reshape(-1, size)
        iters = np.repeat(np.arange(n), nw)

        fig, ax = plt.subplots()
        sc = ax.scatter(flat[:, 0], flat[:, 1], c=iters, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="Iteration")

        # Plot the correct position
        ax.scatter(self.correct[0], self.correct[1], marker=(7, 1, 0), s=50, color='gold')


        ax.set_title(f"XY Position of {self.name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.savefig(f"{save_dir}/{self.name}_position.png")
        plt.close(fig)


class Observed(Dynamic):
    pass

class Hidden(Dynamic):
    pass


@dataclass
class Object_Collection:
    objects : dict[str, Body] = field(default_factory=dict)
    dynamic : list[Dynamic] = field(default_factory=list)

    def finalize(self, model):
        for obj in self.objects.values(): obj.finalize(model)

    def assign_priors(self, prior_json : str,  proposer : Proposer, rng : np.random.Generator):
        with open(prior_json, "r") as f:
            priors = json.load(f)

        children = rng.spawn(len(priors))

        for i, (name, prior) in enumerate(priors.items()):
            self[name].set_proposer(children[i], prior, proposer)



    def assign_correct(self, truth_json : str):
        with open(truth_json, "r") as f:
            truth = json.load(f)

        for name, xform in truth.items():
            if name in self.objects:
                obj = self.objects[name]

                obj.correct = np.array(xform)

    def initialize(self, scene):
        for obj in self.objects.values():
            if isinstance(obj, Dynamic) and obj.prior is not None:
                scene = obj.initialize(scene)
                if isinstance(obj, Hidden):
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

