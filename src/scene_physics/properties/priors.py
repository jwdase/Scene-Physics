from dataclasses import dataclass, field

@dataclass
class Priors:
    """
    Lists out the priors for each sampler this is a dataclass, all in meters
    """
    # Priors
    init_mean: float = 0.0
    init_std: float = 0.01

    # Sampling Params
    pos_std: float = 0.1
    rot_std: float = 0.1
    total_iter: int = 40

    # bounds
    x_max: float= 1.0
    x_min: float= -1.0
    z_max: float=1.0
    z_min: float=-1.0

@dataclass
class SimulationObjects:
    """
    Holds all the objects for forward physics simulation
    """

    observed: list = field(default_factory=list)
    unobserved: list = field(default_factory=list)
    static: list = field(default_factory=list)

    @property
    def all_sampled(self):
        """Returns sampled objects"""
        return self.observed + self.unobserved

    @property
    def all_bodies(self):
        """Returns all objects"""
        return self.observed + self.static + self.unobserved

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"{key} is not a valid field")

    def __str__(self):
        print("======= Observed =======")
        for obj in self.observed:
            print(obj.name)

        print("======= Unobserved =======")
        for obj in self.unobserved:
            print(obj.name)

        print("======= Static ========")
        for obj in self.static:
            print(obj.name)

        return ""


