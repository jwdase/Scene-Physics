import warp as wp
from newton._src.sim.builder import ModelBuilder

class Material:
    """
    Material Class - inherits from CFG
    """
    def __init__(self, mu=0.5, restitution=0.0, contact_ke=1e5, contact_kd=1e3, density=1000.0):
        self.mu = mu
        self.restitution = restitution
        self.contact_ke = contact_ke
        self.contact_kd = contact_kd
        self.density = density
        self.is_solid = True

    def to_cfg(self): 
        return ModelBuilder.ShapeConfig(restitution=self.restitution, mu=self.mu, is_solid=self.is_solid, density=self.density, has_shape_collision=True)
