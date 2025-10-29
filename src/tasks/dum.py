import warp as wp
import newton

class RigidBody:
    def __init__(self, builder, position, quat=None, mass=1.0, cfg=None):
        self.builder = builder
        self.mass = mass
        self.position = wp.vec3(*position)
        self.quat = quat if quat is not None else wp.quat_identity()

        # Use default shape config if none given
        self.cfg = cfg if cfg is not None else builder.default_shape_cfg
        self.body = self.builder.add_body(
            xform=wp.transform(self.position, self.quat),
            mass=self.mass
        )
    
    def add_shape(self):
        """Implemented by subclasses"""
        raise NotImplementedError
