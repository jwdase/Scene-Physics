from scene_physics.properties.material import Material


Dynamic_Material = Material(mu=0.8, restitution=0.3, contact_ke=2e5, contact_kd=5e3, density=1e3)

Still_Material = Material(density=0.0)

