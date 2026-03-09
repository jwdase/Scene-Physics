import copy
import time

import pyvista as pv
import numpy as np
import jax.numpy as jnp
import newton
import warp as wp
from newton.solvers import SolverXPBD


from scene_physics.properties.shapes import Parallel_Mesh

class PyVistaVisualizer:
    """
    General class for visual inputs
    """
    def __init__(self, bodies, num_worlds, camera_pos=None, background_color='white'):
        self.bodies = self._get_bodies(bodies)
        self.camera_pos = camera_pos if camera_pos is not None else self._gen_camera()
        self.background_color = background_color
        self.color = self._gen_colors()
        self.num_worlds = num_worlds

    def _get_bodies(self, bodies):
        return bodies["observed"] + bodies["static"] + bodies["unobserved"]

    def _gen_camera(self):
        return [(20, 20, 20), (0, 0, 0), (0, 1, 0),]

    def _gen_colors(self):
        num_bodies = len(self.bodies)
        colors = ["green", "blue", "white", "black", "yellow",]
        return [colors[i % len(colors)] for i in range(num_bodies)]


    def _fill_scene(self, scene, world_id, func=None):
        """
        Creates a scene with each object placed

        Args:
            scene : a state on the physic simulation
            world_id : Specifies a world_id for rendering

        """
        
        # Rewrite to insert
        pos = scene.body_q.numpy() if scene is not None else None

        # Assign visualizaiton function
        if func is None:
            func = lambda body, pos, wid: body.to_pyvista(pos, wid)

        # Setup plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(self.background_color)
        plotter.add_axes()

        # Ground plane
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
        plotter.add_mesh(plane, color="lightgray", opacity=0.8)

        # Plotter position
        plotter.camera_position = self.camera_pos

        for i, body in enumerate(self.bodies):
            plotter.add_mesh(
                func(body, pos, world_id),
                color=self.color[i],
                smooth_shading=True
            )

        return plotter

    def gen_png(self, scene, name, world_id=0):
        plotter = self._fill_scene(scene, world_id)
        plotter.screenshot(name)
        plotter.close()

    def show_final_scene(self, name):
        plotter = self._fill_scene(None, None, func=Parallel_Mesh.to_pyvista_final)
        plotter.screenshot(name)

    def gen_multi_world_png(self, scene, name):
        """Generates a plot for each scene across all worlds"""
        for i in range(self.num_worlds):
            temp_name = f"{name}/world_{i}.png"
            self.gen_png(scene, name=temp_name, world_id=i)


class PhysicsVideoVisualizer(PyVistaVisualizer):
    def __init__(
        self, bodies, FPS, camera_pos=None, background_color="white"
    ):
        super().__init__(bodies, None, camera_pos, background_color)

    def render_final_scene(self, output_filename, frames=200, dt=0.016, fps=60):
        """Creates a render of the final scene"""
        
        # Build the worlds and run physics
        model, body_idx = self._build_worlds()
        history = self.run_forward_physics(model, frames, dt)

        # Render to target destination
        self.render(history, body_idx, output_filename, fps)
        
    def _build_worlds(self):
        """ Generate our Newton world for forward physics"""
        
        # Ensure all bodies are finalized
        for body in self.bodies:
            assert body.final_position is not None, f"Body: {body.name} not finalized"

        # Generate the builder
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
        builder.add_ground_plane()

        # Add each individual body
        body_idx = {}
        for body in self.bodies:
            body_idx[hash(body)] = len(builder.body_q)
            b = builder.add_body(self._get_xform(body))
            builder.add_shape_mesh(body=b, mesh=body.nt_mesh, cfg=body.cfg)

        return builder.finalize(), body_idx

    @staticmethod
    def _get_xform(body):
        """Creates x_form for body insert"""
        pos = body.final_position  # [x, y, z, qx, qy, qz, qw]
        return wp.transform(
                pos[:3].tolist(),
                wp.quat(float(pos[3]), float(pos[4]), float(pos[5]), float(pos[6])),
            )

    def run_forward_physics(self, model, frames, dt):
        """Runs forward physics and generates a history of the movement"""

        solver = SolverXPBD(model, rigid_contact_relaxation=0.9, iterations=4, angular_damping=0.1, enable_restitution=False)
        control = model.control()

        state_0 = model.state()
        state_1 = model.state()

        # Run forward sim, collecting body_q snapshots
        history = [state_0.body_q.numpy().copy()]
        for _ in range(frames):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt)
            history.append(state_1.body_q.numpy().copy())
            state_0, state_1 = state_1, state_0

        return history

    def render(self, history, body_idx, output_filename, fps):
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(self.background_color)
        plotter.add_axes()

        # Ground plane
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
        plotter.add_mesh(plane, color="lightgray", opacity=0.8)

        # Plotter position and initial state
        plotter.camera_position = self.camera_pos
        
        # Insert Actors
        actors = []
        for i, body in enumerate(self.bodies):
            actor = plotter.add_mesh(
                body.pyvista_body(history[0][body_idx[hash(body)]]),
                color=self.color[i],
                smooth_shading=True,
            )
            actors.append((actor, body))
        
        # Run each frame
        plotter.open_movie(output_filename, framerate=fps, quality=9)
        for frame_idx in range(len(history)):
            for i, (actor, body) in enumerate(actors):
                temp_mesh = self.bodies[i].pyvista_body(history[frame_idx][body_idx[hash(body)]])
                actor.mapper.SetInputData(temp_mesh)
            plotter.write_frame()

            if frame_idx % 50 == 0:
                print(f"Rendering frame {frame_idx}/{len(history)}")

        plotter.close()

        print(f"Visualization saved to {output_filename}")
