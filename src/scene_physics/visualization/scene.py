import copy
import time

import pyvista as pv
import numpy as np
import jax.numpy as jnp
import newton
import warp as wp
from newton.solvers import SolverXPBD


from scene_physics.properties.shapes import Parallel_Mesh

DEFAULT_CAMERA= [(20, 20, 20), (0, 0, 0), (0, 1, 0),]


class PyVistaVisualizer:
    """
    General class for visual inputs
    """
    def __init__(self, bodies, num_worlds, camera_pos=DEFAULT_CAMERA, background_color='white'):
        self.bodies = bodies.all_bodies
        self.camera_pos = camera_pos 
        self.background_color = background_color
        self.color = self._gen_colors()
        self.num_worlds = num_worlds

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

    def render_final_scene(self, output_filename, frames=200, dt=0.016, fps=60,
                           substeps=4, iterations=16, settling_frames=50):
        """Creates a render of the final scene

        Args:
            output_filename: path for the output .mp4
            frames: number of visible frames to record
            dt: total time per visible frame (subdivided by substeps)
            fps: playback framerate for the output video
            substeps: physics substeps per visible frame (higher = more stable)
            iterations: XPBD constraint solver iterations per substep
            settling_frames: physics frames (at substep dt) to run before
                recording, allowing small initial penetrations to resolve
        """

        # Build the worlds and run physics
        model, body_idx = self._build_worlds()
        history = self.run_forward_physics(
            model, frames, dt,
            substeps=substeps,
            iterations=iterations,
            settling_frames=settling_frames,
        )

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

    def run_forward_physics(self, model, frames, dt, substeps=4,
                            iterations=16, settling_frames=50,
                            linear_damping=0.1):
        """Runs forward physics and generates a history of the movement.

        Uses substeps to subdivide each visible frame into smaller physics
        steps, which makes XPBD much more robust against small initial
        penetrations and thin-geometry tunneling.

        Args:
            model: finalized Newton Model
            frames: number of visible frames to record
            dt: total time per visible frame (subdivided by substeps)
            substeps: number of physics steps per visible frame
            iterations: XPBD solver iterations per substep
            settling_frames: extra physics steps (at sub_dt) run before
                recording begins, so initial overlaps are resolved gently
            linear_damping: per-frame linear velocity damping factor applied
                after each recorded frame to prevent energy accumulation
        """
        sub_dt = dt / substeps

        # rigid_contact_relaxation reduced from 0.75 → 0.4 to limit
        # energy injection from XPBD contact overcorrection in persistent
        # contacts (e.g. objects resting on a surface for many frames).
        solver = SolverXPBD(
            model,
            rigid_contact_relaxation=0.4,
            iterations=iterations,
            angular_damping=0.2,
            enable_restitution=False,
        )
        control = model.control()

        state_0 = model.state()
        state_1 = model.state()

        # Settling phase: resolve small initial penetrations before recording.
        # Uses heavier damping so objects don't gain momentum from overlap
        # correction.
        for _ in range(settling_frames):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        # Zero out velocities after settling so objects start at rest
        state_0.body_qd.zero_()

        # Record visible frames, each subdivided into substeps.
        # Linear damping is applied once per visible frame to prevent slow
        # energy accumulation over long simulations. The numpy round-trip is
        # done per-frame (not per-substep) to keep GPU↔CPU transfers minimal.
        damping_factor = 1.0 - linear_damping
        history = [state_0.body_q.numpy().copy()]
        for _ in range(frames):
            for _sub in range(substeps):
                state_0.clear_forces()
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, sub_dt)
                state_0, state_1 = state_1, state_0
            # Damp all velocities to prevent unbounded energy accumulation.
            # Applied once per frame to keep GPU↔CPU transfers minimal.
            vel = state_0.body_qd.numpy()
            vel *= damping_factor
            state_0.body_qd = wp.from_numpy(vel, dtype=state_0.body_qd.dtype, device="cuda")
            history.append(state_0.body_q.numpy().copy())

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
