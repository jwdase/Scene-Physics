import copy
import time

import pyvista as pv
import numpy as np
import jax.numpy as jnp

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

    def show_final_scene(self, name):
        plotter = self._fill_scene(None, None, func=Parallel_Mesh.to_pyvista_final)
        plotter.screenshot(name)

    def gen_multi_world_png(self, scene, name):
        """Generates a plot for each scene across all worlds"""
        for i in range(self.num_worlds):
            temp_name = f"{name}/world_{i}.png"
            self.gen_png(scene, name=temp_name, world_id=i)


class VideoVisualizer(PyVistaVisualizer):
    def __init__(
        self, history, bodies, FPS, camera_pos=None, background_color="white"
    ):
        super().__init__(bodies, camera_pos, background_color)

        self.recorder = history # np.array: [time, positions]
        self.FPS = FPS

    def render(self, world_id=0, output_filename="scene_visualization.mp4"):
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(self.background_color)
        plotter.add_axes()

        # Ground plane
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
        plotter.add_mesh(plane, color="lightgray", opacity=0.8)

        # Plotter position and initial state
        plotter.camera_position = self.camera_pos  # Fix 1
        initial_state = self.recorder[0, :]

        actors = []
        for i, body in enumerate(self.bodies):
            actor = plotter.add_mesh(
                body.to_pyvista(initial_state, world_id),
                color=self.color[i],
                smooth_shading=True,
            )
            actors.append(actor)

        plotter.open_movie(output_filename, framerate=self.FPS, quality=9)

        for frame_idx in range(int(self.recorder.shape[0])):  # Fix 4: added colon
            for i, actor in enumerate(actors):
                temp_mesh = self.bodies[i].to_pyvista(self.recorder[frame_idx], world_id)  # Fix 5: removed trailing comma
                actor.mapper.SetInputData(temp_mesh)

            plotter.write_frame()

            if frame_idx % 50 == 0:
                print(f"Rendering frame {frame_idx}/{self.recorder.shape[0]}")  # Fix 6: use shape[0]

        plotter.close()

        print(f"Visualization saved to {output_filename}")
