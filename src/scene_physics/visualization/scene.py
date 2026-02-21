import copy
import time

import pyvista as pv
import numpy as np
import jax.numpy as jnp

class PyVistaVisualizer:
    """
    General class for visual inputs
    """
    def __init__(self, bodies, camera_pos=None, background_color='white'):
        self.bodies = bodies
        self.camera_pos = camera_pos if camera_pos is not None else self._gen_camera()
        self.background_color = background_color
        self.color = self._gen_colors()

    def _gen_camera(self):
        return [(20, 20, 20), (0, 0, 0), (0, 1, 0),]

    def _gen_colors(self):
        num_bodies = len(self.bodies)
        colors = ["green", "blue", "white", "black", "yellow",]
        return [colors[i % len(colors)] for i in range(num_bodies)]


    def _fill_scene(self, pos, time, world_id):
        """
        Creates a scene with each object placed

        Args:
            pos : Tensor[Time, Location]. Time is frame number and Location
            is a copy of scene.body_q.numpy()
            world_id : Specifies a world_id for rendering

        """
        numpy_bd_q = pos[time, :]  # Fix 3: use `time` index, not hardcoded 0

        # Setup plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(self.background_color)
        plotter.add_axes()

        # Ground plane
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
        plotter.add_mesh(plane, color="lightgray", opacity=0.8)

        # Plotter position
        plotter.camera_position = self.camera_pos  # Fix 1

        for i, body in enumerate(self.bodies):
            plotter.add_mesh(
                body.to_pyvista_png(numpy_bd_q, world_id),
                color=self.color[i],
                smooth_shading=True
            )

        return plotter


    def gen_png(self, pos, name, time=0, world_id=0):
        plotter = self._fill_scene(pos, time, world_id)  # Fix 2: correct name and pass pos
        plotter.screenshot(name)


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
