import copy
import time

import pyvista as pv
import numpy as np

class SceneVisualizer:
    def __init__(
        self, recorder, bodies, FPS, camera_position=None, background_color="white"
    ):
        self.recorder = recorder
        self.bodies = bodies

        self.camera_position = (
            camera_position
            if camera_position is not None
            else [
                (20, 20, 20),
                (0, 0, 0),
                (0, 1, 0),
            ]
        )

        self.background_color = background_color

        self.color = self.gen_colors()
        self.FPS = FPS

    def gen_colors(self):
        num_bodies = len(self.bodies)

        colors = [
            # "red",
            "green",
            "blue",
            "white",
            "black",
            "yellow",
            "cyan",
            "magenta",
            "lightblue",
            "darkblue",
            "brown",
            "beige",
        ]

        return [colors[i % len(colors)] for i in range(num_bodies)]

    def render(self, output_filename="scene_visualization.mp4"):
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(self.background_color)
        plotter.add_axes()

        # Ground plane
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
        plotter.add_mesh(plane, color="lightgray", opacity=0.8)

        # Plotter position
        plotter.camera_position = self.camera_position

        initial_state = self.recorder.history[0]

        actors = []
        for i, body in enumerate(self.bodies):
            actor = plotter.add_mesh(
                body.to_pyvista(initial_state).copy(),
                color=self.color[i],
                smooth_shading=True,
            )
            actors.append(actor)

        plotter.open_movie(output_filename, framerate=self.FPS, quality=9)

        for frame_idx, state in enumerate(self.recorder.history):
            for i, actor in enumerate(actors):
                temp_mesh = self.bodies[i].to_pyvista(state).copy()
                actor.mapper.SetInputData(temp_mesh)

            plotter.write_frame()

            if frame_idx % 50 == 0:
                print(f"Rendering frame {frame_idx}/{len(self.recorder.history)}")

        plotter.close()

        print(f"Visualization saved to {output_filename}")

    def gen_depth(self):
        """
        Generates depth so we can compare similarity
        """

        # Setup screen
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_position = self.camera_position

        # We want to check stability against the final state
        final_state = self.recorder.history[-1]

        # Add all the bodies
        for i, body in enumerate(self.bodies):
            plotter.add_mesh(
                body.to_pyvista(final_state).copy(),
                color=self.color[i],
                smooth_shading=True,
            )
        
        # Create projection
        plotter.screenshot('file.png')

        return np.array(plotter.get_image_depth())
        

        

