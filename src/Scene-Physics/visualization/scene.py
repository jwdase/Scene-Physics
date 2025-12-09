import copy
import time

import pyvista as pv
import numpy as np
import jax.numpy as jnp

class Visualizer:
    """
    General class for visualizing our inputs
    """
    def __init__(self, bodies, camera_position, background_color='white'):
        self.camera_intrinsics = None
        self.point_cloud_camera = [(1, 1.5, 3), (0, 1, 0), (0, 1, 0)]

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


    def gen_camera(self):
        """
        returns camera intrinsics (call it once)
        """

        # Generates plotter 
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_position = self.point_cloud_camera

        # Get Camere properties
        camera = plotter.camera
        near, far = camera.clipping_range
        width, height = plotter.render_window.GetSize()

        # Get field of view in degrees
        fov_degrees = camera.view_angle
        fov_rad = np.radians(fov_degrees)

        # Standard pinhole intrinsics
        fy = height / (2 * np.tan(fov_rad / 2))
        fx = fy * (width / height)

        cx = width / 2
        cy = height / 2

        return {
            'height' : height,
            'width' : width,
            'fx' : fx,
            'fy' : fy,
            'cx' : cx,
            'cy' : cy,
            'near' : near,
            'far' : far
        }

    def set_intrinsics(self, intrinsic_func):
        """ Set Camera Intrinsics Once"""
        camera_data = self.gen_camera()

        # Create camera_intrinsics data once
        self.camera_intrinsics = intrinsic_func(
            height=camera_data["height"],
            width=camera_data["width"],
            fx=camera_data["fx"],
            fy=camera_data["fy"],
            cx=camera_data["cx"],
            cy=camera_data["cy"],
            near=camera_data["near"],
            far=camera_data["far"],
        )

    def plot_point_maps(self, point_cloud, location):
        pts = np.array(point_cloud).reshape(-1, 3)
        pts[:,2] = -pts[:,2]
        pc = pv.PolyData(pts)
        pc.plot(
            point_size=5,
            style="points",
            screenshot=location,
        )

    def point_cloud(self, unprojected_depth_func, clip=True):
        """
        takes in a class that calculates camera intrinsics
        passed into depth function that yields a point cloud
        """

        # Ensures we have the camera intrinsics function
        assert self.camera_intrinsics is not None, "Camera Intrinsics cannot be None - set it first"
        
        # Get the depth
        depth = self.gen_3dPoint()

        # Flip depth from input
        zlinear = np.array(depth, dtype=np.float32)
        point_cloud = unprojected_depth_func(zlinear, self.camera_intrinsics)

        # Turn into np array
        point_cloud = np.array(point_cloud)

        # Remove all values too large
        if clip is True:
            point_cloud[np.isnan(point_cloud)] = 1000
            point_cloud = jnp.array(point_cloud)

        return point_cloud

    def gen_png(self, name):
        plotter = self.fill_scene()
        plotter.screenshot(name)

    def gen_3dPoint(self):
        plotter = self.fill_scene()
        plotter.show()

        return plotter.get_image_depth()


class VideoVisualizer(Visualizer):
    def __init__(
        self, recorder, bodies, FPS, camera_position=None, background_color="white"
    ):
        super().__init__(bodies, camera_position, background_color)

        self.recorder = recorder
        self.FPS = FPS

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

    def fill_scene(self):
        """Places the objects in the scene"""
        # Setup screen
        plotter = pv.Plotter(off_screen=True)
        plotter.camera_position = self.point_cloud_camera
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)

        # We want to check stability against the final state
        final_state = self.recorder.history[-1]

        # Add all the bodies
        for i, body in enumerate(self.bodies):
            plotter.add_mesh(
                body.to_pyvista(final_state).copy(),
                color=self.color[i],
                smooth_shading=True,
            )

        return plotter


class PyVistaVisuailzer(Visualizer):
    """Class for getting png and comparisons - no simulation"""
    def __init__(self, bodies, camera_position, background_color='white'):
        super().__init__(bodies, camera_position, background_color)

    def fill_scene(self):
        """Places the objects in the scene"""

        # Setup plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(self.background_color)
        plotter.add_axes()

        # Ground plane
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=25, j_size=25)
        plotter.add_mesh(plane, color="lightgray", opacity=0.8)

        # Plotter position
        plotter.camera_position = self.camera_position

        for i, body in enumerate(self.bodies):
            plotter.add_mesh(
                body.to_pyvista_png(),
                color=self.color[i],
                smooth_shading=True
            )

        return plotter
