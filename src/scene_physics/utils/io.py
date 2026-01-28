import json
import warp as wp
import numpy as np
import sys
import pyvista as pv

def load_stimuli_start(file, folder):
    """
    Loads the stimuli data sent from becket
    """
    with open(file, "r") as f:
        data = json.load(f)

    # Extract trial numbers
    trials = []
    for val in data.keys():
        try:
            int(val)
            trials.append(val)
        except ValueError:
            pass

    # Extract Ball and Ramp Name
    for i in range(len(trials)):
        ball = data[trials[i]]["conditions"]["ballType"].replace(
            "b", "B"
        )  # Remove if new config
        ball_scale = data[trials[i]]["conditions"]["ballScale"]

        ball_postions = data[trials[i]]["ball_positions"]["1"]["position"]
        ball_rotation = data[trials[i]]["ball_positions"]["1"]["rotation"]

        ramp = data[trials[i]]["conditions"]["rampType"].replace("r", "R")[:-1]
        rampDfriction = data[trials[i]]["conditions"]["rampDFriction"]
        rampSfriction = data[trials[i]]["conditions"]["rampSFriction"]

        yield (
            {
                "ball": f"{folder}/{ball}.obj",
                "ball_scale": ball_scale,
                "ball_postion": convert_positions(ball_postions),
                "ball_rotation": convert_rotation(ball_rotation),
                "ramp": f"{folder}/{ramp}.obj",
                "rampDfriction": rampDfriction,
                "rampSfriction": rampSfriction,
            }
        )


def convert_positions(positions):
    vals = dict_to_array_position(positions)
    return wp.vec3(vals[0], vals[1], vals[2])

def convert_rotation(rotation):
    vals = dict_to_array_rotation(rotation)
    return wp.quat(vals[0], vals[1], vals[2], vals[3])

def dict_to_array_rotation(d):
    return np.array([d['x'], d['y'], d['z'], d['w']])

def dict_to_array_position(d):
    """
    Converts a dictionary to a jax.numpy array
    NOTE: positive X axis in TDW is in the opposite direction of positive X axis in bayes3D. Therefore, we need to negate the X axis    
    """
    
    return np.array([-d['x'], d['y'], d['z']])


simulation_specifictation = load_stimuli_start(
    "objects/stimuli.json", "objects/local_models"
)

# Plot point cloud
def plot_point_maps(point_cloud, location):
    pts = np.array(point_cloud).reshape(-1, 3)
    pts[:,2] = -pts[:,2]
    pc = pv.PolyData(pts)
    pc.plot(
        point_size=5,
        style="points",
        screenshot=location,
    )