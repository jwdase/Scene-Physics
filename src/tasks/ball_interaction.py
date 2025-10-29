import math

import numpy as np
import warp as wp

wp.init()

N_BALLS = 100

device = 'cuda:0'

vel = wp.zeros((N_BALLS,), dtype=wp.vec3, device=device)

def gen_dimensions(num, length, width):
    """
    Reutrns the number of samples over height
    and width
    """

    samples = round(math.sqrt(num))

    x_loc = np.linspace(start=0, stop=width, num=samples)
    y_loc = np.linspace(start=0, stop=length, num=samples)

    return x_loc, y_loc, samples**2

def gen_balls(num, length, width, height):
    """
    Generate balls evenly spaced over an array
    """

    # Generate locations to iterate over
    x_loc, y_loc, samples = gen_dimensions(num, length, width)

    # fill position vector
    pos = np.zeros(shape=(samples, 3))

    for ix, x_val in enumerate(x_loc):
        for iy, y_val in enumerate(y_loc):

            index = ix * len(x_loc) + iy

            pos[index][0], pos[index][1], pos[index][2] = (
                x_val, y_val, height
            ) 

    return wp.array(pos, dtype=wp.vec3)

def gen_velocity(num, length, width):
    """
    Give all balls zero velocity and drop them from 
    a given height
    """

    # Generate number of samples
    _, _, samples = gen_dimensions(num, length, width)

    vel = np.zeros(shape=(samples, 3))

    return  wp.array(vel, dtype=wp.vec3)


# Our Bowl will have geometry: f(x, y, z) = y - k(x^2 + z^2)
# Where f(x, y, z) = 0 means that the ball is on surface
# Our Gradient will be <-[-2Kx, 1, -2*K*z]

K = .6

@wp.func
def bowl_height(x : float, z: float):
    """
    Calculates height of bowl
    """
    return K * (x**2 + z**2)

@wp.func
def bowl_normal(x : float, z: float):
    """
    Calculates normal value of bowl
    """
    n = wp.vec3(-2*K*x, 1.0, 2*K*x)

    return n / w.length(n) 

@wp.kernel
