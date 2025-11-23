# Do Imports
import sys
sys.path.append("/orcd/home/002/jwdase/genjax_tutorial/modules/b3d/src")

# Import b3d feautures
from b3d.chisight.dense.likelihoods.image_likelihoods import threedp3_likelihood_per_pixel_old
from b3d.camera import unproject_depth
import b3d

# Pass in intrinsics
intrinsics = b3d.camera.Intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, near=near, far=far)

# Returns log similariy
def likelihood_func(observed_depth, rendered_depth, intrinsics):

    observed_xyz = unproject_depth(observed_depth, intrinsics)
    rendered_xyz = unproject_depth(rendered_depth, intrinsics)
    image_likelihoods = threedp3_likelihood_per_pixel_old(
        observed_xyz,
        rendered_xyz,
        0.0001,
        0.00001,
        1.0,
        3,
    )

    return image_likelihoods["pix_score"].sum()