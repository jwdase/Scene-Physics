import numpy as np
from scipy.spatial.transform import Rotation
import json

ground_truth = {
    "dining_room_table":  np.array([ 0.0034, -0.0037,  0.0107,    
  0.0, 0.0, 0.0, 1.0]),                                             
      "f10_apple_iphone_4": np.array([ 0.2504,  0.2338,  0.7505,
  0.0, 0.0, 0.0, 1.0]),                                             
      "coffee_0023":        np.array([ 0.2699, -0.0671,  0.7458,
  0.0, 0.0, 0.0, 1.0]),                                             
      "soap_dispenser_01":  np.array([-0.2316, -0.0927,  0.7458,
  0.0, 0.0, 0.0, 1.0]),                                             
  }




x = {}

for name, pos in ground_truth.items():
    if name == "dining_room_table":
        continue

    pos[:3] += np.random.normal(loc=0.0, scale=0.1, size=(3))

    angles = np.random.normal(loc=0.0, scale=0.1, size=(3))
    perturbation = Rotation.from_rotvec(angles)
    current = Rotation.from_quat(pos[3:])

    pos[3:] = (perturbation * current).as_quat()

    x[name] = pos
    print(name, pos)

with open("scene01_truth.json", "w") as f:
    json.dump({k : v.tolist() for k, v in ground_truth.items()}, f, indent=4)

with open("scene01_priors.json", "w") as f:
    json.dump({k : v.tolist() for k, v in x.items()}, f, indent=4)