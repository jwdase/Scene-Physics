import json
import warp as wp

def load_stimuli_start(file, folder):
    """
    Loads the stimuli data sent from becket
    """
    with open(file, 'r') as f:
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
        ball = data[trials[i]]['conditions']['ballType'].replace('b', 'B') # Remove if new config
        ball_scale = data[trials[i]]['conditions']['ballScale']

        ball_postions = data[trials[i]]['ball_positions']['1']['position']
        ball_rotation = data[trials[i]]['ball_positions']['1']['rotation']

        ramp = data[trials[i]]['conditions']['rampType'].replace('r', 'R')[:-1]
        rampDfriction = data[trials[i]]['conditions']['rampDFriction']
        rampSfriction = data[trials[i]]['conditions']['rampSFriction']

        yield({
            'ball' : f"{folder}/{ball}.obj",
            'ball_scale' : ball_scale,
            'ball_postion' : convert_positions(ball_postions),
            'ball_rotation' : convert_rotation(ball_rotation),
            'ramp' : f"{folder}/{ramp}.obj",
            'rampDfriction' : rampDfriction,
            'rampSfriction' : rampSfriction,
        })

def convert_positions(positions):
    return wp.vec3(positions['x'], positions['y'], positions['z'])

def convert_rotation(rotation):
    return wp.quat(
        rotation['x'],
        rotation['y'],
        rotation['z'],
        rotation['w']
    )

simulation_specifictation = (
    load_stimuli_start("objects/stimuli.json", "objects/local_models")
)
sim_1 = next(simulation_specifictation)