# Summary of what is going on

## High Level Alorithm
- Generate (n) good proposals without physics
- Run physics starting from (n) good proposals

### Defining a Good Score
- Calculate the score between scene and itself and then normalize
    - Ceiling is the depth-map and itself
    - Floor is likelihood and noise
    - Normalize from 0 --> 1, and set threshold to (0.8)

### Define Objects (Across n objects)
- 2 or 4 visable objects
    - 2 hidden objects

- 2 Samplers
    (1) 4 objects
    (2) 6 objects

### Define a Placement Order
1. Place object 1 (visable) - 6DoF
2. Place object 2 (visable)
3. Place object 3 (in-visible)
4. Place object 4 (in-visible)

### Scheduler for proposal variance
1. As we get closer to correct proposal finish

### Placement
1. Place within 10 cm of correct placement for **visable**
1. Uniformly sample from occluded points for initilization for **object**
1. 1m x 1m bounding box around possible placements

## Testing Dataset

### Scene01
1. Bowl on a cone

### Scene02
1. Bowl on a cone
1. Bowl on a stick

### Scene 03
1. 

## Likelihood Function

### Chamfer Distance
- Need new likelihoof function off of distance

## Steps Moving Forward

### Behavoir Model - Pilot and Proof of Concept
- Specific need to build in reasoning ability
- Sanity Check (Our Model vs. Humans)

### Ultimate Goal
- Paper of NN more aligned with humans
