import warp as wp

vec6f = wp.types.vector(length=6, dtype=float)

@wp.kernel
def apply_random_force(body_f: wp.array(dtype=vec6f), strength: float, seed: int):
    i = wp.tid()

    rng = wp.rand_init(seed, i)

    fx = (wp.randf(rng) - 0.5) * 2.0 * strength
    fy = (wp.randf(rng) - 0.5) * 2.0 * strength
    fz = (wp.randf(rng) - 0.5) * 2.0 * strength
    f = body_f[i]

    body_f[i] = vec6f(f[0] + fx, f[1] + fy, f[2] + fz, f[3], f[4], f[5])

@wp.kernel
def apply_random_force_rot(body_f: wp.array(dtype=vec6f), strength: float, seed: int):
    i = wp.tid()

    rng = wp.rand_init(seed, i)

    fx = (wp.randf(rng) - 0.5) * 2.0 * strength
    fy = (wp.randf(rng) - 0.5) * 2.0 * strength
    fz = (wp.randf(rng) - 0.5) * 2.0 * strength

    fxr = (wp.randf(rng) - 0.5) * 2.0 * strength
    fxy = (wp.randf(rng) - 0.5) * 2.0 * strength
    fzr = (wp.randf(rng) - 0.5) * 2.0 * strength

    f = body_f[i]

    body_f[i] = vec6f(f[0] + fx, f[1] + fy, f[2] + fz, f[3] + fxr, f[4] + fxy, f[5] + fzr)
