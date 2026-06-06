# Composite Rank-Based Likelihood — Design & Speedup Analysis

**Status:** proposal / discussion notes
**Context:** replacing the multi-snapshot point-cloud likelihood in
`likelihood/likelihoods.py` (`ParallelPhysicsLikelihood`) with a composite score.

## The proposal

Replace "render the physics rollout at several snapshots and average the
point-cloud scores" with a weighted combination of three terms, each converted
to a **rank** across the particle population before combining:

```
score = λ₁ · rank(point_cloud_match)      # existing 3DP3 still-render vs GT
      + λ₂ · rank(initial_penetration)     # collide at proposed pose, sum depths
      + λ₃ · rank(total_movement)          # ‖body_q_final − body_q_init‖ over a rollout
```

Then sample reconstruction quality (pose error vs `truth.json`) across thousands
of generated scenes to tune `λ₁, λ₂, λ₃`.

## Assessment — the direction is good

Trading repeated rendering for cheap physics-derived scalars is the right
instinct. Both new terms are far cheaper than a render and capture failure
modes the single still-render cannot:

- **Initial penetration** — `model.collide(state)` is already called every
  substep (`likelihoods.py:129`). One collide at the proposed pose, sum the
  contact penetration depths → scalar per world, essentially free. Catches
  *jammed* placements (object driven into the occluder/table). **Misses
  floaters** (an object hovering above the table has zero penetration).
- **Total movement** — read straight off the state buffer, no camera. Catches
  *unstable* placements (floats then falls, topples, slides). Catches the
  floater that penetration misses, but needs the sim to run.

They are complementary, not redundant — keep both.

### Rank-normalization is the strongest part

This is the real reason to do it. The point-cloud score is a log-likelihood
with huge dynamic range (scores like −1000), penetration and movement are in
meters. Ranks put them on a common footing **and** make the λ's transfer across
scenes, since raw point-cloud magnitude scales with scene complexity / point
count / occlusion. That cross-scene invariance is what lets you tune the λ's
once over thousands of scenes and reuse them. Without ranks, λ's fit on raw
scores would not transfer and per-scene tuning would be required.

## Speedup — conditional, and smaller than it looks

Cost decomposition. Let R = one all-worlds render+score, P = one physics step
(collide+solve), k = `frames // eval_every` renders, N = `frames * substeps`
steps.

| mode | cost |
|------|------|
| `still()` | R |
| `physics()` | N·P + k·R |
| composite (full rollout) | N·P + R + (free collide) |
| composite (truncated rollout, N′ steps) | N′·P + R |

With current defaults `frames=50, eval_every=20` → **k = 2**. The composite
score *as written* keeps the full 50-frame rollout and only drops **one** of the
two snapshot renders:

```
speedup vs physics()  =  (N·P + 2R) / (N·P + R)
```

- ≈ 1.0  if solver steps dominate (200 steps + 200 collides is a lot of XPBD).
- ≈ 1.3× if the render dominates.

So "drop one render" measures the wrong thing. The movement term still needs the
**full rollout** to run; it just doesn't need the camera. The naive version's
speedup is gated by the R/P ratio and at worst ≈ 0.

### Where the real speedup lives: the rollout length

The new terms make the rollout shortenable, which the old multi-snapshot scheme
could not:

1. **Truncate the rollout.** Instability (topple/slide/fall) shows up in the
   first handful of frames, not at frame 50. If movement is discriminative at
   ~5–10 frames, N drops from 200 steps to 20–40 — a *multiplicative* cut on the
   dominant cost, not a one-render trim.
2. **Penetration as a near-free pre-filter.** Collide-only screening rejects
   obviously-jammed worlds with zero rollout and zero render, so the expensive
   movement rollout only runs on survivors.

**Bottom line:** keep the 50-frame rollout → small win. Truncate it (which
movement + penetration make safe) → that's where a several-× speedup lives. The
load-bearing unknown is the R/P ratio, which needs the H100 to measure (do not
profile on the local RTX 3070).

## Concerns to resolve before building

1. **What does the score feed?** Ranks are great for selecting the argmax
   reconstruction and for λ-tuning, but they are lossy as **importance /
   resampling weights** — `proposer.propose` resamples proportional to weight,
   and ranks flatten "10× better" to "one rank better," dulling the posterior
   peak and slowing convergence. If this score drives resampling, map ranks back
   to a smooth weight (e.g. exponentiate a fractional-rank quantile). If it is
   only for final selection + tuning, raw ranks are fine. **← open question.**

2. **Whose movement?** The stronger physics cue is not "did the hidden target
   move" but "did the *observed / stacked* object move away from the pose you
   actually observed." The GT observed poses are known; a correct hidden-block
   pose keeps the resting object where it was seen, a wrong one lets it
   slide/drop. Weight (or add) movement of the observed objects against their
   known poses — that is the point of the resting-contact probe.

3. **Non-determinism.** The XPBD GPU solver is non-deterministic, so the
   movement term is noisy run-to-run and near-ties in rank will flip. Ranks are
   robust to small perturbations and noise averages out over thousands of scenes
   for tuning, but it adds variance to any single resampling step. Measure
   run-to-run movement variance to size the particle count.

## Tuning notes

- 3 parameters with GT pose error as the target is a tiny learning-to-rank
  problem — coarse grid on the simplex, or Nelder-Mead / CMA-ES, with a
  **scene-level train/test split**.
- Expect penetration and movement to be **collinear** (jammed starts also tend
  to move a lot once simulated): the combination will be well-determined but the
  individual λ's weakly identified. Do not over-read individual λ values.

## Validation gate (do this first)

On a sample of scenes, check that the composite-score ranking of particles
matches the full multi-render `physics()` ranking. The whole bet is that
"one render + low movement" is a faithful surrogate for "render stays matched
throughout settling." Confirm the surrogate before committing to the speedup.

## Arithmetic to confirm

As read in `likelihoods.py`, with `frames=50, eval_every=20` the snapshots land
at frame **20 and 40** (`(frame+1) % eval_every == 0`), and the loop runs all
50 frames regardless (`_num_eval_points = frames // eval_every = 2`). If the
intent is to use frame **0 and 20** instead, note: (a) frame 0 is the
pre-physics still render — physics-free — and (b) frames 21–50 are currently
simulated for nothing. Pin down the real `frames` / `eval_every` and whether the
pre-physics still is one of the two renders, since it changes the cost numbers.
