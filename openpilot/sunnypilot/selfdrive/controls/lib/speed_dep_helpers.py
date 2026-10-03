"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""


FRICTION_REDUCTION_STEP = 0.005  # friction removed per Friction Reduction step
FRICTION_REDUCTION_MAX_STEPS = 20  # -0.100


def friction_reduction_amount(steps: int) -> float:
  """Friction Reduction setting: step N subtracts N*0.005 from learned friction (0 = off, 20 = -0.100).
  An absolute cut rather than a percentage, since the learned friction already falls with speed:
  one setting removes a larger share at highway speeds than in town."""
  return min(max(int(steps), 0), FRICTION_REDUCTION_MAX_STEPS) * FRICTION_REDUCTION_STEP


def reduce_friction(friction: float, steps: int) -> float:
  """Learned friction minus the Friction Reduction amount, floored at zero: a negative
  friction would flip the friction feedforward and push against the lateral error."""
  return max(friction - friction_reduction_amount(steps), 0.0)


def build_speed_dep_bp(speed_bp, factors, frictions, valid_bp, seed_lafs, seed_frictions,
                       global_laf, global_fric, friction_reduction=0):
  """Build the (speeds, latAccelFactor, friction) breakpoint tables the controller
  interpolates over each frame. Shared with the developer UI so the display always
  matches what the controller drives on.

  Learned (valid) bins carry the Friction Reduction cut; unlearned values never do.
  Unlearned bins fall back to, in order of preference:
  - the car's TOML seed curve when one is shipped (full-length table), else
  - the nearest learned bin: the table holds learned bins only, so np.interp clamps
    to the nearest learned value beyond the ends and bridges unlearned gaps, else
  - the global filtered values when nothing is learned yet.
  Every friction in the returned table is >= 0.
  """
  n = len(speed_bp)

  if seed_lafs and seed_frictions and len(seed_lafs) == n and len(seed_frictions) == n:
    laf_bp = [factors[i] if valid_bp[i] else seed_lafs[i] for i in range(n)]
    fric_bp = [reduce_friction(frictions[i], friction_reduction) if valid_bp[i] else max(seed_frictions[i], 0.0)
               for i in range(n)]
    return list(speed_bp), laf_bp, fric_bp

  if any(valid_bp):
    idxs = [i for i in range(n) if valid_bp[i]]
    return ([speed_bp[i] for i in idxs],
            [factors[i] for i in idxs],
            [reduce_friction(frictions[i], friction_reduction) for i in idxs])

  return list(speed_bp), [global_laf] * n, [max(global_fric, 0.0)] * n
