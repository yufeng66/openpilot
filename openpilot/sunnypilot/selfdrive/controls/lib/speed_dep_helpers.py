"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""


def friction_scale(reduction: int) -> float:
  """Friction Reduction setting: step N lowers learned friction by N*10% (0 = off, 9 = -90%)."""
  return 1.0 - min(max(reduction, 0), 9) / 10.0


def build_speed_dep_bp(speed_bp, factors, frictions, valid_bp, seed_lafs, seed_frictions,
                       global_laf, global_fric, friction_reduction=0):
  """Build the (speeds, latAccelFactor, friction) breakpoint tables the controller
  interpolates over each frame. Shared with the developer UI so the display always
  matches what the controller drives on.

  Learned (valid) bins carry the Friction Reduction scale; unlearned values never do.
  Unlearned bins fall back to, in order of preference:
  - the car's TOML seed curve when one is shipped (full-length table), else
  - the nearest learned bin: the table holds learned bins only, so np.interp clamps
    to the nearest learned value beyond the ends and bridges unlearned gaps, else
  - the global filtered values when nothing is learned yet.
  """
  scale = friction_scale(friction_reduction)
  n = len(speed_bp)

  if seed_lafs and seed_frictions and len(seed_lafs) == n and len(seed_frictions) == n:
    laf_bp = [factors[i] if valid_bp[i] else seed_lafs[i] for i in range(n)]
    fric_bp = [frictions[i] * scale if valid_bp[i] else seed_frictions[i] for i in range(n)]
    return list(speed_bp), laf_bp, fric_bp

  if any(valid_bp):
    idxs = [i for i in range(n) if valid_bp[i]]
    return ([speed_bp[i] for i in idxs],
            [factors[i] for i in idxs],
            [frictions[i] * scale for i in idxs])

  return list(speed_bp), [global_laf] * n, [global_fric] * n
