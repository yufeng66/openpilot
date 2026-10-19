"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

Licensed under the MIT License. Ported to dragonpilot from the sunnypilot-based
speed-dependent torque learner (yufeng66/openpilot, pal23sp-hda2-testing).

Controller-side helpers for the speed-dependent torque learner: build the
(speed, latAccelFactor, friction) breakpoint tables from a liveTorqueParameters
message and interpolate them at the current speed. controlsd calls
interp_live_torque_params() every control frame, so the applied torque params
track vEgo continuously between the learner's 4 Hz publishes.
"""
import numpy as np

# Per-car seed curves, keyed by carFingerprint. Cars without an entry seed every
# bin with their global offline latAccelFactor/friction (and unlearned bins fall
# back to the nearest learned bin instead of a seed curve). Shape per entry:
#   {'speed_bp': [m/s...], 'laf_bp': [...], 'friction_bp': [...]}  (equal lengths)
SPEED_DEP_CAR_CONFIG: dict[str, dict] = {}

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
  interpolates over each frame.

  Learned (valid) bins carry the Friction Reduction cut; unlearned values never do.
  Unlearned bins fall back to, in order of preference:
  - the car's configured seed curve when one is shipped (full-length table), else
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


def interp_live_torque_params(tp, v_ego, car_cfg=None, friction_reduction=0):
  """(latAccelFactor, latAccelOffset, friction) to apply at v_ego.

  With the learner off (no speedBinCenters in the message) this returns the
  stock global filtered values, so the caller's behavior is unchanged --
  friction_reduction cuts learned bins only and is therefore inert here too.
  latAccelOffset is not speed-binned and always passes through."""
  speed_bp = list(tp.speedBinCenters)
  if not speed_bp:
    return tp.latAccelFactorFiltered, tp.latAccelOffsetFiltered, tp.frictionCoefficientFiltered

  cfg = car_cfg or {}
  bp_speeds, laf_bp, fric_bp = build_speed_dep_bp(
    speed_bp, list(tp.speedBinLatAccelFactors), list(tp.speedBinFrictions), list(tp.speedBinValid),
    cfg.get('laf_bp'), cfg.get('friction_bp'),
    tp.latAccelFactorFiltered, tp.frictionCoefficientFiltered, friction_reduction)

  return (float(np.interp(v_ego, bp_speeds, laf_bp)),
          tp.latAccelOffsetFiltered,
          # floored again on the per-frame value the controller reads, so whatever
          # lands in the table can never hand it a negative friction
          max(float(np.interp(v_ego, bp_speeds, fric_bp)), 0.0))
