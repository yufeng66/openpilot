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


def friction_scale(reduction: int) -> float:
  """Friction Reduction setting: step N lowers learned friction by N*10% (0 = off, 9 = -90%)."""
  return 1.0 - min(max(reduction, 0), 9) / 10.0


def build_speed_dep_bp(speed_bp, factors, frictions, valid_bp, seed_lafs, seed_frictions,
                       global_laf, global_fric, friction_reduction=0):
  """Build the (speeds, latAccelFactor, friction) breakpoint tables the controller
  interpolates over each frame.

  Learned (valid) bins carry the Friction Reduction scale; unlearned values never do.
  Unlearned bins fall back to, in order of preference:
  - the car's configured seed curve when one is shipped (full-length table), else
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


def interp_live_torque_params(tp, v_ego, car_cfg=None):
  """(latAccelFactor, latAccelOffset, friction) to apply at v_ego.

  With the learner off (no speedBinCenters in the message) this returns the
  stock global filtered values, so the caller's behavior is unchanged.
  latAccelOffset is not speed-binned and always passes through."""
  speed_bp = list(tp.speedBinCenters)
  if not speed_bp:
    return tp.latAccelFactorFiltered, tp.latAccelOffsetFiltered, tp.frictionCoefficientFiltered

  cfg = car_cfg or {}
  bp_speeds, laf_bp, fric_bp = build_speed_dep_bp(
    speed_bp, list(tp.speedBinLatAccelFactors), list(tp.speedBinFrictions), list(tp.speedBinValid),
    cfg.get('laf_bp'), cfg.get('friction_bp'),
    tp.latAccelFactorFiltered, tp.frictionCoefficientFiltered)

  return (float(np.interp(v_ego, bp_speeds, laf_bp)),
          tp.latAccelOffsetFiltered,
          float(np.interp(v_ego, bp_speeds, fric_bp)))
