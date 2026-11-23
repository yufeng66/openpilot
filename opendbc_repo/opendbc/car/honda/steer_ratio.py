"""Variable-ratio rack map for the Honda Accord on the angle-loop EPS firmware.

Two coordinates, and it matters which one a value is in:

  linear    centre-equivalent steering angle. What VehicleModel produces from a
            curvature using a single steer ratio, and the coordinate a steer ratio
            and an angle offset are only meaningful in.
  physical  the angle the EPS publishes as STEERING_SENSORS.STEER_ANGLE, i.e. what
            CS.steeringAngleDeg carries, and the angle the angle-loop EPS servoes to.

A Honda without a rack map keeps the angle as VehicleModel gives it.
"""

import math

from opendbc.car.honda.values import HondaFlags


def _piecewise(x, xp, fp):
  for x0, x1, f0, f1 in zip(xp, xp[1:], fp, fp[1:], strict=False):
    if x <= x1:
      return f0 + (f1 - f0) * (x - x0) / (x1 - x0)
  return fp[-1]


# Honda Accord on the angle-loop EPS firmware (39990-TVA,A16x / A34x). Measured, not traced: the overall ratio
# (wheel angle / road-wheel angle) by published wheel angle over 82 routes. Not measurements: 0-23 deg (pinned by
# convention, not identified), 31-61 deg (interpolated across two bins that failed the lag test), 236+ deg (frozen;
# above 303 deg the data is parking speed). A compensation curve for calc_curvature's k = d/L, not the rack's true
# geometry. Bands: shape +/-0.12, level +/-0.57.
_ACCORD_ANGLE_LOOP_RATIO_BP = [0, 23, 31, 61, 76, 95, 116, 151, 178, 227, 236, 303, 380]
_ACCORD_ANGLE_LOOP_RATIO = [16.88, 16.88, 16.88, 16.25, 15.97, 15.45, 15.03, 14.68, 14.45, 14.09, 14.25, 12.98, 12.31]
# The on-centre ratio every angle drive ran; CP.steerRatio, so the learner's seed. The map is relative to the centre.
ACCORD_ANGLE_LOOP_STEER_RATIO = 16.84

# An overall ratio R, not a local gain: the road wheels are at published / R(published) = linear / R(0). One degree
# steps follow the curve between knots; past the last one the ratio is held, a straight line out to the sensor's 500 deg.
ACCORD_ANGLE_LOOP_VGR_ANGLE_BP = [float(a) for a in (*range(_ACCORD_ANGLE_LOOP_RATIO_BP[-1] + 1), 500)]
ACCORD_ANGLE_LOOP_VGR_LINEAR_BP = [a * _ACCORD_ANGLE_LOOP_RATIO[0] / _piecewise(a, _ACCORD_ANGLE_LOOP_RATIO_BP, _ACCORD_ANGLE_LOOP_RATIO)
                                   for a in ACCORD_ANGLE_LOOP_VGR_ANGLE_BP]

HONDA_VGR_INVERSE_BY_FLAG = {
  int(HondaFlags.EPS_ANGLE_LOOP_FW): (ACCORD_ANGLE_LOOP_VGR_LINEAR_BP, ACCORD_ANGLE_LOOP_VGR_ANGLE_BP),
}


def get_honda_vgr_inverse(flags):
  for flag, inverse in HONDA_VGR_INVERSE_BY_FLAG.items():
    if int(flags) & flag:
      return inverse
  return None


def vgr_linear_to_physical(linear_deg: float, inverse) -> float:
  """Centre-equivalent angle -> the angle the EPS will publish."""
  if inverse is None:
    return linear_deg
  linear_bp, angle_bp = inverse
  return math.copysign(float(_interp(abs(linear_deg), linear_bp, angle_bp)), linear_deg)


def vgr_physical_to_linear(physical_deg: float, inverse) -> float:
  """Published EPS angle -> centre-equivalent angle. Inverse of the above.

  Both breakpoint arrays are monotonically increasing, so the same interpolation
  run with the axes swapped is an exact inverse at the knots.
  """
  if inverse is None:
    return physical_deg
  linear_bp, angle_bp = inverse
  return math.copysign(float(_interp(abs(physical_deg), angle_bp, linear_bp)), physical_deg)


def _interp(x, xp, fp):
  # Local so this module stays importable without numpy on the car-port side.
  if x <= xp[0]:
    return fp[0]
  if x >= xp[-1]:
    return fp[-1]
  lo, hi = 0, len(xp) - 1
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if xp[mid] <= x:
      lo = mid
    else:
      hi = mid
  span = xp[hi] - xp[lo]
  if span <= 0:
    return fp[lo]
  return fp[lo] + (fp[hi] - fp[lo]) * (x - xp[lo]) / span
