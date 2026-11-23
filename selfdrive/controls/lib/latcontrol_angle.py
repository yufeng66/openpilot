import math

from cereal import log
from opendbc.car.honda.steer_ratio import get_honda_vgr_inverse, vgr_linear_to_physical
from openpilot.selfdrive.controls.lib.latcontrol import LatControl

# TODO This is speed dependent
STEER_ANGLE_SATURATION_THRESHOLD = 2.5  # Degrees


class LatControlAngle(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.sat_check_min_speed = 5.
    # honda (angle-loop EPS): the EPS angle loop lags 0.3-0.6 s, so the raw angle-error rule would flag ordinary turn entries
    self.use_steer_limited_by_safety = CP.brand in ("tesla", "hyundai", "honda")
    # CP.flags bits are per brand; None (no variable-ratio rack map) leaves the angle as VehicleModel gives it
    self.vgr_inverse = get_honda_vgr_inverse(CP.flags) if CP.brand == "honda" else None

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    angle_log = log.ControlsState.LateralAngleState.new_message()

    if not active:
      angle_log.active = False
      angle_steers_des = float(CS.steeringAngleDeg)
    else:
      angle_log.active = True
      angle_steers_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
      # VehicleModel's angle is centre-equivalent (one steer ratio); the EPS takes the angle it publishes
      angle_steers_des = vgr_linear_to_physical(angle_steers_des, self.vgr_inverse)
      angle_steers_des += params.angleOffsetDeg

    if self.use_steer_limited_by_safety:
      # these cars' carcontrollers calculate max lateral accel and jerk, so we can rely on carOutput for saturation
      angle_control_saturated = steer_limited_by_safety
    else:
      # for cars which use a method of limiting torque such as a torque signal (Nissan and Toyota)
      # or relying on EPS (Ford Q3), carOutput does not capture maxing out torque  # TODO: this can be improved
      angle_control_saturated = abs(angle_steers_des - CS.steeringAngleDeg) > STEER_ANGLE_SATURATION_THRESHOLD
    angle_log.saturated = bool(self._check_saturation(angle_control_saturated, CS, False, curvature_limited))
    angle_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    angle_log.steeringAngleDesiredDeg = angle_steers_des
    return 0, float(angle_steers_des), angle_log
