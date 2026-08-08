"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc import NeuralNetworkLateralControl
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride
from openpilot.sunnypilot.selfdrive.controls.lib.speed_dep_helpers import build_speed_dep_bp


class LatControlTorqueExt(NeuralNetworkLateralControl, LatControlTorqueExtOverride):
  def __init__(self, lac_torque, CP, CP_SP, CI):
    NeuralNetworkLateralControl.__init__(self, lac_torque, CP, CP_SP, CI)
    LatControlTorqueExtOverride.__init__(self, CP)

  def update(self, CS, VM, pid, params, ff, pid_log, setpoint, measurement, calibrated_pose, roll_compensation,
             desired_lateral_accel, actual_lateral_accel, lateral_accel_deadzone, gravity_adjusted_lateral_accel,
             desired_curvature, actual_curvature, steer_limited_by_safety, output_torque):
    # Store vEgo for update_override_torque_params (which runs before this, next frame)
    self._last_vego = CS.vEgo
    self._ff = ff
    self._pid = pid
    self._pid_log = pid_log
    self._setpoint = setpoint
    self._measurement = measurement
    self._roll_compensation = roll_compensation
    self._lateral_accel_deadzone = lateral_accel_deadzone
    self._desired_lateral_accel = desired_lateral_accel
    self._actual_lateral_accel = actual_lateral_accel
    self._desired_curvature = desired_curvature
    self._actual_curvature = actual_curvature
    self._gravity_adjusted_lateral_accel = gravity_adjusted_lateral_accel
    self._steer_limited_by_safety = steer_limited_by_safety
    self._output_torque = output_torque

    self.update_calculations(CS, VM, desired_lateral_accel)
    self.update_neural_network_feedforward(CS, params, calibrated_pose)

    return self._pid_log, self._output_torque

  def update_speed_dep_torque(self, tp, friction_reduction: int = 0):
    """Apply speed-dependent learned values from torqued.
    Learned bins are used directly, with friction scaled by the Friction Reduction
    setting. Unlearned bins fall back to TOML seed values if available for this car,
    otherwise the nearest learned bin, otherwise global filtered (see
    build_speed_dep_bp)."""
    speed_bp = list(tp.speedBinCenters)
    if not speed_bp:
      self._speed_dep_active = False
      return

    if self._speed_dep_car_cfg is None:
      from opendbc.sunnypilot.car.interfaces import get_speed_dep_config
      self._speed_dep_car_cfg = get_speed_dep_config().get(self.CP.carFingerprint, {})
    cfg = self._speed_dep_car_cfg

    bp_speeds, laf_bp, fric_bp = build_speed_dep_bp(
      speed_bp, list(tp.speedBinLatAccelFactors), list(tp.speedBinFrictions), list(tp.speedBinValid),
      cfg.get('laf_bp'), cfg.get('friction_bp'),
      tp.latAccelFactorFiltered, tp.frictionCoefficientFiltered, friction_reduction)

    self._speed_dep_active = True
    self._speed_dep_speed_bp = bp_speeds
    self._speed_dep_lat_accel_factor_bp = laf_bp
    self._speed_dep_friction_bp = fric_bp

    # Set representative values at 20 m/s for PID limits (actual per-frame
    # interpolation happens in update_override_torque_params before each frame)
    self.lac_torque.torque_params.latAccelFactor = float(np.interp(20.0, bp_speeds, laf_bp))
    self.lac_torque.torque_params.latAccelOffset = tp.latAccelOffsetFiltered
    self.lac_torque.torque_params.friction = float(np.interp(20.0, bp_speeds, fric_bp))
    self.lac_torque.update_limits()
