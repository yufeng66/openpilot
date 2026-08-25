"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

Licensed under the MIT License. Ported to dragonpilot from sunnypilot's
latcontrol_torque_jerk_aware.py, behind the dp_lat_jerk_torque toggle.

Original design by @twilsonco.

What it changes when armed: the stock controller closes the loop in lateral
acceleration and converts to torque once at the end. This runs the error and the
feedforward in torque space instead, and replaces the friction input with a
model-based "deliberate" lateral jerk -- the planned lateral jerk over the next
~1.4-2.0 s, zeroed whenever the plan changes sign in that window, so short-lived
jerk is ignored and only sustained cornering demand raises the friction term.

Two behaviours are inherited deliberately from sunnypilot rather than corrected,
because the point of this port is to reproduce a tune that has been validated on
the road. Both are documented at their call sites:
  * the host controller's PID is updated twice per control frame (once by the
    host in lat-accel space, once here in torque space), so the integrator takes
    two increments per frame; and
  * the latAccelOffset roll-bias correction the host subtracts from its
    feedforward is not carried into the torque-space feedforward.
"""
from opendbc.car.lateral import FRICTION_THRESHOLD
from openpilot.common.params import Params

from dragonpilot.selfdrive.controls.lib.latcontrol_torque_ext_base import (
  LatControlInputs, LatControlTorqueExtBase, get_friction_in_torque_space,
)


class LatControlTorqueJerkAware(LatControlTorqueExtBase):
  def __init__(self, lac_torque, CP, CI):
    super().__init__(lac_torque, CP, CI)
    self.params = Params()
    # Read once. controlsd only runs onroad, so this is a fresh read at the
    # start of every drive and the toggle is deliberately offroad-only.
    self._jerk_aware_enabled = self.params.get_bool("dp_lat_jerk_torque")

  @property
  def enabled(self):
    return self._jerk_aware_enabled

  def update_limits(self):
    """Retarget the host's PID limits to torque space. The host sets them in
    lat-accel space (steer_max * latAccelFactor); once the loop runs on torque the
    bound is steer_max itself, and it also governs integrator anti-windup.

    The PID is shared, so this tighter bound must govern BOTH of the frame's
    writes, not just the extension's own. LatControlTorque.update() therefore
    calls this at its top, after controlsd's per-frame update_live_torque_params()
    has reset the limits to lat-accel space and before the host's pid.update();
    update() below calls it again, which re-pins on the first frame (when _pid is
    not bound yet at the top-of-frame call). Setting the bound only around the
    extension's write anchors anti-windup latAccelFactor x above steer_max and
    the integrator winds until the output pins at steer_max (measured: i 1.05 at
    bound 1.0 under a steady tracking error)."""
    if not self._jerk_aware_enabled or self._pid is None:
      return
    self._pid.set_limits(self.lac_torque.steer_max, -self.lac_torque.steer_max)

  def update(self, CS, VM, pid, pid_log, roll_compensation, desired_lateral_accel, actual_lateral_accel,
             lateral_accel_deadzone, gravity_adjusted_lateral_accel, setpoint, measurement,
             steer_limited_by_safety, output_torque):
    """Returns (pid_log, output_torque). A no-op returning its inputs unchanged
    when the toggle is off, so the host controller keeps stock behaviour."""
    if not self._jerk_aware_enabled:
      return pid_log, output_torque

    self._pid = pid
    self.update_limits()
    self._pid_log = pid_log
    self._setpoint = setpoint
    self._measurement = measurement
    self._roll_compensation = roll_compensation
    self._lateral_accel_deadzone = lateral_accel_deadzone
    self._desired_lateral_accel = desired_lateral_accel
    self._actual_lateral_accel = actual_lateral_accel
    self._gravity_adjusted_lateral_accel = gravity_adjusted_lateral_accel
    self._steer_limited_by_safety = steer_limited_by_safety
    self._output_torque = output_torque

    self.update_calculations(CS, VM, desired_lateral_accel)
    self.update_jerk_aware_torque_control(CS, roll_compensation, gravity_adjusted_lateral_accel)

    return self._pid_log, self._output_torque

  def update_jerk_aware_torque_control(self, CS, roll_compensation, gravity_adjusted_lateral_accel):
    if not self._jerk_aware_enabled:
      return

    torque_from_setpoint = self.torque_from_lateral_accel_in_torque_space(
      LatControlInputs(self._setpoint, roll_compensation, CS.vEgo, CS.aEgo), self.torque_params, gravity_adjusted=False
    )
    torque_from_measurement = self.torque_from_lateral_accel_in_torque_space(
      LatControlInputs(self._measurement, roll_compensation, CS.vEgo, CS.aEgo), self.torque_params, gravity_adjusted=False
    )

    self._pid_log.error = float(torque_from_setpoint - torque_from_measurement)
    # NOTE (sunnypilot parity): the host subtracts torque_params.latAccelOffset from
    # its own feedforward to correct device-vs-car roll misalignment. That term is
    # not reapplied here, so arming this toggle also drops the offset correction.
    self._ff = self.torque_from_lateral_accel_in_torque_space(
      LatControlInputs(gravity_adjusted_lateral_accel, roll_compensation, CS.vEgo, CS.aEgo), self.torque_params, gravity_adjusted=True
    )

    friction_input = self.update_friction_input(self._desired_lateral_accel, self._actual_lateral_accel)
    self._ff += get_friction_in_torque_space(friction_input, self._lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

    # NOTE (sunnypilot parity): this is the second self._pid.update() of the frame --
    # the host already ran one in lat-accel space whose output is about to be
    # discarded. The integrator therefore takes two increments per frame, from
    # errors in two different spaces. Reproduced on purpose: it is part of the
    # tune this port is matching.
    self.update_output_torque(CS)
