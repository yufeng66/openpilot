"""End-to-end checks of the jerk-aware extension inside the real LatControlTorque.

The unit tests exercise the extension in isolation; these run the actual
controller so the wiring (argument order, override point, shared PID) is covered.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch  # noqa: TID251

import numpy as np
import pytest

from cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.modeld.constants import ModelConstants
from dragonpilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware import LatControlTorqueJerkAware

PATCH_PARAMS = 'dragonpilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware.Params'

LAT_ACCEL_FACTOR = 2.5
FRICTION = 0.12


def make_CP():
  CP = car.CarParams.new_message()
  CP.steerLimitTimer = 0.4
  CP.steerActuatorDelay = 0.12
  CP.steerRatio = 15.0
  tune = CP.lateralTuning.init('torque')
  tune.latAccelFactor = LAT_ACCEL_FACTOR
  tune.latAccelOffset = 0.03
  tune.friction = FRICTION
  tune.steeringAngleDeadzoneDeg = 0.0
  return CP.as_reader()   # the controller calls lateralTuning.torque.as_builder()


def make_CI():
  CI = MagicMock()
  CI.torque_from_lateral_accel.return_value = lambda la, tp: la / tp.latAccelFactor
  CI.lateral_accel_from_torque.return_value = lambda t, tp: t * tp.latAccelFactor
  return CI


def make_lac(jerk_enabled):
  with patch(PATCH_PARAMS) as mock_params:
    mock_params.return_value.get_bool.side_effect = \
      lambda p: jerk_enabled if p == "dp_lat_jerk_torque" else False
    return LatControlTorque(make_CP(), make_CI(), DT_CTRL)


def make_CS(v_ego=25.0):
  return SimpleNamespace(vEgo=v_ego, aEgo=0.0, steeringAngleDeg=2.0, steeringRateDeg=5.0,
                         steeringPressed=False)


# a plausible small-angle bicycle model: curvature = steer_angle / (steer_ratio * wheelbase).
# Matters because the same call also produces the deadzone and the measured jerk, so a
# constant would put the controller in permanent saturation and hide the change under test.
STEER_RATIO, WHEELBASE = 15.0, 2.7


def make_VM():
  vm = MagicMock()
  vm.calc_curvature.side_effect = lambda angle, v, roll: angle / (STEER_RATIO * WHEELBASE)
  return vm


def make_params():
  return SimpleNamespace(roll=0.0, angleOffsetDeg=0.0)


def make_model(slope=0.2):
  """A plan that keeps building lateral accel in one direction, i.e. deliberate jerk."""
  n = len(ModelConstants.T_IDXS)
  return SimpleNamespace(acceleration=SimpleNamespace(y=list(np.arange(n) * slope)),
                         orientation=SimpleNamespace(x=[0.0] * 33))


def drive(lac, frames=20, v_ego=25.0, desired_curvature=0.001):
  """Run the controller for a few frames; returns the last (torque, pid_log)."""
  CS, VM, params = make_CS(v_ego), make_VM(), make_params()
  out = None
  for _ in range(frames):
    out = lac.update(True, CS, VM, params, False, desired_curvature, False, 0.15)
  return out


class TestExtensionIsAttached:
  def test_controller_owns_a_jerk_aware_extension(self):
    assert isinstance(make_lac(False).extension, LatControlTorqueJerkAware)

  def test_toggle_state_is_read_from_params(self):
    assert not make_lac(False).extension.enabled
    assert make_lac(True).extension.enabled


class TestDisabledMatchesStock:
  def test_output_is_the_stock_lat_accel_space_formula(self):
    lac = make_lac(False)
    lac.extension.update_model_v2(make_model())
    torque, _, pid_log = drive(lac)
    # stock: PID runs in lat-accel space, converted once at the end
    expected = -(lac.pid.control / lac.torque_params.latAccelFactor)
    assert torque == pytest.approx(expected)
    assert pid_log.active

  def test_pid_is_updated_exactly_once_per_frame(self):
    lac = make_lac(False)
    lac.extension.update_model_v2(make_model())
    lac.pid.update = MagicMock(wraps=lac.pid.update)
    drive(lac, frames=1)
    assert lac.pid.update.call_count == 1

  def test_model_and_lag_updates_are_harmless(self):
    """controlsd calls these every frame regardless of the toggle."""
    lac = make_lac(False)
    lac.extension.update_model_v2(make_model())
    lac.extension.update_lateral_lag(0.2)
    torque, _, _ = drive(lac)
    assert np.isfinite(torque)


class TestEnabledChangesTheLoop:
  def test_output_is_the_torque_space_pid_output(self):
    lac = make_lac(True)
    lac.extension.update_model_v2(make_model())
    torque, _, _ = drive(lac)
    assert torque == pytest.approx(-lac.pid.control)

  def test_output_differs_from_stock(self):
    off, on = make_lac(False), make_lac(True)
    for lac in (off, on):
      lac.extension.update_model_v2(make_model())
    t_off, _, _ = drive(off)
    t_on, _, _ = drive(on)
    assert not np.isclose(t_off, t_on)

  def test_pid_error_is_logged_in_torque_space(self):
    lac = make_lac(True)
    lac.extension.update_model_v2(make_model())
    _, _, pid_log = drive(lac)
    # torque-space error is the lat-accel error divided by latAccelFactor, so it
    # is strictly smaller in magnitude for any latAccelFactor > 1
    assert abs(pid_log.error) < abs(pid_log.desiredLateralAccel - pid_log.actualLateralAccel)

  def test_pid_is_updated_twice_per_frame(self):
    """Documents the sunnypilot parity quirk reproduced on purpose: the host runs
    its lat-accel-space PID, then the extension runs the same PID again in torque
    space. If this ever becomes one call, the tune changes and this test should be
    updated deliberately, not silently."""
    lac = make_lac(True)
    lac.extension.update_model_v2(make_model())
    lac.pid.update = MagicMock(wraps=lac.pid.update)
    drive(lac, frames=1)
    assert lac.pid.update.call_count == 2

  def test_output_stays_within_torque_limits_without_any_caller_help(self):
    """update() re-establishes its own bounds, so a hard demand cannot push past
    steer_max even though the host resets the limits to lat-accel space first."""
    lac = make_lac(True)
    lac.extension.update_model_v2(make_model(slope=3.0))
    for _ in range(50):
      lac.update_live_torque_params(LAT_ACCEL_FACTOR, 0.03, FRICTION)   # resets limits
      torque, _, _ = lac.update(True, make_CS(), make_VM(), make_params(), False, 0.05, False, 0.15)
    assert abs(torque) <= lac.steer_max + 1e-9
    assert (lac.pid.pos_limit, lac.pid.neg_limit) == (lac.steer_max, -lac.steer_max)

  def test_disabled_leaves_the_host_lat_accel_limits_alone(self):
    lac = make_lac(False)
    lac.extension.update_model_v2(make_model())
    lac.update_live_torque_params(LAT_ACCEL_FACTOR, 0.03, FRICTION)
    drive(lac, frames=1)
    assert lac.pid.pos_limit == pytest.approx(lac.steer_max * LAT_ACCEL_FACTOR)

  def test_inactive_still_returns_zero_torque(self):
    lac = make_lac(True)
    lac.extension.update_model_v2(make_model())
    torque, _, pid_log = lac.update(False, make_CS(), make_VM(), make_params(), False, 0.001, False, 0.15)
    assert torque == 0.0
    assert not pid_log.active

  def test_survives_a_missing_model(self):
    """modelV2 can be absent or short before the model is up; the controller must
    still produce finite torque."""
    lac = make_lac(True)
    lac.extension.update_model_v2(None)
    torque, _, _ = drive(lac)
    assert np.isfinite(torque)
