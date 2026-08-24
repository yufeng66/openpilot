"""Tests for the Lateral Jerk Torque Controller (dp_lat_jerk_torque).

Covers the two things that matter for a controller port: it is provably inert
with the toggle off, and with it on the torque-space loop and the model-based
"deliberate jerk" friction input behave the way the sunnypilot original does.
"""
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch  # noqa: TID251

import numpy as np
import pytest

from opendbc.car.lateral import FRICTION_THRESHOLD
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants
from dragonpilot.selfdrive.controls.lib.latcontrol_torque_ext_base import (
  LAT_PLAN_MIN_IDX, LatControlInputs, get_friction_in_torque_space, get_lookahead_value,
  get_predicted_lateral_jerk, sign, torque_from_lateral_accel_linear_in_torque_space,
)
from dragonpilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware import LatControlTorqueJerkAware

PATCH_PARAMS = 'dragonpilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware.Params'


def make_torque_params(lat_accel_factor=2.5, friction=0.12):
  return SimpleNamespace(latAccelFactor=lat_accel_factor, friction=friction, latAccelOffset=0.03)


def make_lac(steer_max=1.0, **tp):
  """Stand-in for LatControlTorque: the extension only reads torque_params and steer_max."""
  return SimpleNamespace(torque_params=make_torque_params(**tp), steer_max=steer_max)


def make_CP(steer_actuator_delay=0.12):
  return SimpleNamespace(steerActuatorDelay=steer_actuator_delay)


def make_CS(v_ego=25.0, a_ego=0.0, steering_rate_deg=0.0, steering_pressed=False):
  return SimpleNamespace(vEgo=v_ego, aEgo=a_ego, steeringRateDeg=steering_rate_deg,
                         steeringPressed=steering_pressed)


def make_VM(curvature=0.001):
  vm = MagicMock()
  vm.calc_curvature.return_value = curvature
  return vm


def make_model(lat_accels=None, n_orientation=CONTROL_N):
  """modelV2 stand-in. acceleration.y must be len(T_IDXS) for the jerk finite difference."""
  if lat_accels is None:
    lat_accels = [0.0] * len(ModelConstants.T_IDXS)
  return SimpleNamespace(acceleration=SimpleNamespace(y=list(lat_accels)),
                         orientation=SimpleNamespace(x=[0.0] * n_orientation))


def make_ext(enabled, lac=None, CP=None):
  with patch(PATCH_PARAMS) as mock_params:
    mock_params.return_value.get_bool.side_effect = \
      lambda p: enabled if p == "dp_lat_jerk_torque" else False
    return LatControlTorqueJerkAware(lac or make_lac(), CP or make_CP(), MagicMock())


def make_pid():
  return PIDController(0.8, 0.15, rate=100)


UPDATE_ARGS = dict(roll_compensation=0.0, desired_lateral_accel=1.0, actual_lateral_accel=0.8,
                   lateral_accel_deadzone=0.0, gravity_adjusted_lateral_accel=1.0,
                   setpoint=1.0, measurement=0.8, steer_limited_by_safety=False)


def run_update(ext, pid, pid_log, output_torque=0.42, CS=None, VM=None, **overrides):
  args = dict(UPDATE_ARGS)
  args.update(overrides)
  return ext.update(CS or make_CS(), VM or make_VM(), pid, pid_log,
                    output_torque=output_torque, **args)


class TestTorqueSpaceHelpers:
  def test_linear_conversion_divides_by_lat_accel_factor(self):
    tp = make_torque_params(lat_accel_factor=2.5)
    out = torque_from_lateral_accel_linear_in_torque_space(
      LatControlInputs(2.0, 0.0, 25.0, 0.0), tp, gravity_adjusted=False)
    assert out == pytest.approx(0.8)

  def test_friction_is_not_scaled_by_lat_accel_factor(self):
    """The torque-space twin returns raw friction; opendbc's lat-accel-space
    get_friction multiplies by latAccelFactor for its own caller."""
    tp = make_torque_params(friction=0.12, lat_accel_factor=2.5)
    assert get_friction_in_torque_space(10.0, 0.0, FRICTION_THRESHOLD, tp) == pytest.approx(0.12)
    assert get_friction_in_torque_space(-10.0, 0.0, FRICTION_THRESHOLD, tp) == pytest.approx(-0.12)
    assert get_friction_in_torque_space(0.0, 0.0, FRICTION_THRESHOLD, tp) == pytest.approx(0.0)

  def test_friction_deadzone_zeroes_small_errors(self):
    tp = make_torque_params(friction=0.12)
    assert get_friction_in_torque_space(0.05, 0.1, FRICTION_THRESHOLD, tp) == pytest.approx(0.0)
    assert get_friction_in_torque_space(0.15, 0.1, FRICTION_THRESHOLD, tp) != pytest.approx(0.0)

  def test_sign_and_lookahead_semantics(self):
    assert (sign(2.0), sign(-2.0), sign(0.0)) == (1.0, -1.0, 0.0)
    # a sign flip anywhere in the lookahead window means "not deliberate"
    assert get_lookahead_value([1.0, -0.5, 2.0], 1.0) == 0.0
    # otherwise the least-committal value in the window wins
    assert get_lookahead_value([2.0, 3.0], 1.5) == pytest.approx(1.5)
    assert get_lookahead_value([2.0, 0.7], 1.5) == pytest.approx(0.7)
    assert get_lookahead_value([], 1.5) == pytest.approx(1.5)

  def test_predicted_jerk_is_a_finite_difference(self):
    t_diffs = np.diff(ModelConstants.T_IDXS)
    accels = list(np.arange(len(ModelConstants.T_IDXS), dtype=float))
    jerk = get_predicted_lateral_jerk(accels, t_diffs)
    assert len(jerk) == len(accels) - 1
    assert jerk == pytest.approx((1.0 / t_diffs).tolist())


class TestDisabled:
  """With the toggle off the extension must not touch anything the host owns."""

  def test_update_returns_inputs_unchanged(self):
    ext = make_ext(False)
    pid, pid_log = make_pid(), SimpleNamespace(error=0.25)
    ext.update_model_v2(make_model())
    out_log, out_torque = run_update(ext, pid, pid_log, output_torque=0.42)
    assert out_log is pid_log
    assert out_log.error == pytest.approx(0.25)
    assert out_torque == pytest.approx(0.42)

  def test_pid_is_not_touched(self):
    ext = make_ext(False)
    pid = make_pid()
    before = (pid.i, pid.pos_limit, pid.neg_limit)
    ext.update_model_v2(make_model())
    run_update(ext, pid, SimpleNamespace(error=0.25))
    ext.update_limits()
    assert (pid.i, pid.pos_limit, pid.neg_limit) == before

  def test_update_limits_before_first_update_is_safe(self):
    """controlsd calls update_limits() every frame, including before the first
    update() has handed over the host's PID."""
    make_ext(True).update_limits()   # must not raise on _pid = None


class TestEnabled:
  def test_error_is_recomputed_in_torque_space(self):
    lac = make_lac(lat_accel_factor=2.5)
    ext = make_ext(True, lac=lac)
    ext.update_model_v2(make_model())
    pid_log = SimpleNamespace(error=999.0)
    out_log, _ = run_update(ext, make_pid(), pid_log, setpoint=1.0, measurement=0.8)
    # (setpoint - measurement) / latAccelFactor
    assert out_log.error == pytest.approx((1.0 - 0.8) / 2.5)

  def test_output_torque_is_replaced_by_the_torque_space_pid(self):
    ext = make_ext(True)
    pid = make_pid()
    ext.update_model_v2(make_model())
    _, out_torque = run_update(ext, pid, SimpleNamespace(error=0.0), output_torque=0.42)
    assert out_torque != pytest.approx(0.42)
    assert out_torque == pytest.approx(pid.control)

  def test_update_moves_the_host_pid_into_torque_space_by_itself(self):
    """No caller-side update_limits() needed: update() sets the bound in the same
    frame it uses it, so it cannot land on the wrong side of the host's reset."""
    lac = make_lac(steer_max=1.0, lat_accel_factor=2.5)
    ext = make_ext(True, lac=lac)
    pid = make_pid()
    pid.set_limits(2.5, -2.5)          # what the host sets, in lat-accel space
    ext.update_model_v2(make_model())
    run_update(ext, pid, SimpleNamespace(error=0.0))
    assert (pid.pos_limit, pid.neg_limit) == (1.0, -1.0)

  def test_feedforward_carries_the_torque_space_friction(self):
    lac = make_lac(lat_accel_factor=2.5, friction=0.12)
    ext = make_ext(True, lac=lac)
    ext.update_model_v2(make_model())
    run_update(ext, make_pid(), SimpleNamespace(error=0.0),
               gravity_adjusted_lateral_accel=1.0, desired_lateral_accel=1.0, actual_lateral_accel=0.0)
    # base term is gravity-adjusted lat accel in torque space...
    assert ext._ff > 1.0 / 2.5
    # ...plus at most a full friction step on top
    assert ext._ff <= 1.0 / 2.5 + 0.12 + 1e-9

  def test_integrator_freezes_when_the_driver_is_steering(self):
    ext = make_ext(True)
    ext.update_model_v2(make_model())
    pid = make_pid()
    run_update(ext, pid, SimpleNamespace(error=0.0), CS=make_CS(steering_pressed=True))
    assert pid.i == pytest.approx(0.0)

  def test_integrator_freezes_below_5_ms(self):
    ext = make_ext(True)
    ext.update_model_v2(make_model())
    pid = make_pid()
    run_update(ext, pid, SimpleNamespace(error=0.0), CS=make_CS(v_ego=3.0))
    assert pid.i == pytest.approx(0.0)


class TestJerkCalculations:
  def test_short_model_is_treated_as_invalid(self):
    ext = make_ext(True)
    ext.update_model_v2(make_model(n_orientation=CONTROL_N - 1))
    assert not ext.model_valid
    ext.update_calculations(make_CS(steering_rate_deg=10.0), make_VM(), 1.0)
    # falls back to the measured jerk alone; no model lookahead
    assert ext.lookahead_lateral_jerk == 0.0
    assert ext.actual_lateral_jerk != 0.0

  def test_missing_model_is_treated_as_invalid(self):
    ext = make_ext(True)
    ext.update_model_v2(None)
    assert not ext.model_valid

  def test_sustained_planned_jerk_survives_the_lookahead_filter(self):
    """A plan that keeps accelerating laterally in one direction is 'deliberate',
    so the friction input keeps a jerk contribution."""
    ext = make_ext(True)
    accels = list(np.linspace(0.0, 3.0, len(ModelConstants.T_IDXS)))
    ext.update_model_v2(make_model(accels))
    ext.update_calculations(make_CS(), make_VM(), 0.0)
    assert ext.lookahead_lateral_jerk > 0.0
    assert ext.lateral_jerk_setpoint == pytest.approx(ext.lat_jerk_friction_factor * ext.lookahead_lateral_jerk)

  def test_sign_flip_in_the_window_zeroes_the_jerk_term(self):
    """An S-bend inside the lookahead window is not a deliberate sustained demand."""
    ext = make_ext(True)
    accels = [0.0] * len(ModelConstants.T_IDXS)
    for i in range(LAT_PLAN_MIN_IDX, len(accels)):
      accels[i] = 1.0 if i % 2 else -1.0
    ext.update_model_v2(make_model(accels))
    ext.update_calculations(make_CS(steering_rate_deg=10.0), make_VM(), 0.0)
    assert ext.lookahead_lateral_jerk == 0.0
    # and the measured jerk is dropped with it, so friction falls back to pure error
    assert ext.actual_lateral_jerk == 0.0
    assert ext.lat_accel_friction_factor == pytest.approx(1.0)

  def test_friction_input_mixes_error_and_lookahead_jerk(self):
    ext = make_ext(True)
    ext.lat_accel_friction_factor = 0.7
    ext.lat_jerk_friction_factor = 0.4
    ext.lookahead_lateral_jerk = 2.0
    assert ext.update_friction_input(1.0, 0.5) == pytest.approx(0.7 * 0.5 + 0.4 * 2.0)

  def test_lateral_lag_update_is_floored(self):
    ext = make_ext(True, CP=make_CP(steer_actuator_delay=0.12))
    assert ext.desired_lat_jerk_time == pytest.approx(0.12)
    ext.update_lateral_lag(0.25)
    assert ext.desired_lat_jerk_time == pytest.approx(0.25)
    ext.update_lateral_lag(0.0)   # a zero lag would divide by zero in update_calculations
    assert ext.desired_lat_jerk_time == pytest.approx(0.01)

  def test_actual_jerk_tracks_steering_rate_and_speed(self):
    ext = make_ext(True)
    ext.update_model_v2(make_model(n_orientation=1))   # model invalid: isolate the measured term
    CS = make_CS(v_ego=20.0, steering_rate_deg=15.0)
    VM = make_VM(curvature=0.002)
    ext.update_calculations(CS, VM, 0.0)
    assert ext.actual_lateral_jerk == pytest.approx(-0.002 * 20.0 ** 2)
    VM.calc_curvature.assert_called_with(math.radians(15.0), 20.0, 0.0)
