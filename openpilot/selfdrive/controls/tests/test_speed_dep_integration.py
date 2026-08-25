"""Integration tests for speed-dependent torque — controller pipeline.

Tests the full data flow from torqued output through controlsd to the
torque_params used in the steering controller: per-frame latAccelFactor and friction
interpolation, sanity bounds, toggle-off behavior, and manual override.

Tests LatControlTorqueExtOverride directly (the class that owns the
per-frame interpolation logic) rather than LatControlTorqueExt, which
inherits from NNLC and requires model files to init.
"""
import numpy as np
import unittest

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from opendbc.car.structs import car
from opendbc.sunnypilot.car.interfaces import get_speed_dep_config
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext import LatControlTorqueExt
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride
from openpilot.sunnypilot.selfdrive.controls.lib.speed_dep_helpers import friction_scale
from openpilot.sunnypilot.selfdrive.test.approx import approx

SPEED_DEP_CARS = get_speed_dep_config()

# Plain unittest.TestCase, not OpenpilotTestCase: every test here mocks Params, so it needs no
# param prefix, and OpenpilotTestCase's fixture shim reads a test's signature - which @patch
# rewrites - so the injected mock arguments would be mistaken for fixtures.
PATCH_PARAMS_OVERRIDE = 'openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override.Params'
PATCH_PARAMS_JERK_AWARE = 'openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware.Params'
PATCH_PARAMS_NNLC = 'openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc.Params'
PATCH_PARAMS_TORQUED_EXT = 'openpilot.sunnypilot.selfdrive.locationd.torqued_ext.Params'
PATCH_PARAMS_TORQUED = 'openpilot.selfdrive.locationd.torqued.Params'
PATCH_GET_SPEED_DEP_CONFIG = 'opendbc.sunnypilot.car.interfaces.get_speed_dep_config'

# Sample tables
SAMPLE_SPEED_BP = [6.5, 10.0, 15.0, 21.0, 26.5, 32.0, 37.5]
SAMPLE_LAT_ACCEL_FACTOR_BP = [2.39, 2.52, 2.71, 2.39, 2.28, 2.22, 2.21]
SAMPLE_FRICTION_BP = [0.177, 0.158, 0.131, 0.118, 0.113, 0.109, 0.108]


class TorqueParams:
  """Mutable stand-in for CarParams.LateralTorqueTuning builder."""
  def __init__(self, latAccelFactor=2.0, latAccelOffset=0.0, friction=0.15):
    self.latAccelFactor = latAccelFactor
    self.latAccelOffset = latAccelOffset
    self.friction = friction


@patch(PATCH_PARAMS_OVERRIDE)
def make_override(mock_params_cls, enforce=False, manual_override=False,
                  manual_lat_accel_factor='200', manual_friction='15'):
  """Create a LatControlTorqueExtOverride with mocked Params."""
  mock_inst = mock_params_cls.return_value
  mock_inst.get_bool.side_effect = lambda k: {
    'EnforceTorqueControl': enforce,
    'TorqueParamsOverrideEnabled': manual_override,
  }.get(k, False)
  mock_inst.get.side_effect = lambda k, **kw: {
    'TorqueParamsOverrideLatAccelFactor': manual_lat_accel_factor,
    'TorqueParamsOverrideFriction': manual_friction,
  }.get(k)

  CP = MagicMock()
  ovr = LatControlTorqueExtOverride(CP)
  # Production provides lac_torque via LatControlTorqueExtBase; the speed-dep
  # block calls lac_torque.update_limits() after applying interpolated params.
  ovr.lac_torque = MagicMock()
  return ovr


def activate_speed_dep(ovr, speed_bp=None, lat_accel_factor_bp=None, friction_bp=None):
  """Simulate update_speed_dep_torque setting tables on the override."""
  ovr._speed_dep_active = True
  ovr._speed_dep_speed_bp = speed_bp or list(SAMPLE_SPEED_BP)
  ovr._speed_dep_lat_accel_factor_bp = lat_accel_factor_bp or list(SAMPLE_LAT_ACCEL_FACTOR_BP)
  ovr._speed_dep_friction_bp = friction_bp or list(SAMPLE_FRICTION_BP)


class TestLafInterpolatedBySpeed(unittest.TestCase):
  """torque_params.latAccelFactor must be speed-interpolated
  before torque_from_lateral_accel reads it."""

  def test_lat_accel_factor_set_to_interpolated_value(self):
    ovr = make_override()
    activate_speed_dep(ovr)
    tp = TorqueParams(latAccelFactor=999.0)  # sentinel

    ovr._last_vego = 10.0
    ovr.update_override_torque_params(tp)

    expected = float(np.interp(10.0, SAMPLE_SPEED_BP, SAMPLE_LAT_ACCEL_FACTOR_BP))
    assert tp.latAccelFactor == approx(expected, atol=1e-4), \
      f"latAccelFactor should be {expected}, got {tp.latAccelFactor}"

  def test_lat_accel_factor_differs_at_different_speeds(self):
    ovr = make_override()
    activate_speed_dep(ovr)

    tp = TorqueParams()
    ovr._last_vego = 6.5
    ovr.update_override_torque_params(tp)
    factor_low = tp.latAccelFactor

    tp = TorqueParams()
    ovr._last_vego = 37.5
    ovr.update_override_torque_params(tp)
    factor_high = tp.latAccelFactor

    assert factor_low != approx(factor_high, atol=0.01), \
      "latAccelFactor must differ between 6.5 m/s and 37.5 m/s"

  def test_lat_accel_factor_not_global_value(self):
    """latAccelFactor should NOT be the global scalar."""
    ovr = make_override()
    activate_speed_dep(ovr)
    global_factor = 2.0
    tp = TorqueParams(latAccelFactor=global_factor)

    ovr._last_vego = 6.5  # seed latAccelFactor at 6.5 is 2.39, not 2.0
    ovr.update_override_torque_params(tp)

    assert tp.latAccelFactor != approx(global_factor, atol=0.01), \
      "latAccelFactor should be speed-interpolated, not the global value"


class TestFrictionInterpolatedBySpeed(unittest.TestCase):
  """torque_params.friction must be speed-interpolated
  before get_friction reads it."""

  def test_friction_set_to_interpolated_value(self):
    ovr = make_override()
    activate_speed_dep(ovr)
    tp = TorqueParams(friction=999.0)

    ovr._last_vego = 35.0
    ovr.update_override_torque_params(tp)

    expected = float(np.interp(35.0, SAMPLE_SPEED_BP, SAMPLE_FRICTION_BP))
    assert tp.friction == approx(expected, atol=1e-4)

  def test_friction_differs_at_different_speeds(self):
    ovr = make_override()
    activate_speed_dep(ovr)

    tp = TorqueParams()
    ovr._last_vego = 6.5
    ovr.update_override_torque_params(tp)
    fric_low = tp.friction

    tp = TorqueParams()
    ovr._last_vego = 37.5
    ovr.update_override_torque_params(tp)
    fric_high = tp.friction

    assert fric_low != approx(fric_high, atol=0.01)


class TestToggleOffClearsState(unittest.TestCase):
  """_speed_dep_active must be cleared when bins disappear."""

  def test_inactive_by_default(self):
    ovr = make_override()
    assert not ovr._speed_dep_active

  def test_deactivated_does_not_modify_params(self):
    ovr = make_override()
    activate_speed_dep(ovr)
    # Deactivate
    ovr._speed_dep_active = False

    tp = TorqueParams(latAccelFactor=99.0, friction=99.0)
    ovr._last_vego = 15.0
    ovr.update_override_torque_params(tp)

    assert tp.latAccelFactor == 99.0, "Should not modify params when inactive"
    assert tp.friction == 99.0

  def test_empty_speed_bp_does_not_modify_params(self):
    ovr = make_override()
    activate_speed_dep(ovr)
    ovr._speed_dep_speed_bp = []  # empty

    tp = TorqueParams(latAccelFactor=99.0, friction=99.0)
    ovr._last_vego = 15.0
    ovr.update_override_torque_params(tp)

    assert tp.latAccelFactor == 99.0
    assert tp.friction == 99.0


class TestManualOverridePriority(unittest.TestCase):
  """Manual override must take priority over speed-dep."""

  def test_manual_overwrites_speed_dep(self):
    ovr = make_override(enforce=True, manual_override=True,
                        manual_lat_accel_factor='350', manual_friction='25')
    activate_speed_dep(ovr)
    ovr._last_vego = 15.0

    tp = TorqueParams()
    # frame = -1, after +1 -> frame=0, 0 % 300 == 0 -> manual fires
    ovr.update_override_torque_params(tp)

    assert tp.latAccelFactor == approx(350.0, atol=0.1), \
      "Manual latAccelFactor should overwrite speed-dep"
    assert tp.friction == approx(25.0, atol=0.1), \
      "Manual friction should overwrite speed-dep"

  def test_speed_dep_used_when_manual_off(self):
    ovr = make_override(enforce=True, manual_override=False)
    activate_speed_dep(ovr)
    ovr._last_vego = 15.0

    tp = TorqueParams()
    ovr.update_override_torque_params(tp)

    expected_factor = float(np.interp(15.0, SAMPLE_SPEED_BP, SAMPLE_LAT_ACCEL_FACTOR_BP))
    assert tp.latAccelFactor == approx(expected_factor, atol=1e-4), \
      "Without manual override, speed-dep should be used"


class TestSpeedDepLimitHandling(unittest.TestCase):
  """The speed-dep block must re-derive the PID limits itself and never report
  changed=True. Returning True makes LatControlTorque.update() call its own
  update_limits() AFTER controlsd has pinned the shared PID to torque-space
  bounds for the jerk/NNLC controllers, which re-anchors integrator anti-windup
  a factor latAccelFactor above steer_max (windup to ~2x steer_max, commanded
  torque ~2.5x, reproduced 2026-08-23)."""

  def test_never_reports_changed(self):
    ovr = make_override()
    activate_speed_dep(ovr)

    tp = TorqueParams()
    ovr._last_vego = 6.5
    assert ovr.update_override_torque_params(tp) is False

    ovr._last_vego = 37.5  # big speed change -> values change, still not reported
    assert ovr.update_override_torque_params(tp) is False, \
      "speed-dep must never return True: the caller would re-widen the pinned torque-space limits"

  def test_rederives_both_limit_spaces(self):
    ovr = make_override()
    activate_speed_dep(ovr)
    ovr.update_limits = MagicMock()  # instance spy over the extension hook
    tp = TorqueParams()
    ovr._last_vego = 15.0

    ovr.update_override_torque_params(tp)

    ovr.lac_torque.update_limits.assert_called_once()  # lat-accel bounds for the new params
    ovr.update_limits.assert_called_once()  # torque-space re-pin (no-op unless jerk/NNLC armed)

  def test_inactive_touches_no_limits(self):
    ovr = make_override()
    ovr.update_limits = MagicMock()
    tp = TorqueParams()
    ovr._last_vego = 15.0

    ovr.update_override_torque_params(tp)

    ovr.lac_torque.update_limits.assert_not_called()
    ovr.update_limits.assert_not_called()

  def test_manual_override_still_reports_changed(self):
    ovr = make_override(enforce=True, manual_override=True)
    tp = TorqueParams()
    # frame = -1, after +1 -> frame=0, 0 % 300 == 0 -> manual fires (upstream path)
    assert ovr.update_override_torque_params(tp) is True


class TestLearnerSanityBounds(unittest.TestCase):
  """Speed-bin sanity bounds must allow learning regardless of
  the 'Less Restrict' toggle."""

  @patch(PATCH_PARAMS_TORQUED_EXT)
  @patch(PATCH_PARAMS_TORQUED)
  def test_sanity_bounds_allow_learning_without_relaxed(self, mock_params_cls, mock_ext_cls):
    """With LiveTorqueParamsRelaxedToggle OFF (factor_sanity=0.0),
    speed bins must still have +/-30% bounds, not (seed, seed)."""
    mock_params_cls.return_value.get.return_value = None
    mock_ext_cls.return_value.get_bool.side_effect = lambda k: {
      'SpeedDependentTorqueToggle': True,
      'EnforceTorqueControl': True,  # required for speed-dep; Relaxed off keeps upstream sanity
      'LiveTorqueParamsRelaxedToggle': False,
    }.get(k, False)
    mock_ext_cls.return_value.get.return_value = None

    from openpilot.selfdrive.locationd.torqued import TorqueEstimator
    CP = MagicMock()
    CP.brand = 'test'
    CP.carFingerprint = next(iter(SPEED_DEP_CARS)) if SPEED_DEP_CARS else 'FAKE'
    CP.lateralTuning.which.return_value = 'torque'
    CP.lateralTuning.torque.latAccelFactor = 2.0
    CP.lateralTuning.torque.friction = 0.15

    est = TorqueEstimator(CP)
    est._on_torque_point(0.1, 0.3, 10.0)  # trigger lazy init

    for i, (lo, hi) in enumerate(est.speed_bin_lat_accel_factor_bounds):
      assert hi > lo, f"Bin {i} latAccelFactor bounds ({lo:.3f}, {hi:.3f}) must allow a range"

    for i, (lo, hi) in enumerate(est.speed_bin_friction_bounds):
      assert hi > lo, f"Bin {i} friction bounds ({lo:.3f}, {hi:.3f}) must allow a range"

  @patch(PATCH_PARAMS_TORQUED_EXT)
  @patch(PATCH_PARAMS_TORQUED)
  def test_clip_allows_10pct_movement(self, mock_params_cls, mock_ext_cls):
    """A learned value 10% above seed should pass through np.clip with +/-30% bounds."""
    mock_params_cls.return_value.get.return_value = None
    mock_ext_cls.return_value.get_bool.side_effect = lambda k: {
      'SpeedDependentTorqueToggle': True,
      'EnforceTorqueControl': True,
    }.get(k, False)
    mock_ext_cls.return_value.get.return_value = None

    from openpilot.selfdrive.locationd.torqued import TorqueEstimator
    CP = MagicMock()
    CP.brand = 'test'
    CP.carFingerprint = next(iter(SPEED_DEP_CARS)) if SPEED_DEP_CARS else 'FAKE'
    CP.lateralTuning.which.return_value = 'torque'
    CP.lateralTuning.torque.latAccelFactor = 2.0
    CP.lateralTuning.torque.friction = 0.15

    est = TorqueEstimator(CP)
    est._on_torque_point(0.1, 0.3, 10.0)

    seed_factor = est.speed_bin_filtered[0]['latAccelFactor'].x
    nudged = seed_factor * 1.10
    lo, hi = est.speed_bin_lat_accel_factor_bounds[0]
    clipped = np.clip(nudged, lo, hi)
    assert clipped == approx(nudged, atol=1e-6), \
      f"+10% nudge ({nudged:.3f}) should not be clipped by +/-30% bounds ({lo:.3f}, {hi:.3f})"


class TestToggleOffFallback(unittest.TestCase):
  """When speed-dep is deactivated, controller must not use stale tables."""

  def test_deactivation_via_empty_bins(self):
    """Simulates toggle-off: update_speed_dep_torque receives empty bins."""
    ovr = make_override()
    activate_speed_dep(ovr)
    assert ovr._speed_dep_active

    # Simulate toggle-off (torqued sends empty speedBinCenters)
    ovr._speed_dep_active = False

    tp = TorqueParams(latAccelFactor=2.35, friction=0.12)
    ovr._last_vego = 15.0
    ovr.update_override_torque_params(tp)

    # Global values should pass through unmodified
    assert tp.latAccelFactor == 2.35
    assert tp.friction == 0.12

  def test_reactivation_after_deactivation(self):
    """Speed-dep can be re-enabled after being disabled."""
    ovr = make_override()
    ovr._speed_dep_active = False

    # Re-enable
    activate_speed_dep(ovr)
    assert ovr._speed_dep_active

    tp = TorqueParams()
    ovr._last_vego = 15.0
    ovr.update_override_torque_params(tp)

    expected_factor = float(np.interp(15.0, SAMPLE_SPEED_BP, SAMPLE_LAT_ACCEL_FACTOR_BP))
    assert tp.latAccelFactor == approx(expected_factor, atol=1e-4)


class TestUpdateSpeedDepTorqueFallback(unittest.TestCase):
  """Tests for update_speed_dep_torque fallback logic (TOML seeds vs global filtered)."""

  @staticmethod
  def _make_mock_tp(speed_bp, lafs, frictions, valid, global_laf=2.0, global_fric=0.15):
    tp = MagicMock()
    tp.speedBinCenters = speed_bp
    tp.speedBinLatAccelFactors = lafs
    tp.speedBinFrictions = frictions
    tp.speedBinValid = valid
    tp.latAccelFactorFiltered = global_laf
    tp.frictionCoefficientFiltered = global_fric
    tp.latAccelOffsetFiltered = 0.0
    return tp

  @staticmethod
  def _make_mock_self(fingerprint='TEST_CAR'):
    mock_self = MagicMock()
    mock_self.CP.carFingerprint = fingerprint
    mock_self._speed_dep_active = False
    mock_self._speed_dep_speed_bp = []
    mock_self._speed_dep_lat_accel_factor_bp = []
    mock_self._speed_dep_friction_bp = []
    mock_self._speed_dep_car_cfg = None
    return mock_self

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_toml_seeds_used_for_invalid_bins(self, mock_get_config):
    """Invalid bins should fall back to TOML seed values, not global filtered."""
    seed_lafs = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
    seed_frictions = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    mock_get_config.return_value = {
      'TEST_CAR': {'speed_bp': SAMPLE_SPEED_BP, 'laf_bp': seed_lafs, 'friction_bp': seed_frictions}
    }

    mock_self = self._make_mock_self()
    mock_tp = self._make_mock_tp(SAMPLE_SPEED_BP, [999.0] * 7, [999.0] * 7,
                                 [False] * 7, global_laf=1.0, global_fric=0.05)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp)

    assert mock_self._speed_dep_lat_accel_factor_bp == seed_lafs
    assert mock_self._speed_dep_friction_bp == seed_frictions

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_global_filtered_fallback_when_no_config(self, mock_get_config):
    """Unconfigured car: invalid bins should use global filtered values."""
    mock_get_config.return_value = {}

    mock_self = self._make_mock_self(fingerprint='UNKNOWN_CAR')
    mock_tp = self._make_mock_tp(SAMPLE_SPEED_BP, [999.0] * 7, [999.0] * 7,
                                 [False] * 7, global_laf=2.0, global_fric=0.15)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp)

    assert mock_self._speed_dep_lat_accel_factor_bp == [2.0] * 7
    assert mock_self._speed_dep_friction_bp == [0.15] * 7

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_friction_bp_missing_uses_global_fallback(self, mock_get_config):
    """Config with laf_bp but no friction_bp should use global fallback (not crash)."""
    mock_get_config.return_value = {
      'TEST_CAR': {'speed_bp': SAMPLE_SPEED_BP, 'laf_bp': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]}
      # 'friction_bp' intentionally missing
    }

    mock_self = self._make_mock_self()
    mock_tp = self._make_mock_tp(SAMPLE_SPEED_BP, [999.0] * 7, [999.0] * 7,
                                 [False] * 7, global_laf=2.0, global_fric=0.15)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp)

    assert mock_self._speed_dep_lat_accel_factor_bp == [2.0] * 7
    assert mock_self._speed_dep_friction_bp == [0.15] * 7

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_laf_bp_length_mismatch_uses_global_fallback(self, mock_get_config):
    """Config with wrong-length laf_bp should use global fallback."""
    mock_get_config.return_value = {
      'TEST_CAR': {'speed_bp': SAMPLE_SPEED_BP, 'laf_bp': [2.1, 2.2], 'friction_bp': [0.1, 0.2]}
    }

    mock_self = self._make_mock_self()
    mock_tp = self._make_mock_tp(SAMPLE_SPEED_BP, [999.0] * 7, [999.0] * 7,
                                 [False] * 7, global_laf=2.0, global_fric=0.15)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp)

    assert mock_self._speed_dep_lat_accel_factor_bp == [2.0] * 7
    assert mock_self._speed_dep_friction_bp == [0.15] * 7

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_mixed_valid_invalid_bins(self, mock_get_config):
    """Valid bins use learned values, invalid bins use TOML seeds."""
    seed_lafs = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
    seed_frictions = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    mock_get_config.return_value = {
      'TEST_CAR': {'speed_bp': SAMPLE_SPEED_BP, 'laf_bp': seed_lafs, 'friction_bp': seed_frictions}
    }

    learned_lafs = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6]
    learned_frictions = [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27]
    valid = [True, False, True, False, True, False, False]

    mock_self = self._make_mock_self()
    mock_tp = self._make_mock_tp(SAMPLE_SPEED_BP, learned_lafs, learned_frictions, valid)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp)

    for i in range(7):
      if valid[i]:
        assert mock_self._speed_dep_lat_accel_factor_bp[i] == learned_lafs[i]
        assert mock_self._speed_dep_friction_bp[i] == learned_frictions[i]
      else:
        assert mock_self._speed_dep_lat_accel_factor_bp[i] == seed_lafs[i]
        assert mock_self._speed_dep_friction_bp[i] == seed_frictions[i]

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_empty_bins_deactivate(self, mock_get_config):
    mock_get_config.return_value = {}
    mock_self = self._make_mock_self()
    mock_self._speed_dep_active = True

    mock_tp = MagicMock()
    mock_tp.speedBinCenters = []

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp)

    assert mock_self._speed_dep_active is False


class TestNearestLearnedBinFallback(unittest.TestCase):
  """Without TOML seeds, unlearned bins fall back to the nearest learned bin:
  the interp table holds learned bins only, so np.interp clamps to the nearest
  learned value beyond the ends and bridges unlearned gaps. Global values are
  used only when nothing is learned yet."""

  LEARNED_LAFS = [9.9, 2.39, 2.55, 2.61, 3.12, 3.09, 9.9]  # 9.9 = sentinel, must never be used
  LEARNED_FRICTIONS = [9.9, 0.113, 0.106, 0.086, 0.076, 0.078, 9.9]
  EDGE_INVALID = [False, True, True, True, True, True, False]

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_edge_bins_extend_nearest_learned(self, mock_get_config):
    mock_get_config.return_value = {}
    mock_self = TestUpdateSpeedDepTorqueFallback._make_mock_self(fingerprint='UNKNOWN_CAR')
    mock_tp = TestUpdateSpeedDepTorqueFallback._make_mock_tp(
      SAMPLE_SPEED_BP, self.LEARNED_LAFS, self.LEARNED_FRICTIONS, self.EDGE_INVALID,
      global_laf=2.0, global_fric=0.15)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp)

    assert mock_self._speed_dep_speed_bp == SAMPLE_SPEED_BP[1:6]
    assert mock_self._speed_dep_lat_accel_factor_bp == self.LEARNED_LAFS[1:6]
    assert mock_self._speed_dep_friction_bp == self.LEARNED_FRICTIONS[1:6]

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_controller_clamps_to_nearest_learned_not_global(self, mock_get_config):
    """End-to-end through the per-frame interpolation: outside the learned span the
    controller must use the nearest learned bin, not the global value."""
    mock_get_config.return_value = {}
    ovr = make_override()
    ovr.lac_torque = MagicMock()
    mock_tp = TestUpdateSpeedDepTorqueFallback._make_mock_tp(
      SAMPLE_SPEED_BP, self.LEARNED_LAFS, self.LEARNED_FRICTIONS, self.EDGE_INVALID,
      global_laf=2.0, global_fric=0.15)
    LatControlTorqueExt.update_speed_dep_torque(ovr, mock_tp)

    tp = TorqueParams()
    ovr._last_vego = 2.0  # below the lowest learned bin (10.0)
    ovr.update_override_torque_params(tp)
    assert tp.latAccelFactor == approx(2.39, atol=1e-4)
    assert tp.friction == approx(0.113, atol=1e-4)

    tp = TorqueParams()
    ovr._last_vego = 45.0  # above the highest learned bin (32.0)
    ovr.update_override_torque_params(tp)
    assert tp.latAccelFactor == approx(3.09, atol=1e-4)
    assert tp.friction == approx(0.078, atol=1e-4)

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_interior_gap_bridges_learned_neighbors(self, mock_get_config):
    """An unlearned bin between two learned bins interpolates across them."""
    mock_get_config.return_value = {}
    valid = [False, True, False, True, True, True, False]
    mock_self = TestUpdateSpeedDepTorqueFallback._make_mock_self(fingerprint='UNKNOWN_CAR')
    mock_tp = TestUpdateSpeedDepTorqueFallback._make_mock_tp(
      SAMPLE_SPEED_BP, self.LEARNED_LAFS, self.LEARNED_FRICTIONS, valid,
      global_laf=2.0, global_fric=0.15)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp)

    # at the unlearned 15.0 bin center, value bridges the 10.0 and 21.0 learned bins
    expected = float(np.interp(15.0, [10.0, 21.0], [2.39, 2.61]))
    got = float(np.interp(15.0, mock_self._speed_dep_speed_bp, mock_self._speed_dep_lat_accel_factor_bp))
    assert got == approx(expected, atol=1e-4)
    assert 9.9 not in mock_self._speed_dep_lat_accel_factor_bp
    assert 2.0 not in mock_self._speed_dep_lat_accel_factor_bp


class TestFrictionReduction(unittest.TestCase):
  """The Friction Reduction setting scales learned friction only — never TOML
  seeds, never the global fallback, never latAccelFactor."""

  def test_friction_scale_steps_and_clamping(self):
    assert friction_scale(0) == approx(1.0)
    assert friction_scale(5) == approx(0.5)
    assert friction_scale(9) == approx(0.1)
    assert friction_scale(-3) == approx(1.0)
    assert friction_scale(42) == approx(0.1)

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_reduction_scales_learned_friction_only(self, mock_get_config):
    mock_get_config.return_value = {}
    lafs = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
    frictions = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    mock_self = TestUpdateSpeedDepTorqueFallback._make_mock_self(fingerprint='UNKNOWN_CAR')
    mock_tp = TestUpdateSpeedDepTorqueFallback._make_mock_tp(
      SAMPLE_SPEED_BP, lafs, frictions, [True] * 7)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp, friction_reduction=5)

    assert mock_self._speed_dep_friction_bp == approx([f * 0.5 for f in frictions])
    assert mock_self._speed_dep_lat_accel_factor_bp == approx(lafs)

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_reduction_not_applied_to_global_fallback(self, mock_get_config):
    mock_get_config.return_value = {}
    mock_self = TestUpdateSpeedDepTorqueFallback._make_mock_self(fingerprint='UNKNOWN_CAR')
    mock_tp = TestUpdateSpeedDepTorqueFallback._make_mock_tp(
      SAMPLE_SPEED_BP, [9.9] * 7, [9.9] * 7, [False] * 7, global_laf=2.0, global_fric=0.15)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp, friction_reduction=9)

    assert mock_self._speed_dep_friction_bp == approx([0.15] * 7)

  @patch(PATCH_GET_SPEED_DEP_CONFIG)
  def test_reduction_spares_toml_seeds(self, mock_get_config):
    seed_lafs = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
    seed_frictions = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    mock_get_config.return_value = {
      'TEST_CAR': {'speed_bp': SAMPLE_SPEED_BP, 'laf_bp': seed_lafs, 'friction_bp': seed_frictions}
    }
    learned_frictions = [0.2] * 7
    valid = [True, False, True, False, True, False, False]
    mock_self = TestUpdateSpeedDepTorqueFallback._make_mock_self()
    mock_tp = TestUpdateSpeedDepTorqueFallback._make_mock_tp(
      SAMPLE_SPEED_BP, [3.0] * 7, learned_frictions, valid)

    LatControlTorqueExt.update_speed_dep_torque(mock_self, mock_tp, friction_reduction=5)

    for i in range(7):
      if valid[i]:
        assert mock_self._speed_dep_friction_bp[i] == approx(0.1), f"learned bin {i} must be scaled"
      else:
        assert mock_self._speed_dep_friction_bp[i] == approx(seed_frictions[i]), f"seed bin {i} must not be scaled"


class TestExtrapolationAtBoundaries(unittest.TestCase):
  """np.interp clamps to edge values for speeds outside the bin range."""

  def test_speed_below_first_bin_clamps(self):
    ovr = make_override()
    activate_speed_dep(ovr)
    tp = TorqueParams()
    ovr._last_vego = 0.0
    ovr.update_override_torque_params(tp)
    assert tp.latAccelFactor == approx(SAMPLE_LAT_ACCEL_FACTOR_BP[0], atol=1e-4)
    assert tp.friction == approx(SAMPLE_FRICTION_BP[0], atol=1e-4)

  def test_speed_above_last_bin_clamps(self):
    ovr = make_override()
    activate_speed_dep(ovr)
    tp = TorqueParams()
    ovr._last_vego = 100.0
    ovr.update_override_torque_params(tp)
    assert tp.latAccelFactor == approx(SAMPLE_LAT_ACCEL_FACTOR_BP[-1], atol=1e-4)
    assert tp.friction == approx(SAMPLE_FRICTION_BP[-1], atol=1e-4)

  def test_speed_at_exact_bin_center(self):
    ovr = make_override()
    activate_speed_dep(ovr)
    for i, speed in enumerate(SAMPLE_SPEED_BP):
      tp = TorqueParams()
      ovr._last_vego = speed
      ovr.update_override_torque_params(tp)
      assert tp.latAccelFactor == approx(SAMPLE_LAT_ACCEL_FACTOR_BP[i], atol=1e-4)
      assert tp.friction == approx(SAMPLE_FRICTION_BP[i], atol=1e-4)


class TestJerkControllerWindupBounded(unittest.TestCase):
  """Full-controller regression for speed-dep + Lateral Jerk Torque Controller.

  With both armed, every write to the shared PID must run under torque-space
  bounds. Before the fix, the speed-dep changed=True return made
  LatControlTorque.update() re-widen the limits to lat-accel space every frame
  (after controlsd's torque-space pinning), so the integrator wound to ~2x
  steer_max and the controller returned ~2.5x steer_max."""

  LAF, FRIC, SR, WB = 2.5, 0.12, 15.0, 2.7
  FRAMES = 1200  # windup previously exceeded steer_max by frame ~319

  def _make_controller(self, jerk_on):
    def flags(k):
      return {'LateralJerkTorqueController': jerk_on}.get(k, False)

    CP = car.CarParams.new_message()
    CP.steerLimitTimer = 0.4
    CP.steerActuatorDelay = 0.12
    CP.steerRatio = self.SR
    t = CP.lateralTuning.init('torque')
    t.latAccelFactor = self.LAF
    t.friction = self.FRIC

    CI = MagicMock()
    CI.torque_from_lateral_accel.return_value = lambda la, tp: la / tp.latAccelFactor
    CI.lateral_accel_from_torque.return_value = lambda torque, tp: torque * tp.latAccelFactor
    CI.torque_from_lateral_accel_in_torque_space.return_value = (
      lambda inp, tp, gravity_adjusted: inp.lateral_acceleration / float(tp.latAccelFactor))

    CP_SP = MagicMock()
    CP_SP.neuralNetworkLateralControl.model.path = ''

    patchers = [patch(p) for p in (PATCH_PARAMS_OVERRIDE, PATCH_PARAMS_JERK_AWARE, PATCH_PARAMS_NNLC)]
    for p in patchers:
      mock_inst = p.start().return_value
      mock_inst.get_bool.side_effect = flags
      mock_inst.get.side_effect = lambda k, **kw: None
      self.addCleanup(p.stop)
    return LatControlTorque(CP.as_reader(), CP_SP, CI, DT_CTRL)

  def _run_steady_error(self, lac, v_ego=25.0):
    ext = lac.extension
    n = len(ModelConstants.T_IDXS)
    ext.update_model_v2(SimpleNamespace(acceleration=SimpleNamespace(y=[0.0] * n),
                                        orientation=SimpleNamespace(x=[0.0] * n)))
    ext.update_lateral_lag(0.15)
    ext._speed_dep_active = True
    ext._speed_dep_speed_bp = [5.0, 20.0, 40.0]
    ext._speed_dep_lat_accel_factor_bp = [self.LAF] * 3
    ext._speed_dep_friction_bp = [self.FRIC] * 3

    VM = MagicMock()
    VM.calc_curvature.side_effect = lambda angle, v, roll: angle / (self.SR * self.WB)
    CS = SimpleNamespace(vEgo=v_ego, aEgo=0.0, steeringAngleDeg=1.0, steeringRateDeg=0.0, steeringPressed=False)
    vehicle_params = SimpleNamespace(roll=0.0, angleOffsetDeg=0.0)

    max_i, max_torque = 0.0, 0.0
    for _ in range(self.FRAMES):
      # controlsd per-frame sequence: global params, then the extension's torque-space pinning
      lac.update_torque_parameters(self.LAF, 0.0, self.FRIC)
      lac.extension.update_limits()
      torque, _, _ = lac.update(True, CS, VM, vehicle_params, False, 0.0006, None, False, 0.15)
      max_i = max(max_i, abs(lac.pid.i))
      max_torque = max(max_torque, abs(torque))
    return max_i, max_torque

  def test_jerk_on_stays_bounded_at_steer_max(self):
    lac = self._make_controller(jerk_on=True)
    max_i, max_torque = self._run_steady_error(lac)

    assert lac.pid.pos_limit == approx(lac.steer_max), \
      "with the jerk controller armed, speed-dep must not re-widen the torque-space limits"
    assert max_i <= lac.steer_max + 1e-6, f"integrator wound past steer_max: {max_i:.4f}"
    assert max_torque <= lac.steer_max + 1e-6, f"commanded torque exceeded steer_max: {max_torque:.4f}"

  def test_jerk_off_keeps_lat_accel_limits(self):
    lac = self._make_controller(jerk_on=False)
    max_i, max_torque = self._run_steady_error(lac)

    # stock behavior: limits in lat-accel space, derived from the speed-interpolated params
    assert lac.pid.pos_limit == approx(lac.steer_max * self.LAF), \
      "with the jerk controller off, speed-dep must still apply lat-accel-space limits"
    assert max_torque <= lac.steer_max + 1e-6
