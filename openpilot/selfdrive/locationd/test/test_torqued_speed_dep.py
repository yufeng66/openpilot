"""Tests for speed-binned learning in torqued (vehicle-agnostic).

Uses get_speed_dep_config() to discover configured cars.
All tests are driven by config, not hardcoded fingerprints.
"""
import math

import numpy as np
import pytest

from unittest.mock import MagicMock, patch
from opendbc.car.structs import car
from openpilot.cereal import log
from opendbc.sunnypilot.car.interfaces import get_speed_dep_config
from openpilot.selfdrive.locationd.torqued import (
  TorqueEstimator, TorqueBuckets, VERSION, MIN_FILTER_DECAY,
  STEER_BUCKET_BOUNDS, FRICTION_FACTOR, slope2rot,
)
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import (
  DEFAULT_SPEED_BIN_BOUNDS as SPEED_BIN_BOUNDS, DEFAULT_SPEED_BIN_CENTERS as SPEED_BIN_CENTERS,
  TorqueEstimatorExt, SpeedBinMoment, SpeedBinMomentBank, MOMENT_SPEED_BIN_CENTERS,
  MOMENT_ESS_CAP, MOMENT_MIN_ESS, MOMENT_MIN_XSTD, MOMENT_CACHE_ROW,
  MOMENT_DENSITY_CEILING, MOMENT_DENSITY_FLOOR, MOMENT_SPEED_KERNEL_H, MOMENT_KERNEL_MIN,
  MOMENT_MIN_IN_BIN_ESS, MOMENT_MIN_LAT_ACCEL_FACTOR,
)

# Discover configured cars
SPEED_DEP_CARS = get_speed_dep_config()
SPEED_DEP_FINGERPRINT = next(iter(SPEED_DEP_CARS)) if SPEED_DEP_CARS else None

# Sentinel fingerprint that must not appear in speed_dependent.toml
NON_SPEED_DEP_FINGERPRINT = 'NOT_IN_SPEED_DEP_TOML'
assert NON_SPEED_DEP_FINGERPRINT not in SPEED_DEP_CARS, f"{NON_SPEED_DEP_FINGERPRINT} unexpectedly in speed_dependent.toml"

# Both Params locations need mocking: torqued.py (cache) and torqued_ext.py (toggles)
PATCH_PARAMS = 'openpilot.selfdrive.locationd.torqued.Params'
PATCH_EXT_PARAMS = 'openpilot.sunnypilot.selfdrive.locationd.torqued_ext.Params'


def _setup_ext_mock(mock_ext_params_cls, speed_dep_on, enforce_on=None, moment_on=False):
  """Configure the torqued_ext Params mock for toggle state.
  EnforceTorqueControl follows speed_dep_on unless overridden: speed-dep
  requires Enforce since its settings live behind the Enforce-gated panel."""
  if enforce_on is None:
    enforce_on = speed_dep_on
  def _get_bool(param):
    if param == "SpeedDependentTorqueToggle":
      return speed_dep_on
    if param == "EnforceTorqueControl":
      return enforce_on
    if param == "SpeedDependentTorqueMomentToggle":
      return moment_on
    return False
  mock_ext_params_cls.return_value.get_bool.side_effect = _get_bool
  mock_ext_params_cls.return_value.get.return_value = None


def make_mock_CP(fingerprint=None, lat_accel_factor=1.25, friction=0.125):
  if fingerprint is None:
    fingerprint = SPEED_DEP_FINGERPRINT
  CP = MagicMock()
  CP.brand = 'test'
  CP.carFingerprint = fingerprint
  CP.lateralTuning.which.return_value = 'torque'
  CP.lateralTuning.torque.friction = friction
  CP.lateralTuning.torque.latAccelFactor = lat_accel_factor
  return CP


class TestSpeedDepConfig:
  """Config-level tests that don't need a TorqueEstimator."""

  @pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
  def test_speed_dep_config_has_entries(self):
    assert len(SPEED_DEP_CARS) > 0

  def test_version_exists(self):
    assert VERSION >= 1

  def test_speed_bin_bounds_cover_full_range(self):
    all_bounds = [b for bounds in SPEED_BIN_BOUNDS for b in bounds]
    assert min(all_bounds) == 5
    assert max(all_bounds) >= 35

  def test_speed_bin_centers_match_bounds(self):
    for center, (lo, hi) in zip(SPEED_BIN_CENTERS, SPEED_BIN_BOUNDS, strict=True):
      assert center >= lo
      assert center <= hi


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestSpeedBinnedLearning:
  """Test speed-binned learning with toggle ON."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_speed_bins_initialized(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    for fingerprint in SPEED_DEP_CARS:
      est = TorqueEstimator(make_mock_CP(fingerprint=fingerprint))
      assert est.speed_binned
      # Bins are lazy-initialized on first point
      est._on_torque_point(0.1, 0.3, 10.0)
      assert len(est.speed_bin_points) == len(SPEED_BIN_BOUNDS)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_speed_bin_routing(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    for bin_idx, (lo, hi) in enumerate(SPEED_BIN_BOUNDS):
      est = TorqueEstimator(make_mock_CP())
      vego = (lo + hi) / 2.0
      est._on_torque_point(0.1, 0.3, vego)
      assert len(est.speed_bin_points[bin_idx]) == 1, \
        f"bin {bin_idx} ({lo}-{hi} m/s) should have 1 point at vego={vego}"
      for j in range(len(SPEED_BIN_BOUNDS)):
        if j != bin_idx:
          assert len(est.speed_bin_points[j]) == 0, \
            f"bin {j} should be empty when vego={vego}"

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cereal_message_fields(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    for fingerprint in SPEED_DEP_CARS:
      est = TorqueEstimator(make_mock_CP(fingerprint=fingerprint))
      # Trigger lazy bin init so _extend_msg populates fields
      est._on_torque_point(0.1, 0.3, 10.0)
      msg = est.get_msg()
      ltp = msg.lateralTorqueParameters
      assert len(ltp.speedBinCenters) == len(SPEED_BIN_CENTERS)
      assert len(ltp.speedBinLatAccelFactors) == len(SPEED_BIN_BOUNDS)
      assert len(ltp.speedBinFrictions) == len(SPEED_BIN_BOUNDS)
      assert len(ltp.speedBinValid) == len(SPEED_BIN_BOUNDS)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_global_fit_unchanged(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP(lat_accel_factor=1.25, friction=0.125))
    msg = est.get_msg()
    ltp = msg.lateralTorqueParameters
    assert ltp.latAccelFactorFiltered == pytest.approx(1.25, abs=1e-2)
    assert ltp.frictionCoefficientFiltered == pytest.approx(0.125, abs=1e-3)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_global_buckets_still_require_min_vel(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    assert len(est.filtered_points) == 0


class TestToggleGate:
  """Toggle OFF should disable speed-binning even for configured cars."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_toggle_off_no_speed_bins(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    if SPEED_DEP_FINGERPRINT:
      est = TorqueEstimator(make_mock_CP(fingerprint=SPEED_DEP_FINGERPRINT))
      assert not est.speed_binned

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_enforce_off_no_speed_bins(self, mock_params_cls, mock_ext):
    """Speed-dep requires EnforceTorqueControl even with its own toggle on."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, enforce_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    assert not est.speed_binned


class TestBackwardCompatibility:
  """Cars with toggle OFF should be unaffected."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_unconfigured_car_no_speed_bins(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    assert not est.speed_binned

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_unconfigured_car_no_speed_bin_fields(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    msg = est.get_msg()
    ltp = msg.lateralTorqueParameters
    assert len(ltp.speedBinCenters) == 0
    assert len(ltp.speedBinLatAccelFactors) == 0
    assert len(ltp.speedBinFrictions) == 0
    assert len(ltp.speedBinValid) == 0

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_unconfigured_car_global_params_still_work(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT, lat_accel_factor=2.0, friction=0.15))
    msg = est.get_msg()
    ltp = msg.lateralTorqueParameters
    assert ltp.latAccelFactorFiltered == pytest.approx(2.0, abs=1e-2)
    assert ltp.frictionCoefficientFiltered == pytest.approx(0.15, abs=1e-3)
    assert not est.speed_binned

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_unconfigured_car_no_speed_bin_attributes(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    assert not hasattr(est, 'speed_bin_points')
    assert not hasattr(est, 'speed_bin_filtered')

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cal_percent_works_for_both(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    fingerprints = [NON_SPEED_DEP_FINGERPRINT]
    if SPEED_DEP_FINGERPRINT:
      fingerprints.append(SPEED_DEP_FINGERPRINT)
    for fp in fingerprints:
      est = TorqueEstimator(make_mock_CP(fingerprint=fp))
      msg = est.get_msg()
      assert msg.lateralTorqueParameters.calPerc == 0


class TestCentersToBoumds:
  """Tests for _centers_to_bounds static method."""

  def test_midpoints_between_centers(self):
    bounds = TorqueEstimatorExt._centers_to_bounds([10.0, 20.0, 30.0])
    assert bounds[0] == (5, 15.0)   # lo=DEFAULT[0][0], hi=midpoint(10,20)
    assert bounds[1] == (15.0, 25.0)
    assert bounds[2] == (25.0, 40)  # hi=DEFAULT[-1][1]

  def test_single_center(self):
    bounds = TorqueEstimatorExt._centers_to_bounds([20.0])
    assert bounds == [(5, 40)]

  def test_edges_use_default_bounds(self):
    bounds = TorqueEstimatorExt._centers_to_bounds([7.0, 35.0])
    assert bounds[0][0] == 5    # DEFAULT_SPEED_BIN_BOUNDS[0][0]
    assert bounds[-1][1] == 40  # DEFAULT_SPEED_BIN_BOUNDS[-1][1]
    assert bounds[0][1] == pytest.approx((7.0 + 35.0) / 2)
    assert bounds[1][0] == pytest.approx((7.0 + 35.0) / 2)

  def test_contiguous_coverage(self):
    """Each bin's upper bound must equal the next bin's lower bound."""
    centers = [8.0, 15.0, 22.0, 30.0]
    bounds = TorqueEstimatorExt._centers_to_bounds(centers)
    for i in range(len(bounds) - 1):
      assert bounds[i][1] == pytest.approx(bounds[i + 1][0])


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestCacheRestore:
  """Tests for _restore_ext_cache."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_successful_restore_updates_filters(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    n_bins = len(est.speed_bin_bounds)
    cached_lafs = [float(i + 1) for i in range(n_bins)]
    cached_frictions = [float(i) * 0.01 for i in range(n_bins)]

    cache_ltp = MagicMock()
    cache_ltp.speedBinLatAccelFactors = cached_lafs
    cache_ltp.speedBinFrictions = cached_frictions
    cache_ltp.speedBinPoints = []  # wrong length, skipped

    est._restore_ext_cache(cache_ltp)

    for i in range(n_bins):
      assert est.speed_bin_filtered[i]['latAccelFactor'].x == pytest.approx(cached_lafs[i])
      assert est.speed_bin_filtered[i]['frictionCoefficient'].x == pytest.approx(cached_frictions[i])

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_mismatched_laf_length_rejected(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    original_laf = est.speed_bin_filtered[0]['latAccelFactor'].x

    cache_ltp = MagicMock()
    cache_ltp.speedBinLatAccelFactors = [999.0]  # wrong length
    cache_ltp.speedBinFrictions = [999.0]

    est._restore_ext_cache(cache_ltp)

    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(original_laf)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_mismatched_friction_length_rejected(self, mock_params_cls, mock_ext):
    """Both LAF and friction must match bin count; if only one matches, nothing is restored."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    n_bins = len(est.speed_bin_bounds)
    original_laf = est.speed_bin_filtered[0]['latAccelFactor'].x

    cache_ltp = MagicMock()
    cache_ltp.speedBinLatAccelFactors = [999.0] * n_bins  # correct length
    cache_ltp.speedBinFrictions = [999.0]  # wrong length

    est._restore_ext_cache(cache_ltp)

    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(original_laf)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_missing_points_still_restores_filters(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    n_bins = len(est.speed_bin_bounds)
    cached_lafs = [3.0 + i * 0.1 for i in range(n_bins)]
    cached_frictions = [0.2 + i * 0.01 for i in range(n_bins)]

    cache_ltp = MagicMock()
    cache_ltp.speedBinLatAccelFactors = cached_lafs
    cache_ltp.speedBinFrictions = cached_frictions
    cache_ltp.speedBinPoints = []  # empty — points not restored, but filters are

    est._restore_ext_cache(cache_ltp)

    for i in range(n_bins):
      assert est.speed_bin_filtered[i]['latAccelFactor'].x == pytest.approx(cached_lafs[i])
      assert est.speed_bin_filtered[i]['frictionCoefficient'].x == pytest.approx(cached_frictions[i])

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_getattr_decay_fallback(self, mock_params_cls, mock_ext):
    """_restore_ext_cache must work when self.decay is not set (init-time call order)."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    n_bins = len(est.speed_bin_bounds)
    saved_decay = est.decay
    del est.decay

    cache_ltp = MagicMock()
    cache_ltp.speedBinLatAccelFactors = [5.0] * n_bins
    cache_ltp.speedBinFrictions = [0.5] * n_bins
    cache_ltp.speedBinPoints = []

    est._restore_ext_cache(cache_ltp)

    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(5.0)
    assert all(d == MIN_FILTER_DECAY for d in est.speed_bin_decays)

    est.decay = saved_decay

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_uses_actual_decay_when_available(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    n_bins = len(est.speed_bin_bounds)
    est.decay = 200  # set a custom decay value

    cache_ltp = MagicMock()
    cache_ltp.speedBinLatAccelFactors = [5.0] * n_bins
    cache_ltp.speedBinFrictions = [0.5] * n_bins
    cache_ltp.speedBinPoints = []

    est._restore_ext_cache(cache_ltp)

    assert all(d == 200 for d in est.speed_bin_decays)


def _make_cache_bytes(version=None, centers=None, lafs=None, frictions=None, n_bins=None):
  """Serialized log.Event holding a lateralTorqueParameters cache blob."""
  if n_bins is None:
    n_bins = len(SPEED_BIN_CENTERS)
  evt = log.Event.new_message()
  ltp = evt.init('lateralTorqueParameters')
  ltp.version = VERSION if version is None else version
  ltp.speedBinCenters = [float(c) for c in (SPEED_BIN_CENTERS if centers is None else centers)]
  ltp.speedBinLatAccelFactors = [float(v) for v in ([2.0 + 0.1 * i for i in range(n_bins)] if lafs is None else lafs)]
  ltp.speedBinFrictions = [float(v) for v in ([0.2 + 0.01 * i for i in range(n_bins)] if frictions is None else frictions)]
  ltp.speedBinValid = [True] * n_bins
  return evt.to_bytes()


def _make_prev_cp_bytes(fingerprint=NON_SPEED_DEP_FINGERPRINT, lat_accel_factor=1.25, friction=0.125):
  """Serialized CarParams as stored in CarParamsPrevRoute."""
  cp = car.CarParams.new_message()
  cp.carFingerprint = fingerprint
  cp.lateralTuning.init('torque')
  cp.lateralTuning.torque.friction = friction
  cp.lateralTuning.torque.latAccelFactor = lat_accel_factor
  return cp.to_bytes()


class TestCacheRestoreKeyGate:
  """The Params-read restore path must honor the global learner's restore key.
  Uses an unconfigured car (default bins), so these run even with an empty TOML.
  Seed values are the offline 1.25/0.125 from make_mock_CP."""

  def _build_est(self, mock_ext, cache_bytes, cp_bytes):
    _setup_ext_mock(mock_ext, speed_dep_on=True)

    def _ext_get(param, **kwargs):
      if param == "LiveTorqueParameters":
        return cache_bytes
      if param == "CarParamsPrevRoute":
        return cp_bytes
      return None
    mock_ext.return_value.get.side_effect = _ext_get
    return TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_accepted_when_key_matches(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    n_bins = len(SPEED_BIN_CENTERS)
    lafs = [3.0 + 0.1 * i for i in range(n_bins)]
    frictions = [0.15 + 0.01 * i for i in range(n_bins)]
    est = self._build_est(mock_ext, _make_cache_bytes(lafs=lafs, frictions=frictions), _make_prev_cp_bytes())
    for i in range(n_bins):
      assert est.speed_bin_filtered[i]['latAccelFactor'].x == pytest.approx(lafs[i], abs=1e-4)
      assert est.speed_bin_filtered[i]['frictionCoefficient'].x == pytest.approx(frictions[i], abs=1e-4)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_rejected_on_version_mismatch(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    est = self._build_est(mock_ext, _make_cache_bytes(version=VERSION + 1), _make_prev_cp_bytes())
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(1.25)
    assert est.speed_bin_filtered[0]['frictionCoefficient'].x == pytest.approx(0.125)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_rejected_on_different_car(self, mock_params_cls, mock_ext):
    """Same default bin centers on every unconfigured car — the key must still reject."""
    mock_params_cls.return_value.get.return_value = None
    est = self._build_est(mock_ext, _make_cache_bytes(), _make_prev_cp_bytes(fingerprint='SOME_OTHER_CAR'))
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(1.25)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_rejected_on_changed_offline_values(self, mock_params_cls, mock_ext):
    """A torque-data update (new offline baseline) must invalidate the bins too."""
    mock_params_cls.return_value.get.return_value = None
    est = self._build_est(mock_ext, _make_cache_bytes(), _make_prev_cp_bytes(friction=0.2))
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(1.25)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_rejected_without_prev_carparams(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    est = self._build_est(mock_ext, _make_cache_bytes(), None)
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(1.25)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_direct_pass_bypasses_key_gate(self, mock_params_cls, mock_ext):
    """Passing cache_ltp directly is the documented test seam: no key check."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    n_bins = len(est.speed_bin_bounds)

    cache_ltp = MagicMock()
    cache_ltp.speedBinCenters = list(SPEED_BIN_CENTERS)
    cache_ltp.speedBinLatAccelFactors = [5.0] * n_bins
    cache_ltp.speedBinFrictions = [0.5] * n_bins
    cache_ltp.speedBinPoints = []

    est._restore_ext_cache(cache_ltp)

    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(5.0)


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestNaNHandling:
  """Tests for bin behavior when SVD fails."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_svd_failure_returns_false(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    target_bin = 1  # (8-12) range
    mock_bucket = MagicMock()
    mock_bucket.is_calculable.return_value = True
    mock_bucket.is_valid.return_value = False
    mock_bucket.get_points.return_value = np.zeros((10, 3))
    est.speed_bin_points[target_bin] = mock_bucket
    est._speed_bin_last_len[target_bin] = -1  # force recalculation

    with patch('numpy.linalg.svd', side_effect=np.linalg.LinAlgError):
      results = est._estimate_params_speed_binned()

    bin_results = dict(results)
    assert bin_results[target_bin] is False

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_valid_bin_svd_failure_resets_bin(self, mock_params_cls, mock_ext):
    """A bin with enough data that produces NaN/error should be reset."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    target_bin = 1
    mock_bucket = MagicMock()
    mock_bucket.is_calculable.return_value = True
    mock_bucket.is_valid.return_value = True  # enough data → triggers reset
    mock_bucket.get_points.return_value = np.zeros((10, 3))
    est.speed_bin_points[target_bin] = mock_bucket
    est._speed_bin_last_len[target_bin] = -1  # force recalculation

    with patch('numpy.linalg.svd', side_effect=np.linalg.LinAlgError):
      est._estimate_params_speed_binned()

    assert est.speed_bin_points[target_bin] is not mock_bucket
    assert isinstance(est.speed_bin_points[target_bin], TorqueBuckets)
    assert est.speed_bin_decays[target_bin] == MIN_FILTER_DECAY

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_non_valid_bin_svd_failure_preserves_bin(self, mock_params_cls, mock_ext):
    """A bin that is calculable but not valid should NOT be reset on SVD failure."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    target_bin = 1
    mock_bucket = MagicMock()
    mock_bucket.is_calculable.return_value = True
    mock_bucket.is_valid.return_value = False  # not enough data → no reset
    mock_bucket.get_points.return_value = np.zeros((10, 3))
    est.speed_bin_points[target_bin] = mock_bucket
    est._speed_bin_last_len[target_bin] = -1  # force recalculation

    with patch('numpy.linalg.svd', side_effect=np.linalg.LinAlgError):
      est._estimate_params_speed_binned()

    assert est.speed_bin_points[target_bin] is mock_bucket  # preserved


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestGetMsgWithPoints:
  """get_msg(with_points=True) should populate speedBinPoints."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_speed_bin_points_populated(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)
    est._on_torque_point(0.2, 0.4, 20.0)

    msg = est.get_msg(with_points=True)
    ltp = msg.lateralTorqueParameters
    assert len(ltp.speedBinPoints) == len(SPEED_BIN_BOUNDS)
    total_points = sum(len(bin_pts) for bin_pts in ltp.speedBinPoints)
    assert total_points >= 2

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_speed_bin_points_empty_without_flag(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est._on_torque_point(0.1, 0.3, 10.0)

    msg = est.get_msg(with_points=False)
    ltp = msg.lateralTorqueParameters
    assert len(ltp.speedBinPoints) == 0


class TestUnconfiguredCarToggleOn:
  """Unconfigured car with speed-dep ON should use default bins and offline seeds."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_default_bins_created(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    assert est.speed_binned
    est._on_torque_point(0.1, 0.3, 10.0)
    assert len(est.speed_bin_bounds) == len(SPEED_BIN_BOUNDS)
    assert est.speed_bin_centers == list(SPEED_BIN_CENTERS)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_seeded_with_offline_values(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT,
                                       lat_accel_factor=2.5, friction=0.18))
    est._on_torque_point(0.1, 0.3, 10.0)
    for i in range(len(SPEED_BIN_BOUNDS)):
      assert est.speed_bin_filtered[i]['latAccelFactor'].x == pytest.approx(2.5)
      assert est.speed_bin_filtered[i]['frictionCoefficient'].x == pytest.approx(0.18)


class TestOnTorquePointWhenOff:
  """_on_torque_point should be a no-op when speed_binned is False."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_no_bins_created_when_off(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    est._on_torque_point(0.1, 0.3, 10.0)
    assert not hasattr(est, 'speed_bin_points')


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestEnsureSpeedBinsIdempotency:
  """_ensure_speed_bins should not re-init on subsequent calls."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_second_call_preserves_points(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())

    est._on_torque_point(0.1, 0.3, 10.0)
    # Bin for 10.0 m/s (8-12 range, index 1) should have 1 point
    assert len(est.speed_bin_points[1]) == 1

    est._on_torque_point(0.2, 0.4, 10.0)
    # Should now have 2 points (not re-initialized)
    assert len(est.speed_bin_points[1]) == 2


# --- Moment-matrix learner (SpeedDependentTorqueMomentToggle) ---
# These use the unconfigured-car path on purpose: speed_dependent.toml carries no
# entries, so the default bins are what actually runs on the device.

def _make_moment_est(fingerprint=NON_SPEED_DEP_FINGERPRINT):
  return TorqueEstimator(make_mock_CP(fingerprint=fingerprint))


def _balanced_points(n_per_bucket=40, slope=3.0, offset=0.0, noise=0.0, seed=0):
  """Points spread evenly over the steer buckets, so inverse-density weights settle
  near 1 and the fit is comparable with an unweighted one."""
  rng = np.random.default_rng(seed)
  xs, ys = [], []
  for lo, hi in STEER_BUCKET_BOUNDS:
    for k in range(n_per_bucket):
      x = lo + (hi - lo) * (k + 0.5) / n_per_bucket
      xs.append(x)
      ys.append(slope * x + offset + (rng.normal(0, noise) if noise else 0.0))
  order = rng.permutation(len(xs))
  return [(xs[i], ys[i]) for i in order]


class TestMomentToggleGate:
  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_off_by_default_uses_point_store(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=False)
    est = _make_moment_est()
    assert est.speed_binned and not est.moment_learner
    assert len(est.speed_bin_points) == len(SPEED_BIN_BOUNDS)
    assert est.moment_bank is None

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_on_uses_moment_store(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    assert est.moment_learner
    assert est.moment_bank is not None
    assert est.moment_bank.n == len(est.speed_bin_bounds)
    assert est.speed_bin_points == []

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cannot_arm_without_speed_dep(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False, enforce_on=True, moment_on=True)
    est = _make_moment_est()
    assert not est.speed_binned
    assert not est.moment_learner


class TestMomentAlgebra:
  """The moment fit must be upstream's TLS, just evaluated on the moments."""

  def test_matches_upstream_svd(self):
    pts = _balanced_points(slope=2.8, offset=0.05, noise=0.03, seed=3)
    A = np.array([[x, 1.0, y] for x, y in pts])

    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    mom.M = A.T @ A                     # uniform weights, bypassing the density term
    mom.S = float(len(A))
    slope, friction = mom.fit(FRICTION_FACTOR)

    _, _, v = np.linalg.svd(A, full_matrices=False)
    exp_slope = float((-v.T[0:2, 2] / v.T[2, 2])[0])
    _, spread = np.matmul(A[:, [0, 2]], slope2rot(exp_slope)).T
    exp_friction = float(np.std(spread) * FRICTION_FACTOR)

    assert slope == pytest.approx(exp_slope, rel=1e-6)
    assert friction == pytest.approx(exp_friction, rel=1e-6)

  def test_recovers_known_slope_through_add(self):
    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    for x, y in _balanced_points(slope=3.1, noise=0.02, seed=5):
      mom.add(x, y, 1.0)
    assert mom.is_valid()
    slope, friction = mom.fit(FRICTION_FACTOR)
    assert slope == pytest.approx(3.1, abs=0.05)
    assert 0.0 <= friction < 0.2

  def test_not_identifiable_without_steer_spread(self):
    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    for _ in range(500):
      mom.add(0.05, 0.15, 1.0)          # every sample at the same steer
    assert mom.S >= MOMENT_MIN_ESS
    assert mom.x_std() < MOMENT_MIN_XSTD
    assert not mom.is_valid()

  def test_ess_cap_bounds_a_single_bad_sample(self):
    """The point of the cap: no one observation can move the fit much."""
    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    while mom.S < MOMENT_ESS_CAP:
      for x, y in _balanced_points(slope=3.0, noise=0.02, seed=7):
        mom.add(x, y, 1.0)
    before = mom.fit(FRICTION_FACTOR)[0]
    mom.add(-0.5, 1.0, 1.0)             # worst case inside torqued's accept envelope
    after = mom.fit(FRICTION_FACTOR)[0]
    assert abs(after - before) < 0.05, f"single sample moved slope by {after - before:.4f}"

  def test_density_weight_is_bounded(self):
    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    for _ in range(2000):
      mom.add(0.05, 0.15, 1.0)               # hammer one bucket
    rare = mom._density_weight(-0.45, 1.0)   # a starved bucket
    common = mom._density_weight(0.05, 1.0)
    assert rare <= MOMENT_DENSITY_CEILING
    assert common >= MOMENT_DENSITY_FLOOR
    assert rare > common


class TestMomentKernelRouting:
  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_point_feeds_neighbouring_bins(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    target = len(est.speed_bin_centers) // 2
    vego = est.speed_bin_centers[target]
    est._on_torque_point(0.1, 0.3, vego)

    weights = est.moment_bank.S
    assert weights[target] > 0
    assert weights[target] == max(weights), "nearest bin must get the most weight"
    assert weights[target + 1] > 0, "neighbour above must be informed too"
    assert weights[0] == 0, "a far bin must not be touched"

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_point_store_untouched_in_moment_mode(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    est._on_torque_point(0.1, 0.3, 32.0)
    assert est.speed_bin_points == []


class TestMomentMessageAndCache:
  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cereal_fields_populated(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    est._on_torque_point(0.1, 0.3, 20.0)
    n = len(est.speed_bin_bounds)
    ltp = est.get_msg().lateralTorqueParameters
    assert len(ltp.speedBinCenters) == n
    assert len(ltp.speedBinLatAccelFactors) == n
    assert len(ltp.speedBinValid) == n

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cache_rows_are_moment_shaped(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    for x, y in _balanced_points(n_per_bucket=10, seed=11):
      est._on_torque_point(x, y, 32.0)
    ltp = est.get_msg(with_points=True).lateralTorqueParameters
    assert len(ltp.speedBinPoints) == len(est.speed_bin_bounds)
    for rows in ltp.speedBinPoints:
      assert len(rows) == 1
      assert len(rows[0]) == MOMENT_CACHE_ROW

  def test_cache_roundtrip_preserves_fit(self):
    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    for x, y in _balanced_points(slope=2.9, noise=0.02, seed=13):
      mom.add(x, y, 1.0)
    row = mom.to_cache()

    restored = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    assert restored.load_cache(row)
    assert restored.S == pytest.approx(mom.S)
    assert restored.S_in == pytest.approx(mom.S_in)
    assert restored.fit(FRICTION_FACTOR)[0] == pytest.approx(mom.fit(FRICTION_FACTOR)[0], rel=1e-9)
    assert np.allclose(restored.dens, mom.dens)

  def test_cache_rejects_wrong_shape_and_garbage(self):
    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    assert not mom.load_cache([0.1, 0.2])                                  # a point row
    assert not mom.load_cache([0.0] * MOMENT_CACHE_ROW)                    # zero weight
    over = [1.0] * MOMENT_CACHE_ROW
    over[6] = MOMENT_ESS_CAP * 2                                           # weight above the cap
    assert not mom.load_cache(over)
    filled = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    for x, y in _balanced_points(n_per_bucket=5, seed=17):
      filled.add(x, y, 1.0)
    row = filled.to_cache()
    row[0] = float('nan')
    assert not mom.load_cache(row)
    inconsistent = filled.to_cache()
    inconsistent[7] = inconsistent[6] * 2                                  # in-bin weight above total
    assert not mom.load_cache(inconsistent)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_ignores_point_cache_in_moment_mode(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    n = len(est.speed_bin_bounds)

    ltp = MagicMock()
    ltp.speedBinCenters = list(est.speed_bin_centers)   # must match, or restore bails early
    ltp.speedBinLatAccelFactors = [2.5] * n
    ltp.speedBinFrictions = [0.1] * n
    ltp.speedBinPoints = [[[0.1, 0.3], [0.2, 0.4]] for _ in range(n)]   # point-store cache

    est._restore_ext_cache(cache_ltp=ltp)
    assert (est.moment_bank.S == 0.0).all(), "point cache must not feed moments"
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(2.5)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_ignores_moment_cache_in_point_mode(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=False)
    est = _make_moment_est()
    n = len(est.speed_bin_bounds)

    filled = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    for x, y in _balanced_points(n_per_bucket=5, seed=19):
      filled.add(x, y, 1.0)

    ltp = MagicMock()
    ltp.speedBinCenters = list(est.speed_bin_centers)   # must match, or restore bails early
    ltp.speedBinLatAccelFactors = [2.5] * n
    ltp.speedBinFrictions = [0.1] * n
    ltp.speedBinPoints = [[filled.to_cache()] for _ in range(n)]        # moment cache

    est._restore_ext_cache(cache_ltp=ltp)
    # restore must have run (filters took the cached values) but dropped the rows
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(2.5)
    assert all(len(b) == 0 for b in est.speed_bin_points), "moment cache must not feed point buckets"


class TestMomentSanityClip:
  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_fit_clipped_to_bin_bounds(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    # narrow the sanity window so a plausible fit is forced to clip
    est.speed_bin_lat_accel_factor_bounds = [(2.0, 2.2)] * len(est.speed_bin_bounds)
    est.speed_bin_friction_bounds = [(0.05, 0.15)] * len(est.speed_bin_bounds)

    for x, y in _balanced_points(slope=3.5, noise=0.01, seed=23):
      est._on_torque_point(x, y, 32.0)
    # the output filter ramps toward the clipped fit; iterate until converged
    for _ in range(400):
      results = est._estimate_params_speed_binned()

    fitted = [i for i, valid in results if valid]
    assert fitted, "feeding at 32 m/s must make that bin fit"
    for i in fitted:
      assert 2.0 <= est.speed_bin_filtered[i]['latAccelFactor'].x <= 2.2
      assert 0.05 <= est.speed_bin_filtered[i]['frictionCoefficient'].x <= 0.15

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_lat_accel_factor_floored_above_zero(self, mock_params_cls, mock_ext):
    """Relaxed sanity gives a 0.0 lower clip bound, but latAccelFactor is a divisor
    in the controller: the moment path must never converge to it."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    n = len(est.speed_bin_bounds)
    est.speed_bin_lat_accel_factor_bounds = [(0.0, 4.0)] * n   # relaxed-shaped window

    # degenerate data: negative slope, so the raw clip target is the 0.0 bound
    for x, y in _balanced_points(slope=-1.0, noise=0.01, seed=37):
      est._on_torque_point(x, y, 32.0)
    for _ in range(400):
      est._estimate_params_speed_binned()

    for i in range(n):
      if est.moment_bank.S[i] > 0.0:
        assert est.speed_bin_filtered[i]['latAccelFactor'].x >= MOMENT_MIN_LAT_ACCEL_FACTOR - 1e-6

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_first_fit_ramps_instead_of_stepping(self, mock_params_cls, mock_ext):
    """One estimate pass must move the applied value only a filter step toward the
    fit, not jump to it — the transition the point path's low-pass also smooths."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    n = len(est.speed_bin_bounds)
    est.speed_bin_lat_accel_factor_bounds = [(1.0, 4.0)] * n
    target = int(np.argmin(np.abs(np.asarray(est.speed_bin_centers) - 32.0)))
    seed = est.speed_bin_filtered[target]['latAccelFactor'].x

    for x, y in _balanced_points(slope=3.5, noise=0.01, seed=41):
      est._on_torque_point(x, y, 32.0)
    est._estimate_params_speed_binned()
    after_one = est.speed_bin_filtered[target]['latAccelFactor'].x
    assert abs(after_one - seed) < 0.2 * abs(3.5 - seed), "single pass must not step to the fit"

    for _ in range(400):
      est._estimate_params_speed_binned()
    assert est.speed_bin_filtered[target]['latAccelFactor'].x == pytest.approx(3.5, abs=0.05)


class TestMomentSampleGates:
  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_below_min_speed_points_rejected(self, mock_params_cls, mock_ext):
    """Parking/creep points reach the hook (upstream has no lower speed gate there)
    but must not feed any bin — the point store drops them implicitly."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    for vego in (0.0, 2.0, 4.99):
      est._on_torque_point(0.1, 0.3, vego)
    assert (est.moment_bank.S == 0.0).all()
    est._on_torque_point(0.1, 0.3, est.speed_bin_bounds[0][0])
    assert est.moment_bank.S[0] > 0.0

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_neighbour_only_data_cannot_validate(self, mock_params_cls, mock_ext):
    """A bin's fitted value may be neighbour-informed, but validity needs at least
    some data from the bin's own speed range."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    # feed at one bin's center; a bin two nodes up only sees it through the kernel
    in_idx = next(i for i, (lo, hi) in enumerate(est.speed_bin_bounds) if lo <= 26.0 < hi)
    target = in_idx + 2
    for _ in range(30):
      for x, y in _balanced_points(n_per_bucket=5, slope=3.0, noise=0.02, seed=29):
        est._on_torque_point(x, y, 26.0)
    valid = dict(est._estimate_params_speed_binned())
    assert est.moment_bank.S[target] >= MOMENT_MIN_ESS, "kernel must still feed the neighbour bin"
    assert est.moment_bank.S_in[target] == 0.0
    assert not valid[target]
    assert valid[in_idx]

  def test_small_steer_data_cannot_validate(self):
    """Cruise-only steer (|x| < 0.15) passes the spread check but must fail inner
    steer-bucket coverage — point mode would never validate on it either."""
    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    rng = np.random.default_rng(31)
    for _ in range(500):
      x = float(rng.uniform(-0.15, 0.15))
      mom.add(x, 3.0 * x + float(rng.normal(0, 0.02)), 1.0)
    assert mom.S >= MOMENT_MIN_ESS and mom.S_in >= MOMENT_MIN_IN_BIN_ESS
    assert mom.x_std() >= MOMENT_MIN_XSTD
    assert not mom.is_valid()


class TestMomentDefaultGrid:
  """Moment mode defaults to the uniform 5-mph grid; point mode keeps the coarse
  bins (it would starve on ~1/15 of the data per bin); a TOML speed_bp wins over
  both."""

  def test_grid_is_5mph_lattice(self):
    mph = np.asarray(MOMENT_SPEED_BIN_CENTERS) / 0.44704
    assert len(MOMENT_SPEED_BIN_CENTERS) == 15
    assert np.allclose(mph, np.arange(15, 90, 5), atol=0.01)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_moment_mode_uses_grid(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    est = _make_moment_est()
    assert est.speed_bin_centers == MOMENT_SPEED_BIN_CENTERS
    assert est.speed_bin_bounds[0][0] == 5
    assert est.speed_bin_bounds[-1][1] == 40
    for (_, hi), (lo, _) in zip(est.speed_bin_bounds[:-1], est.speed_bin_bounds[1:], strict=True):
      assert hi == pytest.approx(lo)
    n = len(est.speed_bin_bounds)
    assert len(est.speed_bin_filtered) == n
    assert len(est.speed_bin_lat_accel_factor_bounds) == n
    assert len(est.speed_bin_friction_bounds) == n

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_point_mode_keeps_coarse_bins(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=False)
    est = _make_moment_est()
    assert est.speed_bin_centers == list(SPEED_BIN_CENTERS)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_toml_speed_bp_wins_over_grid(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True, moment_on=True)
    with patch('opendbc.sunnypilot.car.interfaces.get_speed_dep_config',
               return_value={'TOML_CAR': {'speed_bp': [10.0, 20.0, 30.0]}}):
      est = TorqueEstimator(make_mock_CP(fingerprint='TOML_CAR'))
    assert est.speed_bin_centers == [10.0, 20.0, 30.0]
    assert est.moment_bank.n == 3


def _oracle_feed(moments, centers, bounds, steer, la, vego):
  """The pre-bank production routing, verbatim: per-bin SpeedBinMoment updates
  with the loop-skip kernel cutoff. The bank must reproduce this exactly."""
  if not (bounds[0][0] <= vego < bounds[-1][1]):
    return
  for center, (lo, hi), mom in zip(centers, bounds, moments, strict=True):
    k = math.exp(-0.5 * ((vego - center) / MOMENT_SPEED_KERNEL_H) ** 2)
    if k >= MOMENT_KERNEL_MIN:
      mom.add(steer, la, k, in_bin=lo <= vego < hi)


class TestBankMatchesReference:
  """SpeedBinMomentBank must reproduce the per-bin SpeedBinMoment loop: same
  state, same fits, same validity, over a mixed random stream that includes
  out-of-range speeds and steers and ESS-cap saturation."""

  def _run_pair(self, centers, ess_cap, n_pts=4000, seed=101):
    centers = list(centers)
    bounds = TorqueEstimatorExt._centers_to_bounds(centers)
    moments = [SpeedBinMoment(STEER_BUCKET_BOUNDS, ess_cap=ess_cap) for _ in centers]
    bank = SpeedBinMomentBank(centers, bounds, STEER_BUCKET_BOUNDS, ess_cap=ess_cap)
    rng = np.random.default_rng(seed)
    steers = rng.uniform(-0.6, 0.6, n_pts)     # some outside the tracked steer range
    vegos = rng.uniform(2.0, 42.0, n_pts)      # some below/above the bin range
    las = 2.6 * steers + rng.normal(0.0, 0.1, n_pts)
    for s, la, v in zip(steers, las, vegos, strict=True):
      _oracle_feed(moments, centers, bounds, float(s), float(la), float(v))
      bank.add(float(s), float(la), float(v))
    return moments, bank

  def _assert_state_parity(self, moments, bank):
    for i, mom in enumerate(moments):
      assert np.allclose(bank.M[i], mom.M, rtol=1e-9, atol=1e-12), f"M mismatch bin {i}"
      assert bank.S[i] == pytest.approx(mom.S, rel=1e-9, abs=1e-12)
      assert bank.S_in[i] == pytest.approx(mom.S_in, rel=1e-9, abs=1e-12)
      assert np.allclose(bank.dens[i], mom.dens, rtol=1e-9, atol=1e-12), f"dens mismatch bin {i}"

  @pytest.mark.parametrize("centers", [SPEED_BIN_CENTERS, MOMENT_SPEED_BIN_CENTERS])
  def test_state_and_fit_parity(self, centers):
    moments, bank = self._run_pair(centers, MOMENT_ESS_CAP)
    self._assert_state_parity(moments, bank)
    slopes, frictions, ok = bank.fit(FRICTION_FACTOR)
    valid = bank.is_valid()
    xs = bank.x_std()
    for i, mom in enumerate(moments):
      ref = mom.fit(FRICTION_FACTOR) if mom.S > 0.0 else None
      assert bool(ok[i]) == (ref is not None), f"fit-ok mismatch bin {i}"
      if ref is not None:
        assert slopes[i] == pytest.approx(ref[0], rel=1e-9, abs=1e-9)
        assert frictions[i] == pytest.approx(ref[1], rel=1e-9, abs=1e-9)
      assert bool(valid[i]) == mom.is_valid(), f"validity mismatch bin {i}"
      assert xs[i] == pytest.approx(mom.x_std(), rel=1e-9, abs=1e-12)

  def test_parity_through_cap_saturation(self):
    # a tiny cap exercises the forgetting rescale on nearly every point
    moments, bank = self._run_pair(MOMENT_SPEED_BIN_CENTERS, ess_cap=50.0, n_pts=1500, seed=7)
    assert (bank.S >= 49.0).any(), "cap must actually be reached for this test to bite"
    self._assert_state_parity(moments, bank)

  def test_cache_rows_interchangeable(self):
    moments, bank = self._run_pair(SPEED_BIN_CENTERS, MOMENT_ESS_CAP, n_pts=1500, seed=13)
    bounds = TorqueEstimatorExt._centers_to_bounds(list(SPEED_BIN_CENTERS))
    bank2 = SpeedBinMomentBank(SPEED_BIN_CENTERS, bounds, STEER_BUCKET_BOUNDS)
    for i, mom in enumerate(moments):
      if bank.S[i] <= 0.0:
        continue
      restored = SpeedBinMoment(STEER_BUCKET_BOUNDS)
      assert restored.load_cache(bank.to_cache(i))
      assert restored.fit(FRICTION_FACTOR)[0] == pytest.approx(mom.fit(FRICTION_FACTOR)[0], rel=1e-9)
      assert bank2.load_cache(i, mom.to_cache())
      assert bank2.S[i] == pytest.approx(mom.S)
    assert not bank2.load_cache(0, [0.1, 0.2])                # a point row
    assert not bank2.load_cache(0, [0.0] * MOMENT_CACHE_ROW)  # zero weight

  def test_reset_bin_clears_only_that_row(self):
    moments, bank = self._run_pair(SPEED_BIN_CENTERS, MOMENT_ESS_CAP, n_pts=800, seed=17)
    before = bank.S.copy()
    bank.reset_bin(2)
    assert bank.S[2] == 0.0 and bank.S_in[2] == 0.0
    assert not bank.M[2].any() and not bank.dens[2].any()
    others = [i for i in range(bank.n) if i != 2]
    assert np.allclose(bank.S[others], before[others])
