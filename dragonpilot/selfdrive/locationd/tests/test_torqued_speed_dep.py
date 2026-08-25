"""Tests for the speed-dependent moment torque learner (dp port).

Ported from the sunnypilot-based fork's test_torqued_speed_dep.py, trimmed to
the moment-only design behind the single dp_lat_torqued_sd toggle.
"""
import math

import numpy as np
import pytest

from unittest.mock import MagicMock, patch  # noqa: TID251
from cereal import car, log
from openpilot.selfdrive.locationd.torqued import (
  TorqueEstimator, VERSION, STEER_BUCKET_BOUNDS, FRICTION_FACTOR, slope2rot,
)
from dragonpilot.selfdrive.locationd.torqued_ext import (
  TorqueEstimatorExt, SpeedBinMoment, SpeedBinMomentBank, MOMENT_SPEED_BIN_CENTERS,
  MOMENT_ESS_CAP, MOMENT_MIN_ESS, MOMENT_MIN_XSTD, MOMENT_CACHE_ROW,
  MOMENT_DENSITY_CEILING, MOMENT_DENSITY_FLOOR, MOMENT_SPEED_KERNEL_H, MOMENT_KERNEL_MIN,
  MOMENT_MIN_IN_BIN_ESS, MOMENT_MIN_LAT_ACCEL_FACTOR, SPEED_BIN_MIN, SPEED_BIN_MAX,
  MOMENT_DENSITY_CEILING_HI, MOMENT_DENSITY_CEILING_V, moment_density_ceiling,
)

# Both Params locations need mocking: torqued.py (cache) and torqued_ext.py (toggle)
PATCH_PARAMS = 'openpilot.selfdrive.locationd.torqued.Params'
PATCH_EXT_PARAMS = 'dragonpilot.selfdrive.locationd.torqued_ext.Params'

TEST_FINGERPRINT = 'TEST_CAR_FINGERPRINT'

# A non-uniform coarse grid, used to exercise the bank against a second layout
COARSE_CENTERS = [6.5, 10.0, 15.0, 21.0, 26.5, 32.0, 37.5]


def _setup_ext_mock(mock_ext_params_cls, on):
  """Configure the torqued_ext Params mock for the single toggle."""
  mock_ext_params_cls.return_value.get_bool.side_effect = \
    lambda param: on if param == "dp_lat_torqued_sd" else False
  mock_ext_params_cls.return_value.get.return_value = None


def make_mock_CP(fingerprint=TEST_FINGERPRINT, lat_accel_factor=1.25, friction=0.125, tuning='torque'):
  CP = MagicMock()
  CP.brand = 'toyota'
  CP.carFingerprint = fingerprint
  CP.lateralTuning.which.return_value = tuning
  CP.lateralTuning.torque.friction = friction
  CP.lateralTuning.torque.latAccelFactor = lat_accel_factor
  return CP


def _make_est(**kwargs):
  return TorqueEstimator(make_mock_CP(**kwargs))


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


class TestDefaults:
  def test_grid_is_5mph_lattice(self):
    mph = np.asarray(MOMENT_SPEED_BIN_CENTERS) / 0.44704
    assert len(MOMENT_SPEED_BIN_CENTERS) == 15
    assert np.allclose(mph, np.arange(15, 90, 5), atol=0.01)

  def test_version_exists(self):
    assert VERSION >= 1

  def test_grid_centers_inside_envelope(self):
    assert all(SPEED_BIN_MIN <= c < SPEED_BIN_MAX for c in MOMENT_SPEED_BIN_CENTERS)


class TestToggleGate:
  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_toggle_off_no_speed_bins(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=False)
    est = _make_est()
    assert not est.speed_binned
    assert not hasattr(est, 'moment_bank')
    assert not hasattr(est, 'speed_bin_filtered')

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_non_torque_tuning_cannot_arm(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est(tuning='pid')
    assert not est.speed_binned

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_toggle_on_uses_moment_grid(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
    assert est.speed_binned
    assert est.speed_bin_centers == MOMENT_SPEED_BIN_CENTERS
    assert est.moment_bank is not None
    assert est.moment_bank.n == len(est.speed_bin_bounds)
    assert est.speed_bin_bounds[0][0] == SPEED_BIN_MIN
    assert est.speed_bin_bounds[-1][1] == SPEED_BIN_MAX
    for (_, hi), (lo, _) in zip(est.speed_bin_bounds[:-1], est.speed_bin_bounds[1:], strict=True):
      assert hi == pytest.approx(lo)
    n = len(est.speed_bin_bounds)
    assert len(est.speed_bin_filtered) == n
    assert len(est.speed_bin_lat_accel_factor_bounds) == n
    assert len(est.speed_bin_friction_bounds) == n

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_on_torque_point_noop_when_off(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=False)
    est = _make_est()
    est._on_torque_point(0.1, 0.3, 10.0)
    assert not hasattr(est, 'moment_bank')

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_car_config_speed_bp_wins_over_grid(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    with patch.dict('dragonpilot.selfdrive.locationd.torqued_ext.SPEED_DEP_CAR_CONFIG',
                    {'CFG_CAR': {'speed_bp': [10.0, 20.0, 30.0]}}):
      est = _make_est(fingerprint='CFG_CAR')
    assert est.speed_bin_centers == [10.0, 20.0, 30.0]
    assert est.moment_bank.n == 3

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_seeded_with_offline_values(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est(lat_accel_factor=2.5, friction=0.18)
    for i in range(len(est.speed_bin_bounds)):
      assert est.speed_bin_filtered[i]['latAccelFactor'].x == pytest.approx(2.5)
      assert est.speed_bin_filtered[i]['frictionCoefficient'].x == pytest.approx(0.18)


class TestBackwardCompatibility:
  """The global learner and its message must be unaffected in either toggle state."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_off_no_speed_bin_fields(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=False)
    est = _make_est()
    ltp = est.get_msg().liveTorqueParameters
    assert len(ltp.speedBinCenters) == 0
    assert len(ltp.speedBinLatAccelFactors) == 0
    assert len(ltp.speedBinFrictions) == 0
    assert len(ltp.speedBinValid) == 0
    assert len(ltp.speedBinPoints) == 0

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_off_global_params_still_work(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=False)
    est = _make_est(lat_accel_factor=2.0, friction=0.15)
    ltp = est.get_msg().liveTorqueParameters
    assert ltp.latAccelFactorFiltered == pytest.approx(2.0, abs=1e-2)
    assert ltp.frictionCoefficientFiltered == pytest.approx(0.15, abs=1e-3)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_on_global_fit_unchanged(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est(lat_accel_factor=1.25, friction=0.125)
    ltp = est.get_msg().liveTorqueParameters
    assert ltp.latAccelFactorFiltered == pytest.approx(1.25, abs=1e-2)
    assert ltp.frictionCoefficientFiltered == pytest.approx(0.125, abs=1e-3)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_on_torque_point_does_not_feed_global_buckets(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
    est._on_torque_point(0.1, 0.3, 10.0)
    assert len(est.filtered_points) == 0

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cal_percent_works_for_both(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    for on in (False, True):
      _setup_ext_mock(mock_ext, on=on)
      est = _make_est()
      assert est.get_msg().liveTorqueParameters.calPerc == 0


class TestCentersToBounds:
  """Tests for _centers_to_bounds static method."""

  def test_midpoints_between_centers(self):
    bounds = TorqueEstimatorExt._centers_to_bounds([10.0, 20.0, 30.0])
    assert bounds[0] == (SPEED_BIN_MIN, 15.0)
    assert bounds[1] == (15.0, 25.0)
    assert bounds[2] == (25.0, SPEED_BIN_MAX)

  def test_single_center(self):
    bounds = TorqueEstimatorExt._centers_to_bounds([20.0])
    assert bounds == [(SPEED_BIN_MIN, SPEED_BIN_MAX)]

  def test_edges_use_envelope(self):
    bounds = TorqueEstimatorExt._centers_to_bounds([7.0, 35.0])
    assert bounds[0][0] == SPEED_BIN_MIN
    assert bounds[-1][1] == SPEED_BIN_MAX
    assert bounds[0][1] == pytest.approx((7.0 + 35.0) / 2)
    assert bounds[1][0] == pytest.approx((7.0 + 35.0) / 2)

  def test_contiguous_coverage(self):
    centers = [8.0, 15.0, 22.0, 30.0]
    bounds = TorqueEstimatorExt._centers_to_bounds(centers)
    for i in range(len(bounds) - 1):
      assert bounds[i][1] == pytest.approx(bounds[i + 1][0])


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
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
    target = len(est.speed_bin_centers) // 2
    vego = est.speed_bin_centers[target]
    est._on_torque_point(0.1, 0.3, vego)

    weights = est.moment_bank.S
    assert weights[target] > 0
    assert weights[target] == max(weights), "nearest bin must get the most weight"
    assert weights[target + 1] > 0, "neighbour above must be informed too"
    assert weights[0] == 0, "a far bin must not be touched"


class TestMomentMessageAndCache:
  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cereal_fields_populated(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
    est._on_torque_point(0.1, 0.3, 20.0)
    n = len(est.speed_bin_bounds)
    ltp = est.get_msg().liveTorqueParameters
    assert len(ltp.speedBinCenters) == n
    assert len(ltp.speedBinLatAccelFactors) == n
    assert len(ltp.speedBinFrictions) == n
    assert len(ltp.speedBinValid) == n
    assert len(ltp.speedBinPoints) == 0  # only on cache writes

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cache_rows_are_moment_shaped(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
    for x, y in _balanced_points(n_per_bucket=10, seed=11):
      est._on_torque_point(x, y, 32.0)
    ltp = est.get_msg(with_points=True).liveTorqueParameters
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
  def test_restore_ignores_point_shaped_rows(self, mock_params_cls, mock_ext):
    """A point-store cache (e.g. carried over from a sunnypilot install) must not
    feed the moment bank, but the filter values still restore."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
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
  def test_restore_loads_moment_rows(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    donor = _make_est()
    for x, y in _balanced_points(n_per_bucket=10, seed=19):
      donor._on_torque_point(x, y, 20.0)
    n = len(donor.speed_bin_bounds)

    ltp = MagicMock()
    ltp.speedBinCenters = list(donor.speed_bin_centers)
    ltp.speedBinLatAccelFactors = [2.5] * n
    ltp.speedBinFrictions = [0.1] * n
    ltp.speedBinPoints = [[donor.moment_bank.to_cache(i)] for i in range(n)]

    est = _make_est()
    est._restore_ext_cache(cache_ltp=ltp)
    assert np.allclose(est.moment_bank.S, donor.moment_bank.S)
    assert np.allclose(est.moment_bank.M, donor.moment_bank.M)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_rejected_on_center_mismatch(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
    n = len(est.speed_bin_bounds)
    original = est.speed_bin_filtered[0]['latAccelFactor'].x

    ltp = MagicMock()
    ltp.speedBinCenters = [c + 1.0 for c in est.speed_bin_centers]
    ltp.speedBinLatAccelFactors = [9.0] * n
    ltp.speedBinFrictions = [0.9] * n
    ltp.speedBinPoints = []

    est._restore_ext_cache(cache_ltp=ltp)
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(original)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_rejected_on_length_mismatch(self, mock_params_cls, mock_ext):
    """Both LAF and friction must match bin count; if only one matches, nothing is restored."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
    n = len(est.speed_bin_bounds)
    original = est.speed_bin_filtered[0]['latAccelFactor'].x

    ltp = MagicMock()
    ltp.speedBinCenters = list(est.speed_bin_centers)
    ltp.speedBinLatAccelFactors = [999.0] * n
    ltp.speedBinFrictions = [999.0]   # wrong length
    ltp.speedBinPoints = []

    est._restore_ext_cache(cache_ltp=ltp)
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == pytest.approx(original)


def _make_cache_bytes(version=None, centers=None, lafs=None, frictions=None):
  """Serialized log.Event holding a liveTorqueParameters cache blob."""
  centers = list(MOMENT_SPEED_BIN_CENTERS if centers is None else centers)
  n_bins = len(centers)
  evt = log.Event.new_message()
  ltp = evt.init('liveTorqueParameters')
  ltp.version = VERSION if version is None else version
  ltp.speedBinCenters = [float(c) for c in centers]
  ltp.speedBinLatAccelFactors = [float(v) for v in ([2.0 + 0.1 * i for i in range(n_bins)] if lafs is None else lafs)]
  ltp.speedBinFrictions = [float(v) for v in ([0.2 + 0.01 * i for i in range(n_bins)] if frictions is None else frictions)]
  ltp.speedBinValid = [True] * n_bins
  return evt.to_bytes()


def _make_prev_cp_bytes(fingerprint=TEST_FINGERPRINT, lat_accel_factor=1.25, friction=0.125):
  """Serialized CarParams as stored in CarParamsPrevRoute."""
  cp = car.CarParams.new_message()
  cp.carFingerprint = fingerprint
  cp.lateralTuning.init('torque')
  cp.lateralTuning.torque.friction = friction
  cp.lateralTuning.torque.latAccelFactor = lat_accel_factor
  return cp.to_bytes()


class TestCacheRestoreKeyGate:
  """The Params-read restore path must honor the global learner's restore key.
  Seed values are the offline 1.25/0.125 from make_mock_CP."""

  def _build_est(self, mock_ext, cache_bytes, cp_bytes):
    _setup_ext_mock(mock_ext, on=True)

    def _ext_get(param, **kwargs):
      if param == "LiveTorqueParameters":
        return cache_bytes
      if param == "CarParamsPrevRoute":
        return cp_bytes
      return None
    mock_ext.return_value.get.side_effect = _ext_get
    return _make_est()

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_restore_accepted_when_key_matches(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    n_bins = len(MOMENT_SPEED_BIN_CENTERS)
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


class TestMomentSanityClip:
  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_fit_clipped_to_bin_bounds(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
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
    """The relaxed sanity window gives a 0.0 lower clip bound, but latAccelFactor
    is a divisor in the controller: the moment path must never converge to it."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
    n = len(est.speed_bin_bounds)
    assert all(lo == pytest.approx(0.0) for lo, _ in est.speed_bin_lat_accel_factor_bounds), \
      "relaxed sanity must produce a 0.0 lower bound for this test to bite"

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
    fit, not jump to it."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
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
    """Parking/creep points reach the hook (torqued has no lower speed gate there)
    but must not feed any bin."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
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
    _setup_ext_mock(mock_ext, on=True)
    est = _make_est()
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
    steer-bucket coverage."""
    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS)
    rng = np.random.default_rng(31)
    for _ in range(500):
      x = float(rng.uniform(-0.15, 0.15))
      mom.add(x, 3.0 * x + float(rng.normal(0, 0.02)), 1.0)
    assert mom.S >= MOMENT_MIN_ESS and mom.S_in >= MOMENT_MIN_IN_BIN_ESS
    assert mom.x_std() >= MOMENT_MIN_XSTD
    assert not mom.is_valid()


def _oracle_feed(moments, centers, bounds, steer, la, vego):
  """The pre-bank production routing, verbatim: per-bin SpeedBinMoment updates
  with the loop-skip kernel cutoff. The bank must reproduce this exactly."""
  if not (bounds[0][0] <= vego < bounds[-1][1]):
    return
  for center, (lo, hi), mom in zip(centers, bounds, moments, strict=True):
    k = math.exp(-0.5 * ((vego - center) / MOMENT_SPEED_KERNEL_H) ** 2)
    if k >= MOMENT_KERNEL_MIN:
      mom.add(steer, la, k, in_bin=lo <= vego < hi)


class TestDensityCeilingRamp:
  """The inverse-density ceiling ramps with speed so highway bins get the big-move
  leverage the road cannot supply on its own, while city bins keep exactly the
  weighting they have today."""

  def test_anchors_and_clamping(self):
    lo_v, hi_v = MOMENT_DENSITY_CEILING_V
    assert moment_density_ceiling([lo_v])[0] == pytest.approx(MOMENT_DENSITY_CEILING)
    assert moment_density_ceiling([hi_v])[0] == pytest.approx(MOMENT_DENSITY_CEILING_HI)
    # np.interp clamps, so a car-config grid outside the anchors never extrapolates
    assert moment_density_ceiling([0.0])[0] == pytest.approx(MOMENT_DENSITY_CEILING)
    assert moment_density_ceiling([lo_v - 5.0])[0] == pytest.approx(MOMENT_DENSITY_CEILING)
    assert moment_density_ceiling([hi_v + 20.0])[0] == pytest.approx(MOMENT_DENSITY_CEILING_HI)

  def test_monotone_over_the_default_grid(self):
    c = moment_density_ceiling(MOMENT_SPEED_BIN_CENTERS)
    assert len(c) == len(MOMENT_SPEED_BIN_CENTERS)
    assert np.all(np.diff(c) >= 0.0)
    assert c[0] == pytest.approx(MOMENT_DENSITY_CEILING)
    assert c[-1] == pytest.approx(MOMENT_DENSITY_CEILING_HI)

  def _feed(self, bank, n_pts=3000, seed=41):
    rng = np.random.default_rng(seed)
    steers = rng.normal(0.0, 0.12, n_pts)      # narrow, the way real steering is
    vegos = rng.uniform(6.0, 39.0, n_pts)
    las = 2.6 * steers + rng.normal(0.0, 0.05, n_pts)
    for s, la, v in zip(steers, las, vegos, strict=True):
      bank.add(float(s), float(la), float(v))
    return bank

  def test_low_anchor_bin_is_untouched_and_the_top_bin_is_not(self):
    """The reason for anchoring the ramp at the bottom: city bins already hit the
    equal-share target, so the change must be provably inert there."""
    centers = list(MOMENT_SPEED_BIN_CENTERS)
    bounds = TorqueEstimatorExt._centers_to_bounds(centers)
    ramped = self._feed(SpeedBinMomentBank(centers, bounds, STEER_BUCKET_BOUNDS))
    flat = self._feed(SpeedBinMomentBank(centers, bounds, STEER_BUCKET_BOUNDS,
                                         density_ceiling=MOMENT_DENSITY_CEILING))
    # a bin's weighting reads only its own dens row and its own ceiling, so the
    # low-anchor bin comes out bit-identical, not merely close
    assert np.array_equal(ramped.M[0], flat.M[0])
    assert np.array_equal(ramped.dens[0], flat.dens[0])
    assert ramped.S[0] == flat.S[0]
    # and the change has to actually bite at the top, or the ramp is pointless
    assert not np.allclose(ramped.M[-1], flat.M[-1])
    # up-weighting starved tail buckets widens the weighted steer spread, which is
    # exactly the leverage the top bin's slope was missing
    assert ramped.x_std()[-1] > flat.x_std()[-1]

  def test_scalar_override_still_supported(self):
    centers = list(MOMENT_SPEED_BIN_CENTERS)
    bounds = TorqueEstimatorExt._centers_to_bounds(centers)
    bank = SpeedBinMomentBank(centers, bounds, STEER_BUCKET_BOUNDS, density_ceiling=9.0)
    assert bank.density_ceiling.shape == (len(centers),)
    assert np.all(bank.density_ceiling == 9.0)
    with pytest.raises(ValueError):
      SpeedBinMomentBank(centers, bounds, STEER_BUCKET_BOUNDS, density_ceiling=[1.0, 2.0])


class TestSpeedBinMomentCenterCeiling:
  """SpeedBinMoment(center=...) must resolve the same ramped density ceiling the
  production bank uses for that bin, so offline replays weight points exactly
  like the device instead of silently reproducing the flat pre-ramp ceiling."""

  def test_center_resolves_the_ramped_ceiling(self):
    for c in (MOMENT_SPEED_BIN_CENTERS[0], 20.0, MOMENT_SPEED_BIN_CENTERS[-1], 99.0):
      mom = SpeedBinMoment(STEER_BUCKET_BOUNDS, center=c)
      assert mom.density_ceiling == pytest.approx(float(moment_density_ceiling(c)))

  def test_center_matches_the_bank_bin_for_bin(self):
    centers = list(MOMENT_SPEED_BIN_CENTERS)
    bounds = TorqueEstimatorExt._centers_to_bounds(centers)
    bank = SpeedBinMomentBank(centers, bounds, STEER_BUCKET_BOUNDS)
    for i, c in enumerate(centers):
      assert SpeedBinMoment(STEER_BUCKET_BOUNDS, center=c).density_ceiling == pytest.approx(float(bank.density_ceiling[i]))

  def test_explicit_ceiling_wins_over_center(self):
    mom = SpeedBinMoment(STEER_BUCKET_BOUNDS, density_ceiling=9.0, center=40.0)
    assert mom.density_ceiling == 9.0

  def test_bare_default_stays_flat(self):
    """Cache-row decoding never weights points, so the bare constructor keeps the
    flat pre-ramp ceiling; replays must pass center= (see class docstring)."""
    assert SpeedBinMoment(STEER_BUCKET_BOUNDS).density_ceiling == MOMENT_DENSITY_CEILING


class TestBankMatchesReference:
  """SpeedBinMomentBank must reproduce the per-bin SpeedBinMoment loop: same
  state, same fits, same validity, over a mixed random stream that includes
  out-of-range speeds and steers and ESS-cap saturation."""

  def _run_pair(self, centers, ess_cap, n_pts=4000, seed=101):
    centers = list(centers)
    bounds = TorqueEstimatorExt._centers_to_bounds(centers)
    # center= resolves each bin's ramped ceiling — the construction any
    # device-parity replay should copy
    moments = [SpeedBinMoment(STEER_BUCKET_BOUNDS, ess_cap=ess_cap, center=c)
               for c in centers]
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

  @pytest.mark.parametrize("centers", [COARSE_CENTERS, MOMENT_SPEED_BIN_CENTERS])
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
    moments, bank = self._run_pair(COARSE_CENTERS, MOMENT_ESS_CAP, n_pts=1500, seed=13)
    bounds = TorqueEstimatorExt._centers_to_bounds(list(COARSE_CENTERS))
    bank2 = SpeedBinMomentBank(COARSE_CENTERS, bounds, STEER_BUCKET_BOUNDS)
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
    moments, bank = self._run_pair(COARSE_CENTERS, MOMENT_ESS_CAP, n_pts=800, seed=17)
    before = bank.S.copy()
    bank.reset_bin(2)
    assert bank.S[2] == 0.0 and bank.S_in[2] == 0.0
    assert not bank.M[2].any() and not bank.dens[2].any()
    others = [i for i in range(bank.n) if i != 2]
    assert np.allclose(bank.S[others], before[others])
