"""Tests for the controller-side speed-dep interpolation and fallback tables."""
from types import SimpleNamespace

import numpy as np
import pytest

from dragonpilot.selfdrive.controls.lib.speed_dep_helpers import (
  build_speed_dep_bp, friction_scale, interp_live_torque_params,
)

SPEED_BP = [10.0, 20.0, 30.0]


def make_tp(centers=None, factors=None, frictions=None, valid=None,
            global_laf=1.25, global_fric=0.125, global_offset=0.01):
  n = len(centers) if centers else 0
  return SimpleNamespace(
    speedBinCenters=centers or [],
    speedBinLatAccelFactors=factors if factors is not None else [0.0] * n,
    speedBinFrictions=frictions if frictions is not None else [0.0] * n,
    speedBinValid=valid if valid is not None else [False] * n,
    latAccelFactorFiltered=global_laf,
    latAccelOffsetFiltered=global_offset,
    frictionCoefficientFiltered=global_fric,
  )


class TestFrictionScale:
  def test_steps_and_clamping(self):
    assert friction_scale(0) == pytest.approx(1.0)
    assert friction_scale(3) == pytest.approx(0.7)
    assert friction_scale(9) == pytest.approx(0.1)
    assert friction_scale(15) == pytest.approx(0.1)   # clamped high
    assert friction_scale(-2) == pytest.approx(1.0)   # clamped low


class TestBuildSpeedDepBp:
  def test_seed_curve_used_for_invalid_bins(self):
    speeds, lafs, frics = build_speed_dep_bp(
      SPEED_BP, [2.0, 2.1, 2.2], [0.10, 0.11, 0.12], [True, False, True],
      [3.0, 3.1, 3.2], [0.20, 0.21, 0.22], 1.25, 0.125)
    assert speeds == SPEED_BP
    assert lafs == [2.0, 3.1, 2.2]          # learned, seed, learned
    assert frics == [0.10, 0.21, 0.12]

  def test_global_fallback_when_nothing_learned_and_no_seeds(self):
    speeds, lafs, frics = build_speed_dep_bp(
      SPEED_BP, [2.0, 2.1, 2.2], [0.1, 0.1, 0.1], [False, False, False],
      None, None, 1.25, 0.125)
    assert speeds == SPEED_BP
    assert lafs == [1.25] * 3
    assert frics == [0.125] * 3

  def test_learned_only_table_without_seeds(self):
    """With no seed curve, the table holds learned bins only: np.interp then
    clamps beyond the ends and bridges unlearned gaps."""
    speeds, lafs, frics = build_speed_dep_bp(
      SPEED_BP, [2.0, 9.9, 2.4], [0.10, 9.9, 0.14], [True, False, True],
      None, None, 1.25, 0.125)
    assert speeds == [10.0, 30.0]
    assert lafs == [2.0, 2.4]
    assert frics == [0.10, 0.14]
    # nearest-learned clamp at the edges, linear bridge across the gap
    assert float(np.interp(5.0, speeds, lafs)) == pytest.approx(2.0)
    assert float(np.interp(35.0, speeds, lafs)) == pytest.approx(2.4)
    assert float(np.interp(20.0, speeds, lafs)) == pytest.approx(2.2)

  def test_length_mismatched_seeds_ignored(self):
    speeds, lafs, _ = build_speed_dep_bp(
      SPEED_BP, [2.0, 2.1, 2.2], [0.1, 0.1, 0.1], [True, False, False],
      [3.0, 3.1], [0.2, 0.2], 1.25, 0.125)   # wrong-length seed tables
    assert speeds == [10.0]                  # fell through to learned-only
    assert lafs == [2.0]

  def test_reduction_scales_learned_friction_only(self):
    _, _, frics = build_speed_dep_bp(
      SPEED_BP, [2.0, 2.1, 2.2], [0.10, 0.20, 0.30], [True, False, True],
      [3.0, 3.1, 3.2], [0.20, 0.21, 0.22], 1.25, 0.125, friction_reduction=5)
    assert frics[0] == pytest.approx(0.05)   # learned, scaled by 0.5
    assert frics[1] == pytest.approx(0.21)   # seed, not scaled
    assert frics[2] == pytest.approx(0.15)   # learned, scaled

  def test_reduction_not_applied_to_global_fallback(self):
    _, _, frics = build_speed_dep_bp(
      SPEED_BP, [0.0] * 3, [0.0] * 3, [False] * 3,
      None, None, 1.25, 0.125, friction_reduction=5)
    assert frics == [0.125] * 3


class TestInterpLiveTorqueParams:
  def test_passthrough_without_bins(self):
    tp = make_tp()
    laf, lao, fric = interp_live_torque_params(tp, 20.0)
    assert laf == pytest.approx(1.25)
    assert lao == pytest.approx(0.01)
    assert fric == pytest.approx(0.125)

  def test_interpolates_learned_curve(self):
    tp = make_tp(SPEED_BP, [2.0, 2.4, 2.8], [0.10, 0.14, 0.18], [True, True, True])
    laf, lao, fric = interp_live_torque_params(tp, 15.0)
    assert laf == pytest.approx(2.2)
    assert fric == pytest.approx(0.12)
    assert lao == pytest.approx(0.01)   # offset always passes through

  def test_differs_at_different_speeds(self):
    tp = make_tp(SPEED_BP, [2.0, 2.4, 2.8], [0.10, 0.14, 0.18], [True, True, True])
    low = interp_live_torque_params(tp, 10.0)
    high = interp_live_torque_params(tp, 30.0)
    assert low[0] != high[0]
    assert low[2] != high[2]

  def test_clamps_beyond_learned_range(self):
    tp = make_tp(SPEED_BP, [2.0, 2.4, 2.8], [0.10, 0.14, 0.18], [True, True, True])
    assert interp_live_torque_params(tp, 0.0)[0] == pytest.approx(2.0)
    assert interp_live_torque_params(tp, 45.0)[0] == pytest.approx(2.8)

  def test_all_invalid_falls_back_to_global(self):
    tp = make_tp(SPEED_BP, [9.0] * 3, [9.0] * 3, [False] * 3)
    laf, _, fric = interp_live_torque_params(tp, 20.0)
    assert laf == pytest.approx(1.25)
    assert fric == pytest.approx(0.125)

  def test_car_cfg_seeds_fill_invalid_bins(self):
    tp = make_tp(SPEED_BP, [2.0, 9.9, 2.8], [0.10, 9.9, 0.18], [True, False, True])
    cfg = {'laf_bp': [3.0, 3.1, 3.2], 'friction_bp': [0.2, 0.21, 0.22]}
    laf, _, fric = interp_live_torque_params(tp, 20.0, cfg)
    assert laf == pytest.approx(3.1)     # seed replaces the invalid middle bin
    assert fric == pytest.approx(0.21)


class TestInterpFrictionReduction:
  """The Friction Reduction setting (dp_lat_torqued_sd_friction) reaches the
  controller through interp_live_torque_params' last argument."""

  def test_scales_learned_friction_and_leaves_laf_alone(self):
    tp = make_tp(SPEED_BP, [2.0, 2.4, 2.8], [0.10, 0.14, 0.18], [True, True, True])
    base_laf, _, base_fric = interp_live_torque_params(tp, 15.0)
    laf, _, fric = interp_live_torque_params(tp, 15.0, None, 3)
    assert laf == pytest.approx(base_laf)          # only friction is touched
    assert fric == pytest.approx(base_fric * 0.7)

  def test_zero_is_a_no_op(self):
    tp = make_tp(SPEED_BP, [2.0, 2.4, 2.8], [0.10, 0.14, 0.18], [True, True, True])
    assert interp_live_torque_params(tp, 15.0, None, 0) == interp_live_torque_params(tp, 15.0)

  def test_inert_when_the_learner_is_off(self):
    """No bins in the message means the learner is off; the setting must not
    quietly reduce the global learner's friction."""
    tp = make_tp()
    assert interp_live_torque_params(tp, 20.0, None, 9) == interp_live_torque_params(tp, 20.0)

  def test_seeded_bins_are_not_reduced(self):
    tp = make_tp(SPEED_BP, [2.0, 9.9, 2.8], [0.10, 9.9, 0.18], [True, False, True])
    cfg = {'laf_bp': [3.0, 3.1, 3.2], 'friction_bp': [0.2, 0.21, 0.22]}
    _, _, fric = interp_live_torque_params(tp, 20.0, cfg, 5)
    assert fric == pytest.approx(0.21)   # seed value, untouched by the scale

  def test_max_step_keeps_friction_positive(self):
    tp = make_tp(SPEED_BP, [2.0, 2.4, 2.8], [0.10, 0.14, 0.18], [True, True, True])
    _, _, fric = interp_live_torque_params(tp, 15.0, None, 9)
    assert 0.0 < fric < 0.12 * 0.11
