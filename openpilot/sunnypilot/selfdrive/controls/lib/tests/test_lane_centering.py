"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import openpilot.cereal.messaging as messaging
from opendbc.car.structs import car
from openpilot.sunnypilot.selfdrive.controls.lib.lane_centering import (
  LaneCenteringController, MAX_GAIN, MAX_RAW_CORRECTION, MIN_SPEED,
)

V_EGO = 20.0
BASE_CURVATURE = 0.01  # any nonzero plan the model could have produced
XS = np.linspace(0.0, 50.0, 52)


def path(y, y_std=0.1):
  return SimpleNamespace(x=XS.copy(), y=np.full_like(XS, float(y)), yStd=np.full_like(XS, float(y_std)))


def model(left=-1.8, right=1.8, model_y=0.0, lane_prob=0.9, lane_std=0.1, path_std=0.1, lane_change=0):
  return SimpleNamespace(
    laneLines=[path(0.0), path(left), path(right), path(0.0)],
    laneLineProbs=[0.0, lane_prob, lane_prob, 0.0],
    laneLineStds=[0.0, lane_std, lane_std, 0.0],
    position=path(model_y, path_std),
    meta=SimpleNamespace(laneChangeState=lane_change),
  )


def car_state(v_ego=V_EGO, left_blinker=False, right_blinker=False):
  CS = car.CarState.new_message()
  CS.vEgo = v_ego
  CS.leftBlinker = left_blinker
  CS.rightBlinker = right_blinker
  return CS


def controller(offset=0.0, authority=0.0, pause_on_blinker=True):
  ctrl = LaneCenteringController()
  ctrl.enabled = True
  ctrl.offset = offset
  ctrl.model_authority = authority
  ctrl.pause_on_blinker = pause_on_blinker
  return ctrl


def correction(ctrl, md, CS=None, lat_active=True, model_valid=True):
  CS = car_state() if CS is None else CS
  return ctrl.update(BASE_CURVATURE, md, CS, lat_active, model_valid) - BASE_CURVATURE


def converge(md, offset=0.0, authority=0.0, seconds=3.0):
  ctrl = controller(offset=offset, authority=authority)
  out = 0.0
  for _ in range(int(seconds / 0.01)):
    out = correction(ctrl, md)
  return ctrl, out


def test_params_defaults_are_registered():
  ctrl = LaneCenteringController()
  assert not ctrl.enabled
  assert ctrl.offset == 0.0
  assert ctrl.model_authority == 1.0
  assert ctrl.pause_on_blinker


@pytest.mark.parametrize("kwargs", [
  {"enabled": False},
  {"lat_active": False},
  {"model_valid": False},
  {"v_ego": MIN_SPEED - 0.1},
])
def test_hard_gates_pass_the_plan_through(kwargs):
  ctrl = controller()
  ctrl.enabled = kwargs.pop("enabled", True)
  CS = car_state(v_ego=kwargs.pop("v_ego", V_EGO))
  assert correction(ctrl, model(left=-1.5, right=2.1), CS, **kwargs) == 0.0


def test_lane_change_is_noop():
  assert correction(controller(), model(left=-1.5, right=2.1, lane_change=1)) == 0.0


def test_lane_change_drops_a_built_up_correction():
  ctrl, built_up = converge(model(left=-1.5, right=2.1))
  assert built_up > 0.0
  assert correction(ctrl, model(left=-1.5, right=2.1, lane_change=1)) == 0.0
  assert ctrl.correction == 0.0


def test_steers_toward_lane_center():
  _, from_right = converge(model(left=-1.5, right=2.1))  # sitting right of center
  _, from_left = converge(model(left=-2.1, right=1.5))
  assert from_right > 0.0  # positive curvature steers left
  assert from_left < 0.0


def test_small_center_error_does_not_chatter():
  _, out = converge(model(left=-1.75, right=1.85))
  assert out == 0.0


def test_offset_direction():
  _, left = converge(model(), offset=0.2)
  _, right = converge(model(), offset=-0.2)
  assert left > 0.0
  assert right < 0.0


def test_offset_is_reduced_in_narrow_lane():
  narrow = model(left=-1.3, right=1.3)
  _, at_safe_limit = converge(narrow, offset=0.2)
  _, above_safe_limit = converge(narrow, offset=0.3)
  assert np.isclose(at_safe_limit, above_safe_limit)


@pytest.mark.parametrize("field,value", [
  ("laneLineProbs", np.nan),
  ("laneLineProbs", 1.1),
  ("laneLineProbs", 0.5),
  ("laneLineStds", np.nan),
  ("laneLineStds", -0.1),
  ("laneLineStds", 0.4),
])
def test_invalid_lane_confidence_is_rejected(field, value):
  md = model(left=-1.5, right=2.1)
  getattr(md, field)[1] = value
  assert correction(controller(), md) == 0.0


@pytest.mark.parametrize("width", [2.5, 4.9])
def test_implausible_lane_width_is_rejected(width):
  assert correction(controller(), model(left=-width / 2 - 0.3, right=width / 2 - 0.3)) == 0.0


def test_input_must_cover_lookahead():
  md = model(left=-1.5, right=2.1)
  md.laneLines[1].x = md.laneLines[1].x[:10]
  md.laneLines[1].y = md.laneLines[1].y[:10]
  assert correction(controller(), md) == 0.0


def test_confidence_loss_releases_smoothly():
  ctrl, out = converge(model(left=-1.5, right=2.1))
  assert out > 0.0

  faded = correction(ctrl, model(left=-1.5, right=2.1, lane_prob=0.2))
  assert 0.0 < faded < out

  for _ in range(300):
    faded = correction(ctrl, model(left=-1.5, right=2.1, lane_prob=0.2))
  assert abs(faded) < 1e-6


def test_blinker_fades_the_correction_out():
  md = model(left=-1.5, right=2.1)
  ctrl, centered = converge(md)
  blinking = car_state(left_blinker=True)

  faded = correction(ctrl, md, blinking)
  assert 0.0 < faded < centered

  for _ in range(300):
    faded = correction(ctrl, md, blinking)
  assert abs(faded) < 1e-6


def test_blinker_pause_can_be_disabled():
  md = model(left=-1.5, right=2.1)
  ctrl, centered = converge(md)
  ctrl.pause_on_blinker = False
  assert correction(ctrl, md, car_state(left_blinker=True)) == pytest.approx(centered, abs=1e-7)


def test_confident_model_keeps_authority_when_far_off_center():
  md = model(left=-1.0, right=2.6, path_std=0.1)
  _, lane_lines_win = converge(md, authority=0.0)
  _, model_wins = converge(md, authority=1.0)
  assert lane_lines_win > 0.0
  assert abs(model_wins) < 1e-9


def test_unsure_model_does_not_keep_authority():
  _, out = converge(model(left=-1.0, right=2.6, path_std=0.6), authority=1.0)
  assert out > 0.0


def test_model_authority_blends():
  md = model(left=-1.2, right=2.4, path_std=0.1)
  _, lane_only = converge(md, authority=0.0)
  _, blended = converge(md, authority=0.5)
  _, model_only = converge(md, authority=1.0)
  assert lane_only > blended > model_only >= 0.0


def test_correction_is_smoothed_and_capped():
  md = model(left=0.0, right=3.0, path_std=0.6)  # absurd error, far past the cap
  ctrl = controller()
  first = correction(ctrl, md)
  _, steady = converge(md)
  assert 0.0 < first < steady
  assert np.isclose(steady, MAX_RAW_CORRECTION * MAX_GAIN, atol=1e-6)


def test_runs_on_a_real_model_message():
  msg = messaging.new_message('modelV2')
  md = msg.modelV2
  md.init('laneLines', 4)
  for i, y in enumerate((-3.3, -1.5, 2.1, 3.9)):
    md.laneLines[i].x = XS.tolist()
    md.laneLines[i].y = np.full_like(XS, y).tolist()
  md.laneLineProbs = [0.2, 0.9, 0.9, 0.2]
  md.laneLineStds = [0.3, 0.1, 0.1, 0.3]
  md.position.x = XS.tolist()
  md.position.y = np.zeros_like(XS).tolist()
  md.position.yStd = np.full_like(XS, 0.1).tolist()
  md.meta.laneChangeState = 'off'

  ctrl = controller()
  out = 0.0
  for _ in range(300):
    out = correction(ctrl, md)
  assert out > 0.0
