"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lane-line based centering assist, ported from StarPilot's confidence-gated lane
centering (firestar5683/StarPilot, MIT, commit 9f1066ce plus follow-ups).

openpilot steers end-to-end: the model emits a desired curvature and the lane lines it
predicts are only drawn on screen. This adds a small, heavily gated curvature correction
on top of that plan, pulling the car back toward the geometric center of the lane when
the model settles off-center. The model keeps the last word: when it is confident about
its own path and has deliberately moved well away from center, the correction fades out
instead of fighting it.
"""
import numpy as np

from opendbc.car.structs import car

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import smooth_value

MIN_SPEED = 5.0  # m/s, no centering below this
MIN_LANE_PROB = 0.6  # both lines of our lane have to be at least this likely
MAX_LANE_STD = 0.3  # m, and this well localized
MIN_LANE_WIDTH = 2.6  # m
MAX_LANE_WIDTH = 4.8  # m
MAX_OFFSET = 0.3  # m, how far the driver may bias the target off center
MIN_CENTER_TO_LINE = 1.1  # m, never aim closer than this to a lane line
MAX_RAW_CORRECTION = 0.004  # 1/m, before gain
MAX_GAIN = 0.3
SMOOTH_TAU = 0.4  # s
SIGNAL_RELEASE_TAU = 0.2  # s
CONFIDENCE_RELEASE_TAU = 0.2  # s
CENTER_ERROR_DEADBAND = 0.08  # m
LOOKAHEAD_MIN = 8.0  # m
LOOKAHEAD_MAX = 35.0  # m
E2E_MAX_PATH_STD = 0.35  # m, above this the model is unsure and the lane lines win
E2E_BREAK_IN_START = 0.15  # m
E2E_BREAK_IN_FULL = 0.5  # m


def valid_path(x: np.ndarray, y: np.ndarray) -> bool:
  return bool(x.size >= 2 and x.size == y.size and np.isfinite(x).all() and np.isfinite(y).all() and np.all(np.diff(x) > 0))


def covers(x: np.ndarray, distance: float) -> bool:
  return bool(x[0] <= distance <= x[-1])


class LaneCenteringController:
  def __init__(self):
    self.params = Params()
    self.enabled = False
    self.offset = 0.0
    self.model_authority = 1.0
    self.pause_on_blinker = True
    self.correction = 0.0
    self.get_params()

  def get_params(self) -> None:
    self.enabled = self.params.get_bool("LaneCenteringEnabled")
    self.offset = float(np.clip(self.params.get("LaneCenteringOffset", return_default=True), -MAX_OFFSET, MAX_OFFSET))
    self.model_authority = float(np.clip(self.params.get("LaneCenteringModelAuthority", return_default=True), 0.0, 1.0))
    self.pause_on_blinker = self.params.get_bool("LaneCenteringPauseOnBlinker")

  def reset(self) -> None:
    self.correction = 0.0

  def update(self, desired_curvature: float, model_v2, CS: car.CarState, lat_active: bool, model_valid: bool) -> float:
    if not self.enabled or not lat_active or not model_valid or CS.vEgo < MIN_SPEED:
      self.reset()
      return desired_curvature

    if model_v2.meta.laneChangeState != log.LaneChangeState.off:
      self.reset()
      return desired_curvature

    # bleed off instead of dropping the correction: the model is already steering somewhere
    if self.pause_on_blinker and (CS.leftBlinker or CS.rightBlinker):
      self.correction = float(smooth_value(0.0, self.correction, SIGNAL_RELEASE_TAU, dt=DT_CTRL))
      return desired_curvature + self.correction

    valid, raw_correction = self._raw_correction(model_v2, CS.vEgo)
    if not valid:
      self.correction = float(smooth_value(0.0, self.correction, CONFIDENCE_RELEASE_TAU, dt=DT_CTRL))
      return desired_curvature + self.correction

    target = float(np.clip(raw_correction, -MAX_RAW_CORRECTION, MAX_RAW_CORRECTION)) * MAX_GAIN
    self.correction = float(smooth_value(target, self.correction, SMOOTH_TAU, dt=DT_CTRL))
    return desired_curvature + self.correction

  def _raw_correction(self, model_v2, v_ego: float) -> tuple[bool, float]:
    lane_lines = model_v2.laneLines
    probs = np.asarray(model_v2.laneLineProbs, dtype=float)
    stds = np.asarray(model_v2.laneLineStds, dtype=float)
    if len(lane_lines) < 3 or probs.size < 3 or stds.size < 3:
      return False, 0.0

    # index 1 and 2 are the two lines of the lane we are in
    if not np.isfinite(probs[[1, 2]]).all() or not np.isfinite(stds[[1, 2]]).all():
      return False, 0.0
    if np.any(probs[[1, 2]] < MIN_LANE_PROB) or np.any(probs[[1, 2]] > 1.0):
      return False, 0.0
    if np.any(stds[[1, 2]] < 0.0) or np.any(stds[[1, 2]] > MAX_LANE_STD):
      return False, 0.0

    left_x = np.asarray(lane_lines[1].x, dtype=float)
    left_y = np.asarray(lane_lines[1].y, dtype=float)
    right_x = np.asarray(lane_lines[2].x, dtype=float)
    right_y = np.asarray(lane_lines[2].y, dtype=float)
    pos_x = np.asarray(model_v2.position.x, dtype=float)
    pos_y = np.asarray(model_v2.position.y, dtype=float)
    if not (valid_path(left_x, left_y) and valid_path(right_x, right_y) and valid_path(pos_x, pos_y)):
      return False, 0.0

    lookahead = float(np.clip(v_ego, LOOKAHEAD_MIN, LOOKAHEAD_MAX))
    if not all(covers(x, lookahead) for x in (left_x, right_x, pos_x)):
      return False, 0.0

    left = float(np.interp(lookahead, left_x, left_y))
    right = float(np.interp(lookahead, right_x, right_y))
    width = right - left
    if not MIN_LANE_WIDTH <= width <= MAX_LANE_WIDTH:
      return False, 0.0

    # keep the driver's offset from ever aiming us at a lane line in a narrow lane
    max_safe_offset = min(MAX_OFFSET, max(0.0, width * 0.5 - MIN_CENTER_TO_LINE))
    target_y = 0.5 * (left + right) + float(np.clip(self.offset, -max_safe_offset, max_safe_offset))
    model_y = float(np.interp(lookahead, pos_x, pos_y))
    error = target_y - model_y
    error_abs = abs(error)
    if error_abs <= CENTER_ERROR_DEADBAND:
      error = 0.0
    else:
      error = float(np.copysign(error_abs - CENTER_ERROR_DEADBAND, error))

    # the model gets the last word: while it is sure of its own path, hand authority back
    # to it as its deviation from center grows, so deliberate offsets are not fought
    pos_y_std = np.asarray(model_v2.position.yStd, dtype=float)
    if valid_path(pos_x, pos_y_std):
      path_std = float(np.interp(lookahead, pos_x, pos_y_std))
      if 0.0 <= path_std <= E2E_MAX_PATH_STD:
        break_in = float(np.clip((error_abs - E2E_BREAK_IN_START) / (E2E_BREAK_IN_FULL - E2E_BREAK_IN_START), 0.0, 1.0))
        error *= 1.0 - self.model_authority * break_in

    # curvature of the arc that clears `error` over `lookahead`: y = k * x^2 / 2
    raw_correction = 2.0 * error / lookahead ** 2
    return bool(np.isfinite(raw_correction)), float(raw_correction)
