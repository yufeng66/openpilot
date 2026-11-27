"""dp - the Rel+Gap lead readout: text, opacity, placement. Pure functions, no raylib needed."""
import math

import pytest

from openpilot.common.constants import CV
from dragonpilot.selfdrive.ui import lead_readout as lr

MPH = CV.MPH_TO_MS
KPH = CV.KPH_TO_MS
RECT = (30.0, 30.0, 2100.0, 1020.0)  # the content rect of the big onroad view (2160x1080 minus the 30 px border)


@pytest.mark.parametrize("d_rel, v_rel, v_ego, metric, text", [
  (54.0, -5 * MPH, 30.0, False, "-5, 1.8"),
  (54.0, 5 * MPH, 30.0, False, "+5, 1.8"),
  (54.0, -0.4 * MPH, 30.0, False, "0, 1.8"),  # 0, not -0
  (54.0, 0.0, 30.0, False, "0, 1.8"),
  (30.0, -12 * KPH, 25.0, True, "-12, 1.2"),
  (20.0, -3.0, 0.0, False, "-7, ---"),  # standstill: no time gap
  (0.0, 0.0, 10.0, False, "0, ---"),
  (2500.0, 0.0, 10.0, False, "0, ---"),  # 250 s is not a useful gap
])
def test_text(d_rel, v_rel, v_ego, metric, text):
  assert lr.rel_gap_text(d_rel, v_rel, v_ego, metric) == text


NAN, INF = math.nan, math.inf


@pytest.mark.parametrize("d_rel, v_rel, v_ego, text", [
  (54.0, NAN, 30.0, "---, 1.8"),
  (54.0, INF, 30.0, "---, 1.8"),
  (54.0, -INF, 30.0, "---, 1.8"),
  (NAN, -5 * MPH, 30.0, "-5, ---"),
  (INF, -5 * MPH, 30.0, "-5, ---"),
  (-INF, -5 * MPH, 30.0, "-5, ---"),
  (54.0, -5 * MPH, NAN, "-5, ---"),
  (54.0, -5 * MPH, INF, "-5, ---"),
  (54.0, -5 * MPH, -INF, "-5, ---"),
  (NAN, NAN, NAN, "---, ---"),
  (INF, INF, INF, "---, ---"),
])
def test_text_never_raises(d_rel, v_rel, v_ego, text):
  assert lr.rel_gap_text(d_rel, v_rel, v_ego, False) == text
  metric = lr.rel_gap_text(d_rel, v_rel, v_ego, True)
  assert metric.split(", ")[1] == text.split(", ")[1] and (metric.startswith("---") == text.startswith("---"))


def test_position_default_reserves_the_hud_column():
  x, _ = lr.rel_gap_text_position(40.0, 500.0, 50.0, 200.0, *RECT)
  assert x == RECT[0] + lr.LEFT_COLUMN_RESERVED_PX + lr.MARGIN


def test_alpha():
  assert lr.text_alpha(True) == 1.0
  assert lr.text_alpha(False) == pytest.approx(0.55)


def test_position_centred_under_chevron():
  x, y = lr.rel_gap_text_position(1000.0, 500.0, 50.0, 200.0, *RECT, left_reserved=360.0)
  assert x == 900.0 and y == 500.0 + 50.0 + lr.GAP_BELOW_CHEVRON


def test_position_left_edge_keeps_off_the_hud_column():
  x, _ = lr.rel_gap_text_position(40.0, 500.0, 50.0, 200.0, *RECT, left_reserved=360.0)
  assert x == RECT[0] + 360.0 + lr.MARGIN


def test_position_right_edge():
  x, _ = lr.rel_gap_text_position(2120.0, 500.0, 50.0, 200.0, *RECT, left_reserved=360.0)
  assert x == RECT[0] + RECT[2] - lr.MARGIN - 200.0


def test_position_bottom_goes_above_the_chevron():
  _, y = lr.rel_gap_text_position(1000.0, 1000.0, 50.0, 200.0, *RECT, left_reserved=360.0)
  assert y + lr.LINE_HEIGHT <= 1000.0
  assert y >= RECT[1] + lr.MARGIN
