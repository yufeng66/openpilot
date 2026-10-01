import unittest
from unittest import mock

from openpilot.selfdrive.ui.sunnypilot.onroad import chevron_metrics
from openpilot.selfdrive.ui.sunnypilot.onroad.chevron_metrics import ChevronMetrics, ChevronOptions

# 50 m behind a lead doing 25 m/s while we do 25 m/s: 56 mph, 2.0 s gap, 164 ft
D_REL, V_REL, V_EGO = 50.0, 0.0, 25.0


class TestChevronMetrics(unittest.TestCase):
  def _lines(self, option, is_metric=False, d_rel=D_REL, v_rel=V_REL, v_ego=V_EGO):
    with mock.patch.object(chevron_metrics.ui_state, "chevron_metrics", option, create=True), \
         mock.patch.object(chevron_metrics.ui_state, "is_metric", is_metric, create=True):
      return ChevronMetrics._build_text_lines(d_rel, v_rel, v_ego)

  def test_rel_speed_gap_is_one_line_without_units(self):
    # lead 5 mph slower than us, 1.8 s behind it
    v_rel = -5 / 2.23694
    assert self._lines(ChevronOptions.REL_SPEED_GAP, d_rel=1.8 * V_EGO, v_rel=v_rel) == ["-5, 1.8"]

  def test_rel_speed_gap_signs(self):
    assert self._lines(ChevronOptions.REL_SPEED_GAP, v_rel=3 / 2.23694) == ["+3, 2.0"]
    assert self._lines(ChevronOptions.REL_SPEED_GAP, v_rel=0.0) == ["0, 2.0"]
    # rounds to zero: no "-0" or "+0"
    assert self._lines(ChevronOptions.REL_SPEED_GAP, v_rel=-0.2) == ["0, 2.0"]

  def test_rel_speed_gap_metric(self):
    assert self._lines(ChevronOptions.REL_SPEED_GAP, is_metric=True, v_rel=-10 / 3.6) == ["-10, 2.0"]

  def test_rel_speed_gap_at_standstill(self):
    assert self._lines(ChevronOptions.REL_SPEED_GAP, d_rel=8.0, v_rel=0.0, v_ego=0.0) == ["0, ---"]

  def test_existing_options_unchanged(self):
    assert self._lines(ChevronOptions.OFF) == []
    assert self._lines(ChevronOptions.DISTANCE_ONLY) == ["164 ft"]
    assert self._lines(ChevronOptions.SPEED_ONLY) == ["56 mph"]
    assert self._lines(ChevronOptions.TTC_ONLY) == ["2.0 s"]
    assert self._lines(ChevronOptions.ALL) == ["164 ft", "56 mph", "2.0 s"]


if __name__ == "__main__":
  unittest.main()
