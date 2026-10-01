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

  def test_speed_time_shows_lead_speed_and_gap_only(self):
    assert self._lines(ChevronOptions.SPEED_TIME) == ["56 mph", "2.0 s"]

  def test_speed_time_metric(self):
    assert self._lines(ChevronOptions.SPEED_TIME, is_metric=True) == ["90 km/h", "2.0 s"]

  def test_lead_speed_is_absolute(self):
    # lead 5 m/s slower than us
    assert self._lines(ChevronOptions.SPEED_TIME, v_rel=-5.0)[0] == "45 mph"

  def test_gap_unavailable_at_standstill(self):
    assert self._lines(ChevronOptions.SPEED_TIME, d_rel=8.0, v_rel=0.0, v_ego=0.0) == ["0 mph", "---"]

  def test_existing_options_unchanged(self):
    assert self._lines(ChevronOptions.OFF) == []
    assert self._lines(ChevronOptions.DISTANCE_ONLY) == ["164 ft"]
    assert self._lines(ChevronOptions.SPEED_ONLY) == ["56 mph"]
    assert self._lines(ChevronOptions.TTC_ONLY) == ["2.0 s"]
    assert self._lines(ChevronOptions.ALL) == ["164 ft", "56 mph", "2.0 s"]


if __name__ == "__main__":
  unittest.main()
