"""Schema checks for the two lateral torque settings added alongside the learner.

These entries are only exercised on the device UI, so a typo in a key, a range or
a depends_on expression otherwise surfaces as a silently dead control. The keys
must also match exactly what controlsd and the extension read at runtime.
"""
import pytest

from dragonpilot.settings import SETTINGS, extract_depends_on_refs

FRICTION_KEY = "dp_lat_torqued_sd_friction"
JERK_KEY = "dp_lat_jerk_torque"
LEARNER_KEY = "dp_lat_torqued_sd"


@pytest.fixture(scope="module")
def items():
  return {item["key"]: item for section in SETTINGS for item in section["settings"] if "key" in item}


class TestFrictionReductionSetting:
  def test_declared_in_the_lateral_section(self, items):
    assert items[FRICTION_KEY]["section"] == "Lateral"

  def test_is_an_int_spinner_defaulting_to_off(self, items):
    it = items[FRICTION_KEY]
    assert it["type"] == "spin_button_item"
    assert it["param_type"] == "INT"
    assert it["default"] == "0"

  def test_range_matches_the_friction_scale_the_controller_applies(self, items):
    """friction_scale() clamps to [0, 9]; a wider spinner would offer steps that
    silently do nothing."""
    from dragonpilot.selfdrive.controls.lib.speed_dep_helpers import friction_scale
    it = items[FRICTION_KEY]
    assert (it["min_val"], it["max_val"], it["step"]) == (0, 9, 1)
    assert friction_scale(it["min_val"]) == pytest.approx(1.0)
    assert friction_scale(it["max_val"]) == pytest.approx(0.1)

  def test_depends_on_the_learner_toggle(self, items):
    it = items[FRICTION_KEY]
    assert extract_depends_on_refs(it["depends_on"]) == {LEARNER_KEY}
    assert LEARNER_KEY in items   # the reference must resolve


class TestJerkTorqueSetting:
  def test_is_a_bool_toggle_defaulting_off(self, items):
    it = items[JERK_KEY]
    assert it["section"] == "Lateral"
    assert it["type"] == "toggle_item"
    assert it["param_type"] == "BOOL"
    assert it["default"] == "0"

  def test_is_persistent_so_it_survives_a_reboot(self, items):
    assert "PERSISTENT" in items[JERK_KEY]["flags"]

  def test_asks_for_a_restart(self, items):
    """Both settings are read once at controlsd start, so the UI has to say so."""
    assert items[JERK_KEY]["needs_restart"]
    assert items[FRICTION_KEY]["needs_restart"]

  def test_brand_gate_matches_the_learner_toggle(self, items):
    assert items[JERK_KEY]["brands"] == items[LEARNER_KEY]["brands"]
