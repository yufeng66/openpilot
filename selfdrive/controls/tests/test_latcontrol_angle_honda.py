"""Honda Accord on the angle-loop EPS firmware, the only angle-controlled Honda: LatControlAngle sends the angle
through the rack map and takes saturation from the carcontroller's own limiter; controlsd inverts the map on the
measured angle. Other angle cars are unchanged."""
import math

import pytest

import cereal.messaging as messaging
from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.steer_ratio import get_honda_vgr_inverse, vgr_linear_to_physical
from opendbc.car.honda.values import CAR as HONDA, HondaFlags
from opendbc.car.nissan.values import CAR as NISSAN
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.params import Params
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque

DT = 0.01
ANGLE_LOOP_FW = b"39990-TVA,A16C\x00\x00"
ERROR_DEG = 6.0  # an ordinary turn-entry error on the angle-loop EPS, above the 2.5 deg raw-error threshold


def _accord():
  car_fw = [car.CarParams.CarFw(ecu=car.CarParams.Ecu.eps, fwVersion=ANGLE_LOOP_FW, address=0x18DA30F1, subAddress=0)]
  fingerprint = {i: {} for i in range(8)}
  return interfaces[HONDA.HONDA_ACCORD].get_params(HONDA.HONDA_ACCORD, fingerprint, car_fw, False, False, 0, False).as_reader()


def _leaf():
  return interfaces[NISSAN.NISSAN_LEAF].get_non_essential_params(NISSAN.NISSAN_LEAF).as_reader()


def _desired_angle(CP, curvature, v_ego, angle_offset):
  """The controller's desired angle beside VehicleModel's centre-equivalent one."""
  VM = VehicleModel(CP)
  CS = car.CarState.new_message(vEgo=v_ego)
  params = log.LiveParametersData.new_message(angleOffsetDeg=angle_offset)
  _, angle, _ = LatControlAngle(CP, None, DT).update(True, CS, VM, params, False, curvature, False, 0.2)
  return angle, math.degrees(VM.get_steer_from_curvature(-curvature, v_ego, 0.0))


def _saturated(CP, steer_limited_by_safety):
  """lac_log.saturated per frame while the wheel sits ERROR_DEG off the desired angle, past steerLimitTimer."""
  lac = LatControlAngle(CP, None, DT)
  VM = VehicleModel(CP)
  CS = car.CarState.new_message(vEgo=20.0, steeringAngleDeg=ERROR_DEG)
  params = log.LiveParametersData.new_message()
  return [lac.update(True, CS, VM, params, steer_limited_by_safety, 0.0, False, 0.2)[2].saturated
          for _ in range(round((CP.steerLimitTimer + 0.5) / DT))]


# 7 deg sits on the map's flat centre (below 31 deg), 139 deg well past it
@pytest.mark.parametrize("curvature, v_ego, on_centre", [(0.002, 20.0, True), (-0.05, 5.0, False)])
def test_accord_desired_angle_goes_through_the_rack_map(curvature, v_ego, on_centre):
  CP = _accord()
  inverse = get_honda_vgr_inverse(CP.flags)
  assert inverse is not None

  angle, linear = _desired_angle(CP, curvature, v_ego, angle_offset=1.5)
  assert angle == pytest.approx(vgr_linear_to_physical(linear, inverse) + 1.5)
  if on_centre:
    assert angle - 1.5 == pytest.approx(linear)
  else:
    assert abs(angle - 1.5) < abs(linear) - 5.0  # the rack quickens off centre: less wheel for the same curvature


def test_accord_saturation_follows_the_carcontroller_not_the_angle_error():
  CP = _accord()
  assert not any(_saturated(CP, steer_limited_by_safety=False))

  saturated = _saturated(CP, steer_limited_by_safety=True)
  assert saturated[-1]
  assert not saturated[round(CP.steerLimitTimer / DT) - 10]  # debounced by steerLimitTimer


def test_other_angle_cars_are_unchanged():
  CP = _leaf()
  angle, linear = _desired_angle(CP, 0.05, 5.0, angle_offset=1.5)
  assert angle == linear + 1.5  # no rack map
  assert _saturated(CP, steer_limited_by_safety=False)[-1]  # raw angle-error rule


class FakeSubMaster:
  """The parts of SubMaster that Controls.state_control reads."""
  def __init__(self, services):
    self.msgs = {s: getattr(messaging.new_message(s, 0) if s == 'onroadEvents' else messaging.new_message(s), s) for s in services}
    self.valid = dict.fromkeys(services, True)
    self.valid['lateralManeuverPlan'] = False

  def __getitem__(self, s):
    return self.msgs[s]

  def all_checks(self, services=None):
    return True


def _controls(CP):
  Params().put("CarParams", CP.as_builder().to_bytes())
  controls = Controls()
  controls.sm = FakeSubMaster(['liveDelay', 'liveParameters', 'liveTorqueParameters', 'modelV2', 'selfdriveState', 'liveCalibration',
                               'longitudinalPlan', 'carState', 'carOutput', 'onroadEvents', 'driverAssistance', 'carStateExt'])
  return controls


@pytest.mark.parametrize("wheel", [0.0, 12.0, -60.0, 140.0, -300.0])
def test_controlsd_seeds_the_angle_it_measures(wheel):
  # while lateral is inactive controlsd seeds the desired curvature from the measured angle; through the inverse rack
  # map that curvature comes back as the same angle, so engaging commands no step
  CP = _accord()
  controls = _controls(CP)
  assert isinstance(controls.LaC, LatControlAngle) and controls.vgr_inverse is not None
  lp = controls.sm['liveParameters']
  lp.steerRatio, lp.stiffnessFactor, lp.angleOffsetDeg = CP.steerRatio, 1.0, 1.5
  cs = controls.sm['carState']
  cs.vEgo, cs.steeringAngleDeg = 3.0, wheel

  for _ in range(500):  # clip_curvature rate-limits the seed towards the measured curvature
    CC, lac_log = controls.state_control()
  assert not CC.latActive
  assert controls.desired_curvature == controls.curvature
  _, angle, _ = controls.LaC.update(True, cs, controls.VM, lp, False, controls.desired_curvature, False, 0.2)
  assert angle == pytest.approx(wheel, abs=1e-6)


def test_controlsd_torque_car_has_no_rack_map():
  CP = interfaces[TOYOTA.TOYOTA_RAV4].get_non_essential_params(TOYOTA.TOYOTA_RAV4).as_reader()
  controls = _controls(CP)
  assert isinstance(controls.LaC, LatControlTorque) and controls.vgr_inverse is None


def test_angle_loop_accord_is_not_torque_tuned():
  # controlsd feeds the torque learner and the jerk torque extension only to a torque tune, so this must not be one
  CP = _accord()
  assert CP.flags & HondaFlags.EPS_ANGLE_LOOP_FW
  assert CP.lateralTuning.which() != 'torque'
