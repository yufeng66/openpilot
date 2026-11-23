"""Accord on the angle-loop EPS firmware (39990-TVA,A16<letter> / A34<letter>): detection, the 0xE4 frame, the setpoint limits, the rack map."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.can import CANPacker
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.fw_versions import match_fw_to_car
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.fingerprints import FW_VERSIONS
from opendbc.car.honda.hondacan import CanBus, create_steering_control
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.steer_ratio import ACCORD_ANGLE_LOOP_STEER_RATIO, _ACCORD_ANGLE_LOOP_RATIO, _ACCORD_ANGLE_LOOP_RATIO_BP, \
                                          get_honda_vgr_inverse, vgr_linear_to_physical, vgr_physical_to_linear
from opendbc.car.honda.values import CAR, DBC, CarControllerParams, HondaFlags, is_accord_eps_angle_loop_fw
from opendbc.car.interfaces import get_torque_params
from opendbc.car.lateral import get_max_angle_delta_vm

P = CarControllerParams
SteerControlType = structs.CarParams.SteerControlType
STEERING_CONTROL = 0xE4
ACCORD_DBC = DBC[CAR.HONDA_ACCORD][Bus.pt]
ANGLE_FW = (b"39990-TVA,A16A\x00\x00", b"39990-TVA,A16B\x00\x00", b"39990-TVA,A16C\x00\x00", b"39990-TVA,A34C\x00\x00",
            b"39990-TVA,A34D\x00\x00")
TORQUE_MOD_FW = (b"39990-TVA,A150\x00\x00", b"39990-TVA,A160\x00\x00", b"39990-TVA,A340\x00\x00")
SERIAL = b"\x11L-130520-03461268" + b" " * 13  # a logging-only EPS response (HONDA_ALT_VERSION_REQUEST)


def eps(version, *, logging=False, address=0x18DA30F1, sub=0):
  return structs.CarParams.CarFw(ecu="eps", address=address, subAddress=sub, fwVersion=version, brand="honda", logging=logging)


def params(candidate, firmware):
  fingerprint = gen_empty_fingerprint()
  fingerprint[1][0x1A3] = 8
  return CarInterface.get_params(candidate, fingerprint, firmware, False, False, 0, False)


def accord_cp(*versions):
  return params(CAR.HONDA_ACCORD, [eps(version) for version in versions])


@pytest.mark.parametrize("fw, expected", [
  (b"39990-TVA,A16A\x00\x00", True),   # NUL-padded as the car reports it
  (b"39990-TVA,A16B\x00\x00", True),
  (b"39990-TVA,A16C", True),
  (b"39990-TVA,A34C\x00\x00", True),   # the A340 compile
  (b"39990-TVA,A34D\x00\x00", True),
  (b"39990-TVA,A150\x00\x00", False),  # torque-map (2x) images
  (b"39990-TVA,A160\x00\x00", False),
  (b"39990-TVA,A340\x00\x00", False),
  (b"39990-TVA-A160\x00\x00", False),  # stock
  (b"39990-TVA-A340\x00\x00", False),
  (b"39990-TVA,A24C", False),          # no such family
  (b"39990-TVA,A16", False),
  (b"39990-TVA,A16AB", False),
  (b"39990-TVA,A16a", False),
  (b"39990-TVB,A16A", False),
  (b"", False),
])
def test_fw_detection(fw, expected):
  assert is_accord_eps_angle_loop_fw(fw) is expected


@pytest.mark.parametrize("fw", ANGLE_FW)
def test_angle_fw_selects_angle_control(fw):
  CP = accord_cp(fw)
  assert CP.steerControlType == SteerControlType.angle
  assert CP.flags & HondaFlags.EPS_ANGLE_LOOP_FW
  assert not CP.dashcamOnly
  assert CP.steerActuatorDelay == pytest.approx(P.ANGLE_STEER_ACTUATOR_DELAY)
  assert CP.steerRatio == pytest.approx(ACCORD_ANGLE_LOOP_STEER_RATIO)
  # the onroad torque bar divides by CP.maxLateralAccel; the torque fit's value would peg it on every turn
  assert CP.maxLateralAccel == pytest.approx(3.0 + 9.81 * 0.06)
  # not a torque tune: the torque learner and the torque controller extensions stay off
  assert CP.lateralTuning.which() == 'pid'
  assert get_honda_vgr_inverse(CP.flags) is not None
  assert CarController({Bus.pt: ACCORD_DBC}, CP).angle_mode


def test_angle_limits_match_the_source():
  limits = P.ANGLE_LIMITS
  assert limits.STEER_ANGLE_MAX == 400
  assert limits.MAX_ANGLE_RATE == 2.5
  assert limits.MAX_LATERAL_ACCEL == pytest.approx(3.0 + 9.81 * 0.06)
  assert limits.MAX_LATERAL_JERK == pytest.approx(3.0 + 9.81 * 0.06)
  assert P.STEER_STEP == 1
  assert P.ANGLE_ERROR_MAX_BP == [3.1, 8.0, 10.0, 11.75, 17.5, 26.9]
  assert P.ANGLE_ERROR_MAX_V == [35.4, 23.4, 45.0, 61.0, 25.0, 12.2]
  assert P.ANGLE_ARM == 2
  assert P.ANGLE_STEER_ACTUATOR_DELAY == 0.15


@pytest.mark.parametrize("fw", [None, b"39990-TVA-A160\x00\x00", *TORQUE_MOD_FW])
def test_other_fw_keeps_torque_control(fw):
  CP = accord_cp(*([] if fw is None else [fw]))
  stock = accord_cp()
  assert CP.steerControlType == SteerControlType.torque
  assert not CP.flags & HondaFlags.EPS_ANGLE_LOOP_FW and not CP.dashcamOnly
  if fw in TORQUE_MOD_FW:
    # dp: any comma EPS image keeps the modified-EPS PID tune
    assert CP.lateralTuning.which() == 'pid'
    assert list(CP.lateralTuning.pid.kpV) == pytest.approx([0.3])
  else:
    assert CP.lateralTuning.which() == 'torque'
  assert (CP.steerRatio, CP.steerActuatorDelay) == (stock.steerRatio, stock.steerActuatorDelay)
  assert CP.maxLateralAccel == pytest.approx(get_torque_params()[CAR.HONDA_ACCORD]['MAX_LAT_ACCEL_MEASURED'])
  assert get_honda_vgr_inverse(CP.flags) is None
  assert not CarController({Bus.pt: ACCORD_DBC}, CP).angle_mode


@pytest.mark.parametrize("candidate, firmware", [
  (CAR.HONDA_CIVIC_BOSCH, [eps(ANGLE_FW[2])]),                                # not an Accord
  (CAR.HONDA_CRV_5G, [eps(ANGLE_FW[3])]),
  (CAR.HONDA_ACCORD, [eps(ANGLE_FW[2], address=0x18DA31F1)]),
  (CAR.HONDA_ACCORD, [eps(ANGLE_FW[2], sub=1)]),
  (CAR.HONDA_ACCORD, [eps(ANGLE_FW[2]), eps(b"39990-TVA-A160\x00\x00")]),  # mixed with stock
  (CAR.HONDA_ACCORD, [eps(ANGLE_FW[2]), eps(TORQUE_MOD_FW[0])]),            # mixed with the torque mod
  (CAR.HONDA_ACCORD, [eps(ANGLE_FW[2]), eps(SERIAL)]),                      # mixed with an unknown non-logging response
])
def test_inconsistent_angle_fw_is_dashcam_only(candidate, firmware):
  CP = params(candidate, firmware)
  assert CP.dashcamOnly
  assert CP.steerControlType == SteerControlType.torque
  assert not CP.flags & HondaFlags.EPS_ANGLE_LOOP_FW


def test_angle_loop_eps_ignores_logged_diagnostic_response():
  CP = params(CAR.HONDA_ACCORD, [eps(ANGLE_FW[2]), eps(SERIAL, logging=True)])
  assert CP.flags & HondaFlags.EPS_ANGLE_LOOP_FW and not CP.dashcamOnly
  assert CP.to_dict() == accord_cp(ANGLE_FW[2]).to_dict()


def test_logged_angle_loop_string_alone_selects_nothing():
  # a logging-only response is not a firmware image: it neither selects the angle loop nor forces dashcam
  CP = params(CAR.HONDA_ACCORD, [eps(ANGLE_FW[2], logging=True)])
  assert CP.steerControlType == SteerControlType.torque
  assert not CP.flags & HondaFlags.EPS_ANGLE_LOOP_FW and not CP.dashcamOnly


@pytest.mark.parametrize("fw", ANGLE_FW + TORQUE_MOD_FW)
def test_exact_fw_match(fw):
  # one known image per Accord ECU, with the EPS replaced by the modified image
  cohort = {key: versions[0] for key, versions in FW_VERSIONS[CAR.HONDA_ACCORD].items()}
  cohort[(structs.CarParams.Ecu.eps, 0x18da30f1, None)] = fw
  firmware = [structs.CarParams.CarFw(ecu=ecu, address=address, subAddress=0, fwVersion=value, brand="honda")
              for (ecu, address, _), value in cohort.items()]
  exact, matches = match_fw_to_car(firmware, "", allow_exact=True, allow_fuzzy=False, log=False)
  assert exact and matches == {CAR.HONDA_ACCORD}


def test_torque_frame_leaves_the_arm_bits_clear():
  _, dat, _ = create_steering_control(CANPacker(ACCORD_DBC), CanBus(accord_cp()), -1, True, True)
  assert dat[:2] == b"\xff\xff" and (dat[2] >> 2) & 0x3 == 0


def test_angle_frame_differs_from_the_torque_frame_only_in_the_arm_bits():
  packer, CAN = CANPacker(ACCORD_DBC), CanBus(accord_cp())
  for value, active in ((0, False), (123, True), (-4000, True)):
    _, torque, _ = create_steering_control(packer, CAN, value, active, False)
    _, angle, _ = create_steering_control(packer, CAN, value, active, False, angle_arm=P.ANGLE_ARM)
    assert torque[:2] == angle[:2] and (angle[2] >> 2) & 0x3 == P.ANGLE_ARM
    assert torque[2] == angle[2] & ~0x0C & 0xFF and torque[3] == angle[3]


def _controller():
  return CarController({Bus.pt: ACCORD_DBC}, accord_cp(ANGLE_FW[2]))


def _cc_cs(desired, measured, v, lat_active=True):
  CC = SimpleNamespace(actuators=SimpleNamespace(steeringAngleDeg=desired), latActive=lat_active)
  CS = SimpleNamespace(out=SimpleNamespace(steeringAngleDeg=measured, vEgoRaw=v))
  return CC, CS


@pytest.mark.parametrize("active", [False, True])
def test_angle_frame_from_the_controller(active):
  # the real update(): ARM on every frame; the setpoint and the request only while lateral is active
  ci = CarInterface(accord_cp(ANGLE_FW[2]))
  ci.update([])
  ci.CS.out = structs.CarState(vEgo=15.0, vEgoRaw=15.0, steeringAngleDeg=-12.3).as_reader()
  cc = structs.CarControl(latActive=active)
  cc.actuators.steeringAngleDeg = -40.0
  for tick in range(3):
    actuators, frames = ci.apply(cc.as_reader(), 1_000_000_000 + tick * 10_000_000)
    dat = next(frame for frame in frames if frame[0] == STEERING_CONTROL)[1]
    raw = int.from_bytes(dat[0:2], "big", signed=True)
    assert (dat[2] >> 2) & 0x3 == P.ANGLE_ARM   # byte 2 bits 3:2
    assert bool(dat[2] & 0x80) == active        # STEER_TORQUE_REQUEST
    assert raw == actuators.torqueOutputCan == (math.floor(-10.0 * actuators.steeringAngleDeg + 0.5) if active else 0)
    assert actuators.torque == 0.0
  assert (raw > 123) == active  # right of the wheel (-12.3 deg = raw 123) and moving towards -40 deg


def test_rate_limit_then_error_clip():
  cc = _controller()
  v, measured = 20.0, 0.0
  err = float(np.interp(v, P.ANGLE_ERROR_MAX_BP, P.ANGLE_ERROR_MAX_V))
  previous = measured  # the first frame seeds the limiter from the wheel
  # at this speed the lateral jerk limit binds, well below MAX_ANGLE_RATE
  step_max = get_max_angle_delta_vm(v, cc.VM, cc.params)
  assert 0 < step_max < 0.2 * P.ANGLE_LIMITS.MAX_ANGLE_RATE
  for _ in range(200):
    apply_angle, raw = cc._update_angle(*_cc_cs(90.0, measured, v))
    assert 0 <= cc.apply_angle_last - previous <= step_max + 1e-9
    assert raw == math.floor(-10.0 * apply_angle + 0.5)
    previous = cc.apply_angle_last
  assert apply_angle == pytest.approx(measured + err)  # the wheel never moved: the clip holds the send at +err
  assert cc.apply_angle_last > measured + err         # while the limiter state ran on


def test_lateral_accel_limit_caps_the_setpoint():
  # at highway speed the vehicle model accel limit binds well inside the 400 deg setpoint limit
  cc = _controller()
  for _ in range(500):
    cc._update_angle(*_cc_cs(90.0, 0.0, 30.0))
  max_curvature = P.ANGLE_LIMITS.MAX_LATERAL_ACCEL / 30.0 ** 2
  assert cc.apply_angle_last == pytest.approx(math.degrees(cc.VM.get_steer_from_curvature(max_curvature, 30.0, 0)))
  assert cc.apply_angle_last < 20.0


def test_wheel_motion_does_not_feed_the_limiter_state():
  # same command, different wheel paths: while lateral is active the limiter state must not depend on the wheel
  rng = np.random.default_rng(0)
  a, b = _controller(), _controller()
  a.apply_angle_last = b.apply_angle_last = 0.0
  for i in range(300):
    desired = 60.0 * math.sin(i / 20.0)
    a._update_angle(*_cc_cs(desired, 0.0, 12.0))
    b._update_angle(*_cc_cs(desired, float(rng.normal(0.0, 15.0)), 12.0))
    assert a.apply_angle_last == b.apply_angle_last


def test_first_frame_seeds_from_the_wheel():
  apply_angle, _ = _controller()._update_angle(*_cc_cs(30.0, 25.0, 10.0))
  assert 25.0 < apply_angle <= 25.0 + P.ANGLE_LIMITS.MAX_ANGLE_RATE


def test_inactive_limiter_follows_the_wheel_and_sends_zero():
  cc = _controller()
  cc.apply_angle_last = 50.0
  apply_angle, raw = cc._update_angle(*_cc_cs(50.0, -12.34, 15.0, lat_active=False))
  assert apply_angle == cc.apply_angle_last == -12.34
  assert raw == 0  # no idle hold at the wheel angle: the panda would drop the frame


def test_setpoint_limit_outranks_the_error_clip():
  cc = _controller()
  cc.apply_angle_last = 400.0
  # a wheel past the limit: the error clip alone would send 404.6 deg
  assert cc._update_angle(*_cc_cs(400.0, 440.0, 1.0)) == (P.ANGLE_LIMITS.STEER_ANGLE_MAX, -4000)


def _source_setpoint(linear_deg):
  """The source fork's setpoint: the fixed point theta = linear * R(theta) / R(0)."""
  theta = linear_deg
  for _ in range(60):
    theta = linear_deg * float(np.interp(abs(theta), _ACCORD_ANGLE_LOOP_RATIO_BP, _ACCORD_ANGLE_LOOP_RATIO)) / _ACCORD_ANGLE_LOOP_RATIO[0]
  return theta


def test_rack_map_is_the_source_fixed_point():
  linear_bp, angle_bp = inverse = get_honda_vgr_inverse(HondaFlags.EPS_ANGLE_LOOP_FW)
  assert angle_bp[-1] >= P.ANGLE_LIMITS.STEER_ANGLE_MAX and set(_ACCORD_ANGLE_LOOP_RATIO_BP) <= set(angle_bp)
  assert all(b > a for a, b in zip(linear_bp, linear_bp[1:], strict=False))
  for linear in np.arange(0.0, linear_bp[-1], 0.37):
    for signed in (float(linear), -float(linear)):
      physical = vgr_linear_to_physical(signed, inverse)
      assert physical == pytest.approx(_source_setpoint(signed), abs=0.02)
      assert vgr_physical_to_linear(physical, inverse) == pytest.approx(signed, abs=1e-9)
      if abs(signed) <= 31.0:
        assert physical == pytest.approx(signed, abs=1e-12)  # flat on centre: the map is the identity there


def test_no_rack_map_without_the_flag():
  assert get_honda_vgr_inverse(0) is None
  assert vgr_linear_to_physical(123.4, None) == 123.4
  assert vgr_physical_to_linear(-123.4, None) == -123.4


def _steer_fault_temporary(CP, status):
  ci = CarInterface(CP)
  ci.update([])  # the parser subscribes to a message on its first read
  values = {f"STEER_SENSOR_STATUS_{i}": bit for i, bit in zip((1, 2, 3), status, strict=True)}
  return ci.update([(1, [CANPacker(ACCORD_DBC).make_can_msg("STEERING_SENSORS", CanBus(CP).pt, values)])]).steerFaultTemporary


@pytest.mark.parametrize("status, fault", [((1, 1, 1), False), ((0, 1, 1), True), ((1, 0, 1), True), ((1, 1, 0), True)])
def test_steer_sensor_status_fault(status, fault):
  assert _steer_fault_temporary(accord_cp(ANGLE_FW[2]), status) == fault
  assert not _steer_fault_temporary(accord_cp(), status)  # only the angle-loop EPS servoes from this angle
