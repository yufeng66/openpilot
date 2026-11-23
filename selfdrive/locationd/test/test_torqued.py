from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR as HONDA
from openpilot.common.params import Params
from openpilot.selfdrive.locationd.torqued import TorqueEstimator, MIN_VEL


def test_cal_percent():
  est = TorqueEstimator(car.CarParams())
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 0

  for (low, high), min_pts in zip(est.filtered_points.buckets.keys(),
                                  est.filtered_points.buckets_min_points.values(), strict=True):
    for _ in range(int(min_pts)):
      est.filtered_points.add_point((low + high) / 2.0, 0.0)

  # enough bucket points, but not enough total points
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == (len(est.filtered_points) / est.min_points_total * 100 + 100) / 2

  # add enough points to bucket with most capacity
  key = list(est.filtered_points.buckets)[0]
  for _ in range(est.min_points_total - len(est.filtered_points)):
    est.filtered_points.add_point((key[0] + key[1]) / 2.0, 0.0)

  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 100


def _accord_cp(eps_fw):
  car_fw = [car.CarParams.CarFw(ecu=car.CarParams.Ecu.eps, fwVersion=eps_fw, address=0x18DA30F1, subAddress=0)]
  fingerprint = {i: {} for i in range(8)}
  return interfaces[HONDA.HONDA_ACCORD].get_params(HONDA.HONDA_ACCORD, fingerprint, car_fw, False, False, 0, False).as_reader()


def test_angle_loop_accord_is_idle():
  # the same Accord on the stock EPS (torque tune, learner on) and on the angle-loop EPS firmware
  torque_CP = _accord_cp(b"39990-TVA-A160\x00\x00")
  angle_CP = _accord_cp(b"39990-TVA,A16C\x00\x00")
  assert torque_CP.lateralTuning.which() == 'torque'
  assert angle_CP.steerControlType == car.CarParams.SteerControlType.angle

  params = Params()
  params.put_bool("dp_lat_torqued_sd", True)
  cache = TorqueEstimator(torque_CP).get_msg(with_points=True).to_bytes()
  params.put("LiveTorqueParameters", cache)
  params.put("CarParamsPrevRoute", torque_CP.as_builder().to_bytes())

  est = TorqueEstimator(angle_CP)
  assert not est.use_params and not est.speed_binned
  assert params.get("LiveTorqueParameters") == cache  # a restore-key mismatch does not delete the cache at start

  # an engaged angle-mode drive: the carcontroller reports zero torque, so nothing is learned
  for i in range(200):
    t = i * 0.05
    est.handle_log(t, "carControl", car.CarControl(latActive=True))
    est.handle_log(t, "carOutput", car.CarOutput(actuatorsOutput=car.CarControl.Actuators(torque=0.0, steeringAngleDeg=5.0)))
    est.handle_log(t, "carState", car.CarState(vEgo=MIN_VEL + 5.0, steeringPressed=False))
    est.handle_log(t, "livePose", log.LivePose(timestamp=int(t * 1e9), inputsOK=True, sensorsOK=True, posenetOK=True,
                                               angularVelocityDevice=log.LivePose.XYZMeasurement(z=0.02, valid=True),
                                               orientationNED=log.LivePose.XYZMeasurement(valid=True)))
  assert len(est.filtered_points) == 0
  msg = est.get_msg()
  assert not msg.liveTorqueParameters.useParams and not msg.liveTorqueParameters.liveValid
