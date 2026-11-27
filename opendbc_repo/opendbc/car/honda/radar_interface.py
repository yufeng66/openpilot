#!/usr/bin/env python3
from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.carlog import carlog
from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car.honda.bosch_object_list import BoschObjectList
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.values import CAR, DBC

# dp - Bosch platforms where the stock camera's object list may serve as radar points (dp_honda_bosch_radar).
# Verified on our own logs (HONDA_ACCORD, 2026-11-26) with the current scale constants. The community also verified
# Civic Bosch and CR-V 5G, but with the older scale; add them only after a check on their logs.
HONDA_BOSCH_OBJECT_LIST_CARS = {CAR.HONDA_ACCORD}


def _create_nidec_can_parser(car_fingerprint):
  radar_messages = [0x400] + list(range(0x430, 0x43A)) + list(range(0x440, 0x446))
  messages = [(m, 20) for m in radar_messages]
  return CANParser(DBC[car_fingerprint][Bus.radar], messages, 1)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.track_id = 0
    self.radar_fault = False
    self.radar_wrong_config = False
    self.radar_off_can = CP.radarUnavailable

    # dp - Bosch: the camera object list on the camera bus, read only (the interface clears radarUnavailable only
    # behind the toggle). The list is optional: no error flags, and points are published at a fixed 20 Hz.
    self.object_list = None
    self._object_list_error_frame = None
    if not self.radar_off_can and CP.carFingerprint in HONDA_BOSCH_OBJECT_LIST_CARS:
      self.object_list = BoschObjectList(CanBus(CP).camera)

    # Nidec
    if self.radar_off_can or self.object_list is not None:  # dp
      self.rcp = None
    else:
      self.rcp = _create_nidec_can_parser(CP.carFingerprint)
    self.trigger_msg = 0x445
    self.updated_messages = set()

  def update(self, can_strings):
    # dp
    if self.object_list is not None:
      return self._update_object_list(can_strings)

    # in Bosch radar and we are only steering for now, so sleep 0.05s to keep
    # radard at 20Hz and return no points
    if self.radar_off_can:
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    rr = self._update(self.updated_messages)
    self.updated_messages.clear()
    return rr

  def _update_object_list(self, can_strings):
    # dp - consume every call; publish the latest points at a fixed 20 Hz (like the base class), so that the liveTracks
    # alive and frequency checks of radard hold even when the list is stale (no points then) or absent
    self.frame += 1
    try:
      self.object_list.update(can_strings)
    except Exception:
      # a malformed list must not take card down: drop the points, start over
      if self._object_list_error_frame is None or self.frame - self._object_list_error_frame > 1000:
        carlog.exception("Honda Bosch object list: update failed, points dropped")
        self._object_list_error_frame = self.frame
      self.object_list.reset()
    if self.frame % 5 != 0:
      return None
    ret = structs.RadarData()
    ret.points = self.object_list.points()
    return ret

  def _update(self, updated_messages):
    ret = structs.RadarData()

    for ii in sorted(updated_messages):
      cpt = self.rcp.vl[ii]
      if ii == 0x400:
        # check for radar faults
        self.radar_fault = cpt['RADAR_STATE'] != 0x79
        self.radar_wrong_config = cpt['RADAR_STATE'] == 0x69
      elif cpt['LONG_DIST'] < 255:
        if ii not in self.pts or cpt['NEW_TRACK']:
          self.pts[ii] = structs.RadarData.RadarPoint()
          self.pts[ii].trackId = self.track_id
          self.track_id += 1
        self.pts[ii].dRel = cpt['LONG_DIST']  # from front of car
        self.pts[ii].yRel = -cpt['LAT_DIST']  # in car frame's y axis, left is positive
        self.pts[ii].vRel = cpt['REL_SPEED']
        self.pts[ii].aRel = float('nan')
        self.pts[ii].yvRel = float('nan')
        self.pts[ii].measured = True
      else:
        if ii in self.pts:
          del self.pts[ii]

    if not self.rcp.can_valid:
      ret.errors.canError = True
    if self.radar_fault:
      ret.errors.radarFault = True
    if self.radar_wrong_config:
      ret.errors.wrongConfig = True

    ret.points = list(self.pts.values())

    return ret
