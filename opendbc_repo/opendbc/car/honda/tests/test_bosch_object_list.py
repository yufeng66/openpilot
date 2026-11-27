"""dp - the stock camera's object list as radar points (dp_honda_bosch_radar): decoder, gate, stale path, cadence.

Most frames are built by a small encoder from the signal layout. One test pins the decoder to real data: three sweeps
of raw object-list frames from our Accord drive (bosch_object_list_route166.json, bus 2 only) with the points that the
validated offline port (lib.parse_points, 'new' constants) gives for them.
"""
import json
import math
import os

import pytest

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.honda import bosch_object_list as bol
from opendbc.car.honda.hondacan import CanBus, honda_checksum
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.radar_interface import HONDA_BOSCH_OBJECT_LIST_CARS
from opendbc.car.honda.values import CAR, HONDA_BOSCH

FLAG = structs.DPFlags.HondaBoschRadar
DT_NS = 10_000_000  # one card call
SWEEP_CALLS = 7  # about 14.4 sweeps per second


def car_params(candidate=CAR.HONDA_ACCORD, dp_params=FLAG, alpha_long=False):
  return CarInterface.get_params(candidate, gen_empty_fingerprint(), [], alpha_long, False, dp_params, False)


class Encoder:
  """Builds valid frames (Honda checksum, 2-bit counter per address) of the object list."""

  def __init__(self):
    self.counters = {}

  def frame(self, address, fields, checksum_ok=True, counter=None):
    x = 0
    for (shift, mask), value in fields:
      x |= (int(value) & mask) << shift
    dat = bytearray(x.to_bytes(8, 'big'))
    if counter is None:
      counter = (self.counters.get(address, 0) + 1) & 0x3
      self.counters[address] = counter
    dat[7] = (dat[7] & 0xC0) | (counter << 4)
    dat[7] |= honda_checksum(address, None, dat)
    if not checksum_ok:
      dat[7] ^= 0x1
    return address, bytes(dat), 2

  def slot(self, slot, idx, track_id, range_raw, az_raw=1024, life=100, status=1, rsig=0, exist=100, u11=864, u10=0, ratio_raw=0x3FF,
           aux=True, idx_f3=None):
    a0, a1, a2, a3 = bol.MAIN_ADDRS[slot]
    frames = [
      self.frame(a0, [(bol.F0_IDX, idx), (bol.F0_STATUS, status), (bol.F0_RANGE, range_raw), (bol.F0_AZIMUTH, az_raw), (bol.F0_RANGE_SIGMA, rsig)]),
      self.frame(a1, [(bol.F1_IDX, idx), (bol.F1_EXISTENCE, exist)]),
      self.frame(a2, [(bol.F2_IDX, idx), (bol.F2_LIFE, life)]),
      self.frame(a3, [(bol.F3_IDX, idx if idx_f3 is None else idx_f3), (bol.F3_TRACK_ID, track_id)]),
    ]
    if aux:
      frames.append(self.frame(bol.AUX_ADDRS[slot], [(bol.AUX_IDX, idx), (bol.AUX_VREL, u11), (bol.AUX_VREL_UNCERTAINTY, u10),
                                                     (bol.AUX_RANGE_RATIO, ratio_raw)]))
    return frames

  def sweep(self, objects, start=True, trigger=True):
    """objects: list of slot() kwargs. Empty slots are sent with an invalid status, as the camera does."""
    by_slot = {o['slot']: o for o in objects}
    frames = []
    for s in range(bol.NUM_SLOTS):
      if s in by_slot:
        frames += self.slot(**by_slot[s])
      else:
        frames += self.slot(s, idx=0, track_id=0, range_raw=0xFFF, az_raw=0x7FF, life=0xFFF, status=0xF, u11=0x7FE, u10=0x3FF)
    if not start:
      frames = [f for f in frames if f[0] != bol.SWEEP_START_ADDR]
    if not trigger:
      frames = [f for f in frames if f[0] != bol.SWEEP_TRIGGER_ADDR]
    return frames


class Drive:
  """Feeds an object-list parser (or a RadarInterface) like card: one CAN packet per call, 10 ms apart."""

  def __init__(self, target):
    self.target = target
    self.enc = Encoder()
    self.t = 1_000_000_000
    self.out = []

  def call(self, frames=()):
    self.t += DT_NS
    r = self.target.update([(self.t, list(frames))])
    self.out.append(r)
    return r

  def sweep(self, objects, **kw):
    r = [self.call(self.enc.sweep(objects, **kw))]
    for _ in range(SWEEP_CALLS - 1):
      r.append(self.call())
    return r


def lead(step, slot=0, track_id=5, range_raw=800, life0=100, u11=864 - 72, **kw):
  # an object that keeps its incarnation: frame index +1, lifecycle +2 per sweep
  return dict(slot=slot, idx=step & 0xF, track_id=track_id, range_raw=range_raw, life=(life0 + 2 * step) & 0xFFF, u11=u11, **kw)


class TestDecoder:
  def test_scales_and_maturity(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    d.sweep([lead(0, az_raw=1024 + 64)])
    assert ol.points() == []  # a birth observation is not published
    d.sweep([lead(1, az_raw=1024 + 64)])
    (p,) = ol.points()
    d_rel = 800 / 16 - 346 / 128
    assert p.dRel == pytest.approx(d_rel, abs=1e-4)
    assert p.yRel == pytest.approx(d_rel * math.tan(64 / 2048), abs=1e-4)  # left positive
    assert p.vRel == pytest.approx(-1.0, abs=1e-6)  # (792 - 864) / 72
    assert p.measured
    assert math.isnan(p.aRel) and math.isnan(p.yvRel)

  def test_track_id_kept_and_renewed(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(3):
      d.sweep([lead(step)])
    first = ol.points()[0].trackId
    d.sweep([lead(3)])
    assert ol.points()[0].trackId == first
    # a lifecycle jump is a new object behind the same Bosch id: it must not inherit the old trackId
    d.sweep([lead(4, life0=900)])
    assert ol.points() == []
    d.sweep([lead(5, life0=900)])
    assert ol.points()[0].trackId != first

  def test_bad_checksum_drops_the_frame(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    d.sweep([lead(0)])
    frames = d.enc.sweep([lead(1)])
    a1 = bol.MAIN_ADDRS[0][1]
    frames = [d.enc.frame(a1, [(bol.F1_IDX, 1), (bol.F1_EXISTENCE, 100)], checksum_ok=False) if f[0] == a1 else f for f in frames]
    d.call(frames)
    assert ol.points() == []  # slot 0 incomplete, no second sample yet

  def test_stuck_counter_drops_frames(self):
    ol = bol.BoschObjectList(2)
    enc = Encoder()
    addr = bol.MAIN_ADDRS[0][0]
    ok = [ol._frame_ok(addr, enc.frame(addr, [], counter=2)[1]) for _ in range(8)]
    assert ok[:4] == [True] * 4 and not any(ok[4:])

  def test_wrong_length_and_other_bus_ignored(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    d.sweep([lead(0)])
    frames = d.enc.sweep([lead(1)])
    d.call([(a, dat, 0) for a, dat, _ in frames] + [(a, dat[:7], 2) for a, dat, _ in frames])
    assert ol.points() == []

  def test_incoherent_frame_index(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    d.sweep([lead(0)])
    d.sweep([lead(1, idx_f3=9)])
    assert ol.points() == []

  def test_invalid_status_hides_the_object(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    d.sweep([lead(0)])
    d.sweep([lead(1)])
    assert len(ol.points()) == 1
    d.sweep([lead(2, status=0xF)])
    assert ol.points() == []

  def test_range_jump_is_not_published(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    d.sweep([lead(0)])
    d.sweep([lead(1)])
    before = ol.points()[0].dRel
    d.sweep([lead(2, range_raw=800 + 16 * 20)])  # +20 m in one sweep, the speed field says -1 m/s
    (p,) = ol.points()
    assert p.dRel == pytest.approx(before) and not p.measured

  @pytest.mark.parametrize("rail", [0, bol.VREL_MAX_RAW])
  def test_speed_at_the_rail_is_not_published(self, rail):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(2):
      d.sweep([lead(step)])
    (p,) = ol.points()
    first_id = p.trackId
    for step in range(2, 4):  # the speed field sits at its rail: the true speed is unknown
      d.sweep([lead(step, u11=rail)])
      assert ol.points() == []
    d.sweep([lead(4, u11=rail + (1 if rail == 0 else -1))])  # one count inside the rail: published again, as a new track
    (p,) = ol.points()
    assert abs(p.vRel) == pytest.approx(12.0 - 1 / 72, abs=1e-6)
    assert p.trackId != first_id

  def test_no_trigger_no_update(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    d.sweep([lead(0)])
    d.sweep([lead(1)], trigger=False)
    assert ol.points() == []

  def test_two_objects_one_id_lowest_slot_wins(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(2):
      d.sweep([lead(step, slot=3), lead(step, slot=7, range_raw=1600)])
    (p,) = ol.points()
    assert p.dRel == pytest.approx(800 / 16 - 346 / 128, abs=1e-4)

  def test_late_companion_frame_of_slot_15(self):
    # 0x297 (speed of slot 15) one CAN packet after 0x2FF: the sweep waits for it, the sweep time stays the 0x2FF time
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(3):
      frames = d.enc.sweep([lead(step, slot=15)])
      d.call([f for f in frames if f[0] != bol.SWEEP_END_ADDR])
      t_trigger = d.t
      assert ol._last_sweep_nanos != t_trigger
      d.call([f for f in frames if f[0] == bol.SWEEP_END_ADDR])
      assert ol._last_sweep_nanos == t_trigger
      for _ in range(SWEEP_CALLS - 2):
        d.call()
    (p,) = ol.points()
    assert p.vRel == pytest.approx(-1.0, abs=1e-6) and p.measured

  def test_companion_frame_never_comes(self):
    # without 0x297 the sweep goes at the end of the next call; slot 15 then has no speed, the other slots are complete
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(2):
      frames = [f for f in d.enc.sweep([lead(step, slot=3), lead(step, slot=15, track_id=9, range_raw=1200)]) if f[0] != bol.SWEEP_END_ADDR]
      d.call(frames)
      t_trigger = d.t
      assert ol._last_sweep_nanos != t_trigger
      d.call()
      assert ol._last_sweep_nanos == t_trigger
      for _ in range(SWEEP_CALLS - 2):
        d.call()
    (p,) = ol.points()
    assert p.dRel == pytest.approx(800 / 16 - 346 / 128, abs=1e-4) and p.vRel == pytest.approx(-1.0, abs=1e-6)

  def test_old_frames_are_not_used_again(self):
    # 0x280 is lost and only the end of a sweep arrives: the object's frames of the last sweep must not count again
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(2):
      d.sweep([lead(step, slot=2)])
    assert len(ol.points()) == 1
    for _ in range(8):  # 0.56 s of sweep ends only
      d.call([f for f in d.enc.sweep([]) if f[0] in (bol.SWEEP_TRIGGER_ADDR, bol.SWEEP_END_ADDR)])
      for _ in range(SWEEP_CALLS - 1):
        d.call()
    assert not ol.stale()
    assert ol.points() == []  # the object was not seen for more than 0.2 s

  def test_time_that_does_not_advance_restarts_the_object(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(2):
      d.sweep([lead(step)])
    first = ol.points()[0].trackId
    d.t -= 500_000_000  # the packet time goes back
    d.sweep([lead(2)])
    assert ol.points() == []  # no old point kept while time is behind
    d.sweep([lead(3)])
    (p,) = ol.points()
    assert p.measured and p.trackId != first

  @pytest.mark.parametrize("rail", [0, bol.VREL_MAX_RAW])
  def test_rail_on_the_coast_path_is_not_published(self, rail):
    # speed field at the rail with a high uncertainty: the parser coasts the last trusted speed, but must not publish it
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(2):
      d.sweep([lead(step)])
    assert len(ol.points()) == 1
    d.sweep([lead(2, u11=rail, u10=bol.VREL_MAX_UNCERTAINTY + 100)])
    assert ol.points() == []
    d.sweep([lead(3)])
    (p,) = ol.points()
    assert p.vRel == pytest.approx(-1.0, abs=1e-6)

  @pytest.mark.parametrize("rail", [0, bol.VREL_MAX_RAW])
  def test_no_coast_of_an_older_speed_after_a_rail_reading(self, rail):
    # the sweep after a rail reading loses its companion frame: the speed trusted before the rail must not come back
    near_rail = 36 if rail == 0 else bol.VREL_MAX_RAW - 36  # 11.5 m/s
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(2):
      d.sweep([lead(step, u11=near_rail)])
    assert len(ol.points()) == 1
    d.sweep([lead(2, u11=rail, u10=bol.VREL_MAX_UNCERTAINTY + 100)])
    assert ol.points() == []
    d.sweep([lead(3, aux=False)])
    assert ol.points() == []

  def test_time_going_back_drops_an_object_that_is_no_longer_sent(self):
    # replay only: after a jump back in time, an object whose slot goes silent must not stay published
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(3):
      d.sweep([lead(step, slot=1)])
    assert len(ol.points()) == 1
    d.t -= 5_000_000_000
    silent = set(bol.MAIN_ADDRS[1]) | {bol.AUX_ADDRS[1]}
    for _ in range(3):
      d.call([f for f in d.enc.sweep([]) if f[0] not in silent])
      for _ in range(SWEEP_CALLS - 1):
        d.call()
      assert ol.points() == []

  @pytest.mark.parametrize("range_raw, published", [(0, False), (20, False), (43, False), (44, True)])
  def test_no_point_at_or_behind_the_bumper(self, range_raw, published):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(2):
      d.sweep([lead(step, range_raw=range_raw, u11=864)])
    assert (len(ol.points()) == 1) == published


# lib.parse_points(..., 'new') on the same three sweeps from a fresh state, sorted by dRel: (dRel, yRel, vRel)
ROUTE166_EXPECTED = [
  [],
  [(21.234375, -0.9753096140479811, 0.8055555555555555), (72.109375, -10.927078944832816, -6.027777777777778),
   (92.296875, -19.85714483818941, -0.8055555555555555), (114.046875, 28.410190890942825, 3.125)],
  [(21.234375, -0.9960909938497244, 0.625), (71.859375, -11.032811113154617, -5.888888888888888),
   (92.234375, -19.985105730650695, -0.7222222222222222), (113.796875, 28.761358062274386, 3.0277777777777777)],
]


def test_real_sweeps_route166():
  with open(os.path.join(os.path.dirname(__file__), 'bosch_object_list_route166.json')) as f:
    packets = json.load(f)['packets']
  ol = bol.BoschObjectList(2)
  got = []
  last = None
  for t, frames in packets:
    ol.update([(1_000_000_000 + t, [(addr, bytes.fromhex(dat), 2) for addr, dat in frames])])
    if ol._last_sweep_nanos != last:
      last = ol._last_sweep_nanos
      got.append(sorted(((p.dRel, p.yRel, p.vRel) for p in ol.points()), key=lambda v: v[0]))
  assert len(got) == len(ROUTE166_EXPECTED)
  for sweep, expected in zip(got, ROUTE166_EXPECTED, strict=True):
    assert len(sweep) == len(expected)
    for p, e in zip(sweep, expected, strict=True):
      assert p == pytest.approx(e, abs=1e-4)  # float32 in RadarPoint


class TestStale:
  def test_list_stops_points_go_empty(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(3):
      d.sweep([lead(step)])
    assert len(ol.points()) == 1
    for _ in range(bol.STALE_CALLS - SWEEP_CALLS):
      d.call()
    assert len(ol.points()) == 1  # within 0.2 s of the last sweep: still the latest points
    for _ in range(SWEEP_CALLS + 1):
      d.call()
    assert ol.points() == []
    # the list comes back: a fresh start, no history from before the gap
    d.sweep([lead(3)])
    assert ol.points() == []
    d.sweep([lead(4)])
    assert len(ol.points()) == 1

  def test_can_stream_stops_points_go_empty(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(3):
      d.sweep([lead(step)])
    for _ in range(bol.STALE_CALLS + 1):
      ol.update([])  # no CAN at all: card still calls once per loop
    assert ol.points() == []

  def test_time_gap_inside_one_call(self):
    ol = bol.BoschObjectList(2)
    d = Drive(ol)
    for step in range(3):
      d.sweep([lead(step)])
    d.t += 300_000_000  # a late packet, 0.3 s after the last sweep, without the list
    d.call([(0x1A6, bytes(8), 0)])
    assert ol.points() == []


class TestRadarInterface:
  def ri(self):
    CP = car_params()
    assert not CP.radarUnavailable
    RI = CarInterface.RadarInterface(CP)
    assert RI.object_list is not None and RI.object_list.bus == CanBus(CP).camera and RI.rcp is None
    return RI

  def test_cadence_and_no_error_flags(self):
    d = Drive(self.ri())
    for step in range(30):
      d.sweep([lead(step)])
    for _ in range(100):  # list stops
      d.call()
    for step in range(30, 40):
      d.sweep([lead(step)])
    garbage_start = len(d.out)
    for _ in range(50):  # garbage on the list addresses
      d.call([(a, bytes([0xFF] * 8), 2) for a in sorted(bol.ALL_ADDRS)] + [(0x280, b'\x01', 2)])
    pubs = [(i, r) for i, r in enumerate(d.out) if r is not None]
    assert [i for i, _ in pubs] == list(range(4, len(d.out), 5))  # every 5th call, 20 Hz, whatever the list does
    assert not any(any(r.errors.to_dict().values()) for _, r in pubs)
    with_pts = [i for i, r in pubs if len(r.points)]
    assert with_pts and max(with_pts) < garbage_start + bol.STALE_CALLS  # garbage gives no points: the last ones expire
    stop = 30 * SWEEP_CALLS
    assert all(len(r.points) == 0 for i, r in pubs if stop + bol.STALE_CALLS < i < stop + 100)

  def test_exception_is_contained(self):
    RI = self.ri()

    def boom(_):
      raise ValueError('malformed')
    RI.object_list.update = boom
    out = [RI.update([(i * DT_NS, [])]) for i in range(20)]
    pubs = [r for r in out if r is not None]
    assert len(pubs) == 4 and all(len(r.points) == 0 and not any(r.errors.to_dict().values()) for r in pubs)


class TestGate:
  @pytest.mark.parametrize('candidate', sorted(HONDA_BOSCH))
  @pytest.mark.parametrize('flag', [0, FLAG])
  @pytest.mark.parametrize('alpha_long', [False, True])
  def test_bosch(self, candidate, flag, alpha_long):
    CP = car_params(candidate, flag, alpha_long)
    on = bool(flag) and candidate in HONDA_BOSCH_OBJECT_LIST_CARS and not CP.openpilotLongitudinalControl
    assert CP.radarUnavailable == (not on)
    RI = CarInterface.RadarInterface(CP)
    assert (RI.object_list is not None) == on
    if not on:
      assert RI.rcp is None and RI.update([]) is None  # unchanged: the base 20 Hz empty publication

  def test_only_accord(self):
    assert HONDA_BOSCH_OBJECT_LIST_CARS == {CAR.HONDA_ACCORD}

  def test_nidec_unchanged(self):
    for flag in (0, FLAG):
      CP = car_params(CAR.HONDA_PILOT, flag)
      assert not CP.radarUnavailable
      RI = CarInterface.RadarInterface(CP)
      assert RI.object_list is None and RI.rcp is not None

  def test_flag_changes_only_radar_unavailable(self):
    off = car_params(dp_params=0).to_dict()
    on = car_params(dp_params=FLAG).to_dict()
    assert {k for k in off if off[k] != on[k]} == {'radarUnavailable'}
