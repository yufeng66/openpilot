# dp - optional radar source for Honda Bosch: the object list that the stock camera sends to the radar unit.
#
# The camera publishes a 16-slot object bank on the camera bus (CanBus(CP).camera): per slot four main frames
# (0x280-0x28F for slots 0-3, 0x2D0-0x2FF for slots 4-15) and one companion frame (0x2C8-0x2CF for slots 0-7,
# 0x290-0x297 for slots 8-15). One sweep runs 0x280 ... 0x2FF, 0x297 at about 14.4 Hz. Every frame carries the
# standard Honda checksum and 2-bit counter in byte 7.
#
# Origin: the community parser by JamesL787 in the StarPilot fork (firestar5683/StarPilot, honda radar_interface.py,
# _update_bosch_a) and the research branch of trung791997 (ns-bosch-radar-testing, which corrected the scale
# constants: range = raw/16 - 346/128 m, relative speed 1/72 m/s per count). This is a re-implementation of that
# logic, validated against our own Accord drive (route 00000166, 2026-11-26) with a replay through radard. The later
# rules of the research branch (rail interval, re-anchor, coast bounds, newborn publish) are not ported. Our own rules:
# a Bosch id whose speed field is at its rail (+-12 m/s) in a sweep is not published in that sweep, on any path; a point
# at or behind the bumper (dRel <= 0) is not published; a sweep time that does not advance restarts the object.
#
# Known limits (not changed):
# - the range-ratio speed divides by the CAN packet time difference, which jitters by about -24/+15 % (this speed was
#   used for 1 observation in the 2026-11-26 drive);
# - after a range correction of more than 5 m an object is lost for the rest of its incarnation (seen only at 94 m or
#   more in that drive).
#
# Fail soft: this source is optional. A missing, stale or malformed list only removes the points; it never sets a
# RadarData error flag (each flag becomes a selfdrived event that blocks or ends engagement).
import math

from opendbc.car import structs
from opendbc.car.honda.hondacan import honda_checksum

NUM_SLOTS = 16


def _main_base(slot: int) -> int:
  return 0x280 + 4 * slot if slot < 4 else 0x2D0 + 4 * (slot - 4)


def _aux_addr(slot: int) -> int:
  return 0x2C8 + slot if slot < 8 else 0x290 + (slot - 8)


MAIN_ADDRS = [[_main_base(s) + i for i in range(4)] for s in range(NUM_SLOTS)]
AUX_ADDRS = [_aux_addr(s) for s in range(NUM_SLOTS)]
ALL_ADDRS = frozenset(a for addrs in MAIN_ADDRS for a in addrs) | frozenset(AUX_ADDRS)
SWEEP_START_ADDR = MAIN_ADDRS[0][0]  # 0x280
SWEEP_TRIGGER_ADDR = MAIN_ADDRS[NUM_SLOTS - 1][3]  # 0x2FF, last main frame of a sweep; its packet time is the sweep time
# 0x297 (companion of slot 15) follows 0x2FF; in 2.7 % of the sweeps of the 2026-11-26 drive it came one CAN packet later
SWEEP_END_ADDR = AUX_ADDRS[NUM_SLOTS - 1]


def _field(start_bit: int, size: int) -> tuple[int, int]:
  # Motorola (big endian) DBC signal "start_bit|size@0+" -> (shift, mask) on the frame read as one big-endian integer
  msb = (7 - start_bit // 8) * 8 + start_bit % 8
  return msb - size + 1, (1 << size) - 1


# frame 0 (f0)
F0_IDX = _field(27, 4)
F0_STATUS = _field(15, 4)
F0_RANGE = _field(23, 12)
F0_AZIMUTH = _field(39, 11)
F0_RANGE_SIGMA = _field(7, 7)
# frame 1
F1_IDX = _field(28, 4)
F1_EXISTENCE = _field(46, 7)
# frame 2
F2_IDX = _field(11, 4)
F2_LIFE = _field(7, 12)
# frame 3
F3_IDX = _field(12, 4)
F3_TRACK_ID = _field(55, 8)
# companion frame
AUX_IDX = _field(12, 4)
AUX_VREL = _field(7, 11)
AUX_VREL_UNCERTAINTY = _field(23, 10)
AUX_RANGE_RATIO = _field(55, 10)

STATUS_INVALID = 0xF
RANGE_INVALID = 0xFFF
AZIMUTH_INVALID = 0x7FF
LIFE_INVALID = 0xFFF
TRACK_ID_MIN, TRACK_ID_MAX = 1, 0x3F

RANGE_SCALE_M = 1 / 16
RANGE_OFFSET_M = -346 / 128
AZIMUTH_CENTER = 1024
AZIMUTH_SCALE_RAD = 1 / 2048

VREL_INVALID = 0x7FE
VREL_MAX_RAW = 1728  # the field rails at 0 and 1728 (+-12 m/s): a faster closing speed reads as 12 m/s
VREL_CENTER = 864
VREL_SCALE_MPS = 1 / 72
VREL_RAIL_MPS = (VREL_MAX_RAW - VREL_CENTER) * VREL_SCALE_MPS - 1e-6  # a speed at or beyond this is not published
VREL_MAX_UNCERTAINTY = 511

RANGE_RATIO_INVALID = 0x3FF
RANGE_RATIO_OFFSET = 0.5
RANGE_RATIO_SCALE = 0.001

RANGE_SIGMA_DEGRADED = 4
INNOVATION_MAX_M = 2.0
INNOVATION_HARD_MAX_M = 5.0
FALLBACK_RANGE_RATE_MAX_MPS = 50.0

STALE_S = 0.20  # about three missed sweeps: a track (or the whole list) not seen for longer is dropped
STALE_CALLS = 20  # the same 0.2 s counted in update() calls (card runs at 100 Hz), for a CAN stream that stops
MAX_BAD_COUNTER = 5  # same rule as CANParser: a frame is dropped once this many counter errors accumulate


def _direct_vrel(raw: int | None, uncertainty: int | None = None) -> float | None:
  if raw is None or raw == VREL_INVALID or not 0 <= raw <= VREL_MAX_RAW:
    return None
  if uncertainty is not None and uncertainty > VREL_MAX_UNCERTAINTY:
    return None
  return (raw - VREL_CENTER) * VREL_SCALE_MPS


def _range_ratio(raw: int | None) -> float | None:
  if raw is None or not 0 <= raw < RANGE_RATIO_INVALID:
    return None
  return RANGE_RATIO_OFFSET + RANGE_RATIO_SCALE * raw


class _Track:
  __slots__ = ('prev_idx', 'prev_life', 'last_seen', 'n_samples', 'last_sample', 'vtrust', 'vtrust_t')

  def __init__(self):
    self.prev_idx: int | None = None
    self.prev_life: int | None = None
    self.last_seen: float | None = None
    self.n_samples = 0  # accepted ranges of this incarnation (only "at least two" matters)
    self.last_sample: tuple[float, float] | None = None  # (time, range) of the last accepted range
    self.vtrust: float | None = None
    self.vtrust_t: float | None = None

  def reset_history(self):
    self.n_samples = 0
    self.last_sample = None
    self.vtrust = self.vtrust_t = None


class _Point:
  __slots__ = ('dRel', 'yRel', 'vRel', 'measured')

  def __init__(self):
    self.dRel = self.yRel = self.vRel = math.nan
    self.measured = True


class BoschObjectList:
  """Read-only parser of the camera object list. update() consumes CAN, points() gives the latest RadarPoints."""

  def __init__(self, bus: int):
    self.bus = bus
    self._ctr: dict[int, int] = {}
    self._ctr_fail: dict[int, int] = {}
    self._next_track_id = 0
    self.reset()

  def reset(self):
    self._sweep: dict[int, int] = {}
    self._sweep_open = False
    self._sweep_pending = False
    self._sweep_pending_calls = 0  # update() calls ended since 0x2FF arrived without 0x297
    self._sweep_nanos = 0
    self._tracks: dict[int, _Track] = {}
    self._pts: dict[int, _Point] = {}  # keyed by the Bosch track id
    self._slot_ids: list[int | None] = [None] * NUM_SLOTS
    self._track_ids: dict[int, int] = {}  # Bosch track id -> published trackId, for the points of the last sweep
    self._points: list[structs.RadarData.RadarPoint] = []
    self._last_sweep_nanos: int | None = None
    self._last_nanos: int | None = None
    self._calls_since_sweep = 0

  def _frame_ok(self, address: int, dat: bytes) -> bool:
    if len(dat) != 8 or honda_checksum(address, None, dat) != (dat[7] & 0xF):
      return False
    counter = (dat[7] >> 4) & 0x3
    fail = self._ctr_fail.get(address, 0)
    if ((self._ctr.get(address, 0) + 1) & 0x3) != counter:
      fail = min(fail + 1, MAX_BAD_COUNTER)
    elif fail > 0:
      fail -= 1
    self._ctr[address] = counter
    self._ctr_fail[address] = fail
    return fail < MAX_BAD_COUNTER

  def update(self, can_packets) -> None:
    self._calls_since_sweep += 1
    bus = self.bus
    for nanos, frames in can_packets:
      self._last_nanos = nanos
      for address, dat, src in frames:
        if src != bus or address not in ALL_ADDRS or not self._frame_ok(address, dat):
          continue
        if address == SWEEP_START_ADDR:
          if self._sweep_pending:
            self._process_sweep()
          self._sweep = {}
          self._sweep_open = True
        elif not self._sweep_open:
          continue  # wait for the first sweep start
        if address == SWEEP_TRIGGER_ADDR:
          if self._sweep_pending:
            self._process_sweep()  # a second 0x2FF without 0x297 or 0x280 in between
          self._sweep_pending = True
          self._sweep_pending_calls = 0
          self._sweep_nanos = nanos
        self._sweep[address] = int.from_bytes(dat, 'big')
        if address == SWEEP_END_ADDR and self._sweep_pending:
          self._process_sweep()
    # 0x297 is waited for until the end of the next call after 0x2FF, then the sweep goes without it
    if self._sweep_pending:
      if self._sweep_pending_calls >= 1:
        self._process_sweep()
      else:
        self._sweep_pending_calls += 1

  def stale(self) -> bool:
    if self._last_sweep_nanos is None:
      return True
    if self._calls_since_sweep > STALE_CALLS:
      return True
    return self._last_nanos is not None and (self._last_nanos - self._last_sweep_nanos) * 1e-9 > STALE_S

  def points(self) -> list[structs.RadarData.RadarPoint]:
    """Latest points, or none (and a clean state) when the list is stale."""
    if self.stale():
      if self._last_sweep_nanos is not None:
        self.reset()
      return []
    return self._points

  def _process_sweep(self):
    self._sweep_pending = False
    self._calls_since_sweep = 0
    self._last_sweep_nanos = self._sweep_nanos
    now = self._sweep_nanos * 1e-9
    sweep = self._sweep
    self._sweep = {}  # every frame is used once: a lost 0x280 must not let old frames count again
    tracks, pts, slot_ids = self._tracks, self._pts, self._slot_ids

    for tid in list(tracks):
      last_seen = tracks[tid].last_seen
      if last_seen is not None and (now - last_seen > STALE_S or now < last_seen):  # also after a jump back in time
        self._retire(tid)

    # coherent observations: all four main frames of the slot present, with the same frame index
    obs = []
    for slot in range(NUM_SLOTS):
      a0, a1, a2, a3 = MAIN_ADDRS[slot]
      x0, x1, x2, x3 = sweep.get(a0), sweep.get(a1), sweep.get(a2), sweep.get(a3)
      if x0 is None or x1 is None or x2 is None or x3 is None:
        continue
      idx = (x0 >> F0_IDX[0]) & F0_IDX[1]
      if idx != (x1 >> F1_IDX[0]) & F1_IDX[1] or idx != (x2 >> F2_IDX[0]) & F2_IDX[1] or idx != (x3 >> F3_IDX[0]) & F3_IDX[1]:
        continue
      status = (x0 >> F0_STATUS[0]) & F0_STATUS[1]
      range_raw = (x0 >> F0_RANGE[0]) & F0_RANGE[1]
      az_raw = (x0 >> F0_AZIMUTH[0]) & F0_AZIMUTH[1]
      life = (x2 >> F2_LIFE[0]) & F2_LIFE[1]
      tid = (x3 >> F3_TRACK_ID[0]) & F3_TRACK_ID[1]
      u11 = u10 = ratio_raw = None
      xa = sweep.get(AUX_ADDRS[slot])
      if xa is not None and (xa >> AUX_IDX[0]) & AUX_IDX[1] == idx:
        u11 = (xa >> AUX_VREL[0]) & AUX_VREL[1]
        u10 = (xa >> AUX_VREL_UNCERTAINTY[0]) & AUX_VREL_UNCERTAINTY[1]
        ratio_raw = (xa >> AUX_RANGE_RATIO[0]) & AUX_RANGE_RATIO[1]
      obs.append((slot, idx, life, tid, TRACK_ID_MIN <= tid <= TRACK_ID_MAX,
                  status != STATUS_INVALID and range_raw != RANGE_INVALID and az_raw != AZIMUTH_INVALID and life != LIFE_INVALID,
                  (x0 >> F0_RANGE_SIGMA[0]) & F0_RANGE_SIGMA[1], (x1 >> F1_EXISTENCE[0]) & F1_EXISTENCE[1],
                  range_raw, az_raw, u11, u10, ratio_raw))

    # an invalid observation hides the object of its slot; one Bosch track id gives at most one point (lowest slot)
    valid_by_id = {}
    for o in obs:
      slot, tid, tid_ok, valid = o[0], o[3], o[4], o[5]
      if not (valid and tid_ok):
        hide = {slot_ids[slot]}
        if tid_ok:
          hide.add(tid)
        for h in hide - {None}:
          pts.pop(h, None)
        continue
      cur = valid_by_id.get(tid)
      if cur is None or slot < cur[0]:
        valid_by_id[tid] = o
    valid_ids = set(valid_by_id)
    # ids whose speed field is at a rail in this sweep: their true speed is unknown, whatever path they take below
    railed_ids = {tid for tid, o in valid_by_id.items() if o[10] in (0, VREL_MAX_RAW)}

    for tid, o in sorted(valid_by_id.items(), key=lambda kv: kv[1][0]):
      slot, idx, life, _, _, _, rsig, exist, range_raw, az_raw, u11, u10, ratio_raw = o
      old = slot_ids[slot]
      if old is not None and old != tid and old not in valid_ids:
        pts.pop(old, None)
      tr = tracks.get(tid)
      if tr is None:
        tr = tracks[tid] = _Track()
      if tid in railed_ids:
        tr.vtrust = tr.vtrust_t = None  # a speed trusted before a rail reading is never coasted afterwards

      # same incarnation: the lifecycle counter advanced by two per frame index step
      same = tr.prev_idx is not None and ((life - tr.prev_life) & 0xFFF) == 2 * ((idx - tr.prev_idx) & 0xF)
      if not same:
        tr.reset_history()
        pts.pop(tid, None)

      d_rel = RANGE_SCALE_M * range_raw + RANGE_OFFSET_M
      y_rel = d_rel * math.tan((az_raw - AZIMUTH_CENTER) * AZIMUTH_SCALE_RAD)  # left positive
      dv = _direct_vrel(u11, u10)
      live = _direct_vrel(u11)
      high_u10 = dv is None and live is not None and u10 is not None and u10 > VREL_MAX_UNCERTAINTY
      prev = tr.last_sample
      ratio_v = None
      rejected = False
      degraded = rsig >= RANGE_SIGMA_DEGRADED or exist in (0, 0x7F) or (u10 is not None and u10 > VREL_MAX_UNCERTAINTY)
      if prev is not None and now - prev[0] <= 0:
        # the sweep time did not advance: start the object again as a new one rather than keep an old point
        tr.reset_history()
        pts.pop(tid, None)
        prev = None
      if prev is not None:
        dt = now - prev[0]
        ratio = _range_ratio(ratio_raw)
        ratio_v = d_rel * (1 - ratio) / dt if ratio is not None else None
        residuals = []
        if dv is not None:
          residuals.append(abs(d_rel - (prev[1] + dv * dt)))
        if ratio is not None:
          residuals.append(abs(prev[1] - d_rel * ratio))
        if residuals:
          innovation = min(residuals)
          rejected = innovation > INNOVATION_HARD_MAX_M or (degraded and innovation > INNOVATION_MAX_M)
        else:
          rejected = abs((d_rel - prev[1]) / dt) > FALLBACK_RANGE_RATE_MAX_MPS

      if rejected:
        # keep the last accepted point briefly as an estimate; the rejected range is not published
        p = pts.get(tid)
        if prev is not None and now - prev[0] <= STALE_S and p is not None:
          p.measured = False
        else:
          pts.pop(tid, None)
        self._bind(tr, tid, idx, life, now, slot)
        continue

      if high_u10 or (dv is None and (ratio_v is None or degraded) and prev is not None):
        # range accepted, speed not trusted: coast the last trusted speed for a short while, else drop the point
        if tr.vtrust is not None and now - tr.vtrust_t <= STALE_S:
          p = pts.get(tid)
          if p is not None:
            p.dRel, p.yRel, p.vRel, p.measured = d_rel, y_rel, tr.vtrust, False
        else:
          tr.vtrust = tr.vtrust_t = None
          pts.pop(tid, None)
        self._bind(tr, tid, idx, life, now, slot)
        continue

      tr.last_sample = (now, d_rel)
      tr.n_samples += 1
      v_rel = dv if dv is not None else ratio_v
      tr.vtrust, tr.vtrust_t = v_rel, now
      if tr.n_samples >= 2 and v_rel is not None and math.isfinite(v_rel):
        p = pts.get(tid)
        if p is None:
          p = pts[tid] = _Point()
        p.dRel, p.yRel, p.vRel, p.measured = d_rel, y_rel, v_rel, True
      else:
        pts.pop(tid, None)
      self._bind(tr, tid, idx, life, now, slot)

    # publish: a Bosch id that was not in the previous sweep's output gets a new trackId (radard keys its filter on it)
    track_ids = {}
    points = []
    for tid in sorted(pts):
      # A point at the rail of the speed field has an unknown, larger speed. It is not published: radard then uses the
      # vision lead for that object, as without this source, instead of a closing speed that reads too low (FCW).
      # This holds also when the parser coasts an older speed for an id whose field is at the rail now.
      if abs(pts[tid].vRel) >= VREL_RAIL_MPS or tid in railed_ids:
        continue
      # the range offset makes small raw ranges negative: nothing is published at or behind the bumper
      if not pts[tid].dRel > 0:
        continue
      track_id = self._track_ids.get(tid)
      if track_id is None:
        track_id = self._next_track_id
        self._next_track_id += 1
      track_ids[tid] = track_id
      p = pts[tid]
      points.append(structs.RadarData.RadarPoint(trackId=track_id, dRel=float(p.dRel), yRel=float(p.yRel), vRel=float(p.vRel),
                                                 aRel=math.nan, yvRel=math.nan, measured=p.measured))
    self._track_ids = track_ids
    self._points = points

  def _bind(self, tr: _Track, tid: int, idx: int, life: int, now: float, slot: int):
    tr.prev_idx, tr.prev_life, tr.last_seen = idx, life, now
    for s in range(NUM_SLOTS):
      if s != slot and self._slot_ids[s] == tid:
        self._slot_ids[s] = None
    self._slot_ids[slot] = tid

  def _retire(self, tid: int):
    self._tracks.pop(tid, None)
    self._pts.pop(tid, None)
    for s in range(NUM_SLOTS):
      if self._slot_ids[s] == tid:
        self._slot_ids[s] = None
