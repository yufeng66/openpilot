"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

Licensed under the MIT License. Ported to dragonpilot from the sunnypilot-based
speed-dependent torque learner (yufeng66/openpilot, pal23sp-hda2-testing).

Speed-dependent torque learner, moment-store mode only: every speed bin keeps a
running 3x3 second-moment matrix instead of a point store, all bins stacked in
one SpeedBinMomentBank over a uniform 5-mph grid. A single toggle
(dp_lat_torqued_sd) arms it; the sunnypilot point-store speed-bin mode and its
toggle chain were deliberately not ported.
"""
import math

import numpy as np

from cereal import car, log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from dragonpilot.selfdrive.controls.lib.speed_dep_helpers import SPEED_DEP_CAR_CONFIG

# Speed envelope of the learner. Below the lower edge (parking/creep) the
# steer->lat-accel relation leaves the regime the fit models.
SPEED_BIN_MIN = 5.0   # m/s
SPEED_BIN_MAX = 40.0  # m/s

# Uniform 5-mph grid (15-85 mph). Highway cruise speeds cluster on 5-mph
# multiples (posted limits, long-press cruise steps), so nodes centered on that
# lattice accrue in-bin evidence at exactly the speeds drivers actually hold,
# and the nearest-learned clamp is never more than ~2.5 mph from a learned
# node. Only viable with the moment store: a point store at this bin count
# would starve on ~1/15 of the data per bin.
MOMENT_SPEED_BIN_CENTERS = [round(mph * 0.44704, 3) for mph in range(15, 90, 5)]

# Sanity window for the per-bin fits, as +/- fractions of the bin seeds. These
# are sunnypilot's relaxed-profile values (the config the learner was tuned
# under); the global learner keeps upstream's tighter stock window.
SPEED_DEP_FACTOR_SANITY = 1.0
SPEED_DEP_FRICTION_SANITY = 1.0

# --- Moment-matrix learner ---
# Tuned offline against 39k C3 points (learner_analysis 2026-08-09).
# MOMENT_ESS_CAP is in POINTS, not seconds: one observation can take at most
# ceiling/(cap+ceiling) of the state, which is what bounds a bad sample's impact.
# At the device's 20 Hz livePose rate a saturated bin holds ~20 min of in-bin
# driving (half-life ~14 min); the offline replay ran on 4x-decimated qlogs, so
# its wall-clock horizon looked 4x longer for the same cap. Raised from 6000
# (3.5 min half-life) after road feedback 2026-08-23: at that horizon a single
# 5-8 min pass through a speed range rewrote 65-80% of those bins' state, which
# read as the learner chasing the most recent drive. Raising the cap is cache
# compatible in one direction only - load_cache rejects rows with S above the
# cap, so old (smaller-S) rows still load, but lowering it would drop them all.
MOMENT_ESS_CAP = 24000.0
MOMENT_DENSITY_CEILING = 7.0    # max inverse-density up-weight, at MOMENT_DENSITY_CEILING_V[0] and below
MOMENT_DENSITY_CEILING_HI = 15.0  # ... and at MOMENT_DENSITY_CEILING_V[1] and above
MOMENT_DENSITY_CEILING_V = (MOMENT_SPEED_BIN_CENTERS[0], MOMENT_SPEED_BIN_CENTERS[-1])
MOMENT_DENSITY_FLOOR = 0.1      # min down-weight for over-represented steer ranges
MOMENT_SPEED_KERNEL_H = 3.0     # m/s, Gaussian speed kernel width
MOMENT_KERNEL_MIN = 0.01        # ignore bins further than ~3 kernel widths away
MOMENT_MIN_ESS = 60.0           # readiness: accumulated weight
MOMENT_MIN_IN_BIN_ESS = 15.0    # readiness: weight that came from the bin's own speed range
MOMENT_MIN_BUCKET_ESS = 3.0     # readiness: weight in every inner steer bucket (edges exempt)
MOMENT_MIN_XSTD = 0.06          # readiness: steer spread, i.e. the fit is identifiable
# Output smoothing and clip floor for the applied fit. The filter decay is in
# FirstOrderFilter's rc units (dt=DT_MDL) but the estimate runs at 4 Hz, so the
# wall-clock ramp is ~5x the rc: rc=1.0 -> ~5 s. The floor exists because
# latAccelFactor is a divisor in the controller and the relaxed sanity window's
# lower clip bound is (1 - 1.0) * seed = 0.0.
MOMENT_FILTER_DECAY = 1.0
MOMENT_MIN_LAT_ACCEL_FACTOR = 0.5
MOMENT_CACHE_ROW = 16           # 6 unique moments + S + S_in + 8 density counters


def moment_density_ceiling(centers):
  """Per-bin cap on the inverse-density up-weight, ramped with speed.

  The steer distribution narrows as speed rises. Measured on a Palisade cache
  2026-08-23, |steer| > 0.3 is ~9% of accepted points at 25 mph but ~1% above
  60 mph, so the inverse-density ratio the road asks for runs ~4 in the city and
  17-43 on the highway. One flat ceiling therefore has to be either inert in the
  city or clipped to a third of the intended big-move leverage on the highway --
  and the highway bins are the ones whose slope is hardest to identify, because
  large-steer points are what carry it. Ramping tracks the distribution instead:
  inert at and below the low anchor, where bins already reach the equal-share
  target that upstream's fixed-capacity steer buckets aim for, and opening up
  only where the clip actually binds.

  Deliberately stops well short of the ratio the top bins ask for: those tail
  buckets hold ~1% of arrivals, so buying full equal-share up there costs most of
  the bin's effective sample size (Kish ESS ~15% of raw evidence at ceiling 30 vs
  ~23% at 15, measured on the same cache). np.interp clamps outside the anchors,
  so a SPEED_DEP_CAR_CONFIG speed_bp grid wider or narrower than the default
  stays well defined.
  """
  return np.interp(np.asarray(centers, dtype=float), MOMENT_DENSITY_CEILING_V,
                   (MOMENT_DENSITY_CEILING, MOMENT_DENSITY_CEILING_HI))


class SpeedBinMoment:
  """Reference implementation: one speed bin's moment state as a standalone object.

  Production uses SpeedBinMomentBank (same math, all bins in stacked arrays); this
  class is kept as the readable single-bin reference and as the oracle the bank is
  tested against, and it doubles as the cache-row decoder for offline analysis.

  Running second-moment matrix of p = [steer, 1, lateral_accel] for one speed bin.

  Upstream fits latAccelFactor with a total-least-squares SVD of the stacked point
  matrix A; the slope it takes is the smallest right-singular vector of A, which is
  identically the smallest eigenvector of A^T A. So the fit only ever needed the 3x3
  moment matrix, never the points — friction (the spread perpendicular to the fitted
  line) falls out of the same moments. That makes this O(1) in memory: ~10k stored
  points per bin collapse to MOMENT_CACHE_ROW floats.

  Forgetting is data-clocked. Once the accumulated weight reaches ess_cap the whole
  matrix is rescaled, which is exactly an exponential moving average over the rank-1
  updates with the decay back-solved to pin the total weight:

      M <- lambda*M + (w*C/(C+w))*pp^T,   lambda = C/(C+w)

  A bin therefore only ages when it actually sees driving in its speed range.
  """

  def __init__(self, steer_bucket_bounds, ess_cap=MOMENT_ESS_CAP,
               density_ceiling=MOMENT_DENSITY_CEILING):
    self.bounds = list(steer_bucket_bounds)
    self.ess_cap = float(ess_cap)
    self.density_ceiling = float(density_ceiling)
    self.M = np.zeros((3, 3))
    self.S = 0.0
    self.S_in = 0.0  # weight that came from inside this bin's own speed range
    self.dens = np.zeros(len(self.bounds))

  def _density_weight(self, steer, kernel_weight):
    """Continuous replacement for upstream's fixed-capacity steer buckets: weight a
    point by how under-represented its steer range currently is. Counters advance
    by the kernel weight, so a far-speed point claims proportionally less
    representation of its steer range. Returns 0 outside the tracked range, which
    upstream drops as well."""
    for i, (lo, hi) in enumerate(self.bounds):
      if lo <= steer < hi:
        total = self.dens.sum()
        mean_d = total / len(self.dens) if total > 0 else 1.0
        w = float(np.clip(mean_d / max(self.dens[i], 0.5),
                          MOMENT_DENSITY_FLOOR, self.density_ceiling))
        self.dens[i] += kernel_weight
        if self.dens.sum() > self.ess_cap:
          self.dens *= self.ess_cap / self.dens.sum()
        return w
    return 0.0

  def add(self, steer, lateral_acc, kernel_weight, in_bin=True):
    w = self._density_weight(steer, kernel_weight) * kernel_weight
    if w <= 0.0:
      return
    p = np.array([steer, 1.0, lateral_acc])
    self.M += w * np.outer(p, p)
    self.S += w
    if in_bin:
      self.S_in += w
    if self.S > self.ess_cap:
      f = self.ess_cap / self.S
      self.M *= f
      self.S_in *= f
      self.S = self.ess_cap

  def x_std(self):
    if self.S <= 0.0:
      return 0.0
    m = self.M / self.S
    return math.sqrt(max(m[0, 0] - m[0, 1] ** 2, 0.0))

  def is_valid(self):
    """Mirrors point-mode validity invariants: enough total evidence, at least some
    evidence from inside the bin's own speed range (so a bin cannot go valid on
    neighbour-speed data alone and override a per-car seed), enough steer spread
    for the slope to be identifiable, and coverage of every inner steer bucket
    (edges exempt, matching the relaxed min-bucket profile). The fitted value stays
    neighbour-informed either way — this only gates when it is applied."""
    if self.S < MOMENT_MIN_ESS or self.S_in < MOMENT_MIN_IN_BIN_ESS or self.x_std() < MOMENT_MIN_XSTD:
      return False
    return bool((self.dens[1:-1] >= MOMENT_MIN_BUCKET_ESS).all())

  def fit(self, friction_factor):
    """Returns (latAccelFactor, frictionCoefficient), or None if not identifiable.
    Same algebra as upstream's estimate_params(), evaluated on the moments."""
    if self.S <= 0.0:
      return None
    try:
      _, vecs = np.linalg.eigh(self.M)
    except np.linalg.LinAlgError:
      return None
    v = vecs[:, 0]
    if abs(v[2]) < 1e-12:
      return None
    slope = -v[0] / v[2]

    m = self.M / self.S
    e_x, e_y = m[0, 1], m[1, 2]
    e_xx, e_xy, e_yy = m[0, 0], m[0, 2], m[2, 2]
    # project onto the direction perpendicular to the fit, i.e. upstream's slope2rot
    a = -math.sqrt(slope ** 2 / (slope ** 2 + 1))
    b = math.sqrt(1 / (slope ** 2 + 1))
    e_sp = a * e_x + b * e_y
    e_sp2 = a * a * e_xx + 2 * a * b * e_xy + b * b * e_yy
    friction = math.sqrt(max(e_sp2 - e_sp * e_sp, 0.0)) * friction_factor
    if math.isnan(slope) or math.isnan(friction):
      return None
    return float(slope), float(friction)

  def to_cache(self):
    """Flat row for the liveTorqueParameters cache. Density counters are included so
    a reboot does not hand the first few points the full rare-range up-weight."""
    return [float(self.M[0, 0]), float(self.M[0, 1]), float(self.M[0, 2]),
            float(self.M[1, 1]), float(self.M[1, 2]), float(self.M[2, 2]),
            float(self.S), float(self.S_in)] + [float(d) for d in self.dens]

  def load_cache(self, row):
    row = list(row)
    # Width is derived from the bucket count, so a change to STEER_BUCKET_BOUNDS
    # (or to the row layout) invalidates old caches instead of misreading them.
    if len(row) != 8 + len(self.bounds):
      return False
    if not all(math.isfinite(v) for v in row):
      return False
    m00, m01, m02, m11, m12, m22, s, s_in = row[:8]
    if s <= 0.0 or s > self.ess_cap * 1.01:
      return False
    if s_in < 0.0 or s_in > s * 1.01:
      return False
    self.M = np.array([[m00, m01, m02], [m01, m11, m12], [m02, m12, m22]])
    self.S = float(s)
    self.S_in = float(s_in)
    self.dens = np.array(row[8:], dtype=float)
    return True


class SpeedBinMomentBank:
  """Every speed bin's moment state in stacked arrays, updated with one numpy
  call chain per point instead of one per bin.

  Semantically identical to looping SpeedBinMoment.add over bins (that class is
  the reference this one is tested against): each bin's row only ever sees its
  own data, so vectorizing across bins changes evaluation order, never results.
  The kernel cutoff is applied by zeroing weights instead of skipping bins —
  the update stays dense for one-call math, while out-of-reach bins get +0
  everywhere and their state, including data-clocked forgetting, stays frozen
  exactly as with the loop-skip.

  Why this exists: on the device's in-order cores numpy per-call overhead
  dominates 3x3-scale math (~100 us per touched bin measured on the per-object
  path at 20 Hz), so per-bin objects priced the bin count. Stacked state makes
  ingest and fit roughly O(1) in bins, which is what affords the 5-mph grid.
  """

  def __init__(self, centers, bounds, steer_bucket_bounds, ess_cap=MOMENT_ESS_CAP,
               density_ceiling=None):
    self.centers = np.asarray(centers, dtype=float)
    self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    self.n = len(self.bounds)
    steer_bounds = [(float(lo), float(hi)) for lo, hi in steer_bucket_bounds]
    # searchsorted bucket lookup needs contiguous ranges; both the speed bins
    # (midpoint-derived) and STEER_BUCKET_BOUNDS satisfy it today. Fail loudly
    # if a future layout gaps them instead of silently misrouting points.
    assert len(self.centers) == self.n
    assert all(a[1] == b[0] for a, b in zip(self.bounds[:-1], self.bounds[1:], strict=False))
    assert all(a[1] == b[0] for a, b in zip(steer_bounds[:-1], steer_bounds[1:], strict=False))
    self.n_buckets = len(steer_bounds)
    self.steer_edges = np.array([b[0] for b in steer_bounds] + [steer_bounds[-1][1]])
    self.speed_edges = np.array([b[0] for b in self.bounds] + [self.bounds[-1][1]])
    self.ess_cap = float(ess_cap)
    # per-bin, so np.clip below applies each bin's own ceiling elementwise; a
    # scalar override broadcasts, and a wrong-length array fails here, not silently
    self.density_ceiling = (moment_density_ceiling(self.centers) if density_ceiling is None else
                            np.broadcast_to(np.asarray(density_ceiling, dtype=float), (self.n,)).copy())
    self.M = np.zeros((self.n, 3, 3))
    self.S = np.zeros(self.n)
    self.S_in = np.zeros(self.n)
    self.dens = np.zeros((self.n, self.n_buckets))

  def add(self, steer, lateral_acc, vego):
    """One accepted point updates all bins at once, weighted by the speed kernel.

    Below the lowest bound (parking/creep) the steer->lat-accel relation leaves
    the regime the fit models, so the point is dropped — the same envelope the
    point store enforces implicitly by having no bin there."""
    if not (self.speed_edges[0] <= vego < self.speed_edges[-1]):
      return
    b = int(np.searchsorted(self.steer_edges, steer, side='right')) - 1
    if b < 0 or b >= self.n_buckets:
      return  # outside the tracked steer range, which upstream drops as well
    k = np.exp(-0.5 * ((vego - self.centers) / MOMENT_SPEED_KERNEL_H) ** 2)
    k = np.where(k >= MOMENT_KERNEL_MIN, k, 0.0)
    # inverse-density up/down-weight per bin, read before this point advances
    # the counters (mirrors SpeedBinMoment._density_weight exactly)
    totals = self.dens.sum(axis=1)
    mean_d = np.where(totals > 0.0, totals / self.n_buckets, 1.0)
    w = k * np.clip(mean_d / np.maximum(self.dens[:, b], 0.5),
                    MOMENT_DENSITY_FLOOR, self.density_ceiling)
    self.dens[:, b] += k
    totals = self.dens.sum(axis=1)
    self.dens *= np.where(totals > self.ess_cap,
                          self.ess_cap / np.maximum(totals, 1e-12), 1.0)[:, None]
    p = np.array([steer, 1.0, lateral_acc])
    self.M += w[:, None, None] * np.outer(p, p)
    self.S += w
    i = int(np.searchsorted(self.speed_edges, vego, side='right')) - 1
    self.S_in[i] += w[i]
    over = self.S > self.ess_cap
    f = np.where(over, self.ess_cap / np.maximum(self.S, 1e-12), 1.0)
    self.M *= f[:, None, None]
    self.S_in *= f
    self.S = np.where(over, self.ess_cap, self.S)

  def x_std(self):
    m = self.M / np.maximum(self.S, 1e-12)[:, None, None]
    xs = np.sqrt(np.maximum(m[:, 0, 0] - m[:, 0, 1] ** 2, 0.0))
    return np.where(self.S > 0.0, xs, 0.0)

  def is_valid(self):
    """Vector of per-bin validity, same gates as the reference."""
    ok = (self.S >= MOMENT_MIN_ESS) & (self.S_in >= MOMENT_MIN_IN_BIN_ESS) \
         & (self.x_std() >= MOMENT_MIN_XSTD)
    return ok & (self.dens[:, 1:-1] >= MOMENT_MIN_BUCKET_ESS).all(axis=1)

  def fit(self, friction_factor):
    """Batched fit of every bin. Returns (slopes, frictions, ok) arrays; ok False
    where the reference per-bin fit would return None (no evidence, degenerate
    eigenvector, or NaN). np.linalg.eigh is a gufunc, so the (n,3,3) stack goes
    through LAPACK in one call."""
    ok = self.S > 0.0
    ok &= np.isfinite(self.M.reshape(self.n, 9)).all(axis=1)
    m_safe = np.where(ok[:, None, None], self.M, np.eye(3))
    try:
      _, vecs = np.linalg.eigh(m_safe)
      v = vecs[:, :, 0]
    except np.linalg.LinAlgError:
      # a batched eigh fails as a unit; retry bins one at a time like the reference
      v = np.zeros((self.n, 3))
      for i in range(self.n):
        if ok[i]:
          try:
            v[i] = np.linalg.eigh(m_safe[i])[1][:, 0]
          except np.linalg.LinAlgError:
            ok[i] = False
    identifiable = np.abs(v[:, 2]) >= 1e-12
    ok &= identifiable
    slope = -v[:, 0] / np.where(identifiable, v[:, 2], 1.0)
    m = self.M / np.maximum(self.S, 1e-12)[:, None, None]
    e_x, e_y = m[:, 0, 1], m[:, 1, 2]
    e_xx, e_xy, e_yy = m[:, 0, 0], m[:, 0, 2], m[:, 2, 2]
    # project onto the direction perpendicular to the fit, i.e. upstream's slope2rot
    a = -np.sqrt(slope ** 2 / (slope ** 2 + 1.0))
    b = np.sqrt(1.0 / (slope ** 2 + 1.0))
    e_sp = a * e_x + b * e_y
    e_sp2 = a * a * e_xx + 2.0 * a * b * e_xy + b * b * e_yy
    friction = np.sqrt(np.maximum(e_sp2 - e_sp * e_sp, 0.0)) * friction_factor
    ok &= ~np.isnan(slope) & ~np.isnan(friction)
    return slope, friction, ok

  def reset_bin(self, i):
    self.M[i] = 0.0
    self.S[i] = 0.0
    self.S_in[i] = 0.0
    self.dens[i] = 0.0

  def to_cache(self, i):
    """Flat row for the liveTorqueParameters cache, identical layout to the
    reference so device caches decode with either implementation."""
    mi = self.M[i]
    return [float(mi[0, 0]), float(mi[0, 1]), float(mi[0, 2]),
            float(mi[1, 1]), float(mi[1, 2]), float(mi[2, 2]),
            float(self.S[i]), float(self.S_in[i])] + [float(d) for d in self.dens[i]]

  def load_cache(self, i, row):
    row = list(row)
    # Width is derived from the bucket count, so a change to STEER_BUCKET_BOUNDS
    # (or to the row layout) invalidates old caches instead of misreading them.
    if len(row) != 8 + self.n_buckets:
      return False
    if not all(math.isfinite(val) for val in row):
      return False
    m00, m01, m02, m11, m12, m22, s, s_in = row[:8]
    if s <= 0.0 or s > self.ess_cap * 1.01:
      return False
    if s_in < 0.0 or s_in > s * 1.01:
      return False
    self.M[i] = np.array([[m00, m01, m02], [m01, m11, m12], [m02, m12, m22]])
    self.S[i] = float(s)
    self.S_in[i] = float(s_in)
    self.dens[i] = np.asarray(row[8:], dtype=float)
    return True


class TorqueEstimatorExt:
  """dp extension mixed into TorqueEstimator via multiple inheritance.

  Adds per-speed-bin moment learning on top of upstream's single-value torqued.
  Gated by the dp_lat_torqued_sd toggle (read once at init; torqued restarts
  each drive, so an offroad toggle change applies on the next onroad cycle).

  Data flow:
    1. torqued calls _on_torque_point() for each quality-filtered sample → moment bank
    2. _estimate_params_speed_binned() fits every bin from its moment matrix at 4 Hz
    3. _extend_msg() writes per-bin values to cereal; controlsd interpolates them by
       speed each control frame via speed_dep_helpers.interp_live_torque_params(),
       with unlearned bins falling back to seed/nearest-learned/global values
  """

  def __init__(self, CP: car.CarParams):
    self.CP = CP
    self._params = Params()
    self.speed_binned = (CP.lateralTuning.which() == 'torque'
                         and self._params.get_bool("dp_lat_torqued_sd"))

  def initialize_custom_params(self):
    """Init speed bins and restore cached values before first get_msg. Must run
    after TorqueEstimator.__init__ has set the offline seed values."""
    if self.speed_binned:
      self._post_reset()
      self._restore_ext_cache()

  # --- Speed-binned learning hooks (called from TorqueEstimator) ---

  @staticmethod
  def _centers_to_bounds(centers):
    """Derive bin bounds from centers using midpoints between consecutive centers.
    First bin starts at the envelope's lower edge, last bin ends at its upper edge."""
    bounds = []
    for i, c in enumerate(centers):
      lo = SPEED_BIN_MIN if i == 0 else (centers[i - 1] + c) / 2
      hi = SPEED_BIN_MAX if i == len(centers) - 1 else (c + centers[i + 1]) / 2
      bounds.append((lo, hi))
    return bounds

  def _post_reset(self):
    """Initializes the moment bank, per-bin filters, and sanity bounds."""
    if not self.speed_binned:
      return

    from openpilot.selfdrive.locationd.torqued import STEER_BUCKET_BOUNDS, MIN_FILTER_DECAY

    cfg = SPEED_DEP_CAR_CONFIG.get(self.CP.carFingerprint, {})

    if 'speed_bp' in cfg:
      self.speed_bin_centers = list(cfg['speed_bp'])
    else:
      self.speed_bin_centers = list(MOMENT_SPEED_BIN_CENTERS)
    self.speed_bin_bounds = self._centers_to_bounds(self.speed_bin_centers)
    n_bins = len(self.speed_bin_bounds)

    self.moment_bank = SpeedBinMomentBank(self.speed_bin_centers, self.speed_bin_bounds, STEER_BUCKET_BOUNDS)
    self._speed_bin_last_valid = [False] * n_bins

    # Seed values: from the per-car config if present, otherwise the global
    # offline values for all bins
    ref_lafs = cfg.get('laf_bp', [self.offline_latAccelFactor] * n_bins)
    ref_frictions = cfg.get('friction_bp', [self.offline_friction] * n_bins)
    self.speed_bin_filtered = [
      {'latAccelFactor': FirstOrderFilter(ref_lafs[i], float(MIN_FILTER_DECAY), DT_MDL),
       'frictionCoefficient': FirstOrderFilter(ref_frictions[i], float(MIN_FILTER_DECAY), DT_MDL)}
      for i in range(n_bins)
    ]
    # Fits get clipped to +/-sanity fractions of the seeds to prevent runaway learning
    self.speed_bin_lat_accel_factor_bounds = [
      ((1.0 - SPEED_DEP_FACTOR_SANITY) * factor, (1.0 + SPEED_DEP_FACTOR_SANITY) * factor)
      for factor in ref_lafs
    ]
    self.speed_bin_friction_bounds = [
      ((1.0 - SPEED_DEP_FRICTION_SANITY) * f, (1.0 + SPEED_DEP_FRICTION_SANITY) * f)
      for f in ref_frictions
    ]

  def _on_torque_point(self, steer, lateral_acc, vego):
    """Called from handle_log for every quality-filtered sample. The moment
    learner feeds every nearby bin with a Gaussian speed weight, so a bin is
    informed by its neighbours rather than starving until its own range is
    driven — the whole reason the top bin can report anything at all."""
    if not self.speed_binned:
      return
    self.moment_bank.add(steer, lateral_acc, vego)

  def _restore_ext_cache(self, cache_ltp=None):
    """Restores per-bin filter values and moment rows from cache.
    Reads from Params when cache_ltp is not provided; that path also requires the
    global learner's restore key to match (same car, offline baseline, and learner
    version), so the speed bins live and die with the global cache. Passing
    cache_ltp directly bypasses the key gate (test seam)."""
    if not self.speed_binned:
      return
    try:
      from openpilot.selfdrive.locationd.torqued import VERSION
      if cache_ltp is None:
        cache = self._params.get("LiveTorqueParameters")
        if not cache:
          return
        with log.Event.from_bytes(cache) as evt:
          cache_ltp = evt.liveTorqueParameters
        params_cache = self._params.get("CarParamsPrevRoute")
        if params_cache is None:
          cloudlog.info("speed-dep: no previous CarParams, restarting learning")
          return
        with car.CarParams.from_bytes(params_cache) as cache_CP:
          if self.get_restore_key(cache_CP, cache_ltp.version) != self.get_restore_key(self.CP, VERSION):
            cloudlog.info("speed-dep: cache from different car or version, restarting learning")
            return
      n_bins = len(self.speed_bin_bounds)
      # Reject cache from a different bin config (e.g. a per-car seed update
      # changed the centers)
      cached_centers = list(cache_ltp.speedBinCenters)
      if len(cached_centers) != n_bins or not np.allclose(cached_centers, self.speed_bin_centers, atol=0.01):
        cloudlog.info("speed-dep: config changed, restarting learning")
        return
      if (len(cache_ltp.speedBinLatAccelFactors) == n_bins and
          len(cache_ltp.speedBinFrictions) == n_bins):
        for i in range(n_bins):
          self.speed_bin_filtered[i]['latAccelFactor'].x = cache_ltp.speedBinLatAccelFactors[i]
          self.speed_bin_filtered[i]['frictionCoefficient'].x = cache_ltp.speedBinFrictions[i]
        if len(cache_ltp.speedBinPoints) == n_bins:
          for i in range(n_bins):
            rows = cache_ltp.speedBinPoints[i]
            # A moment cache is a single MOMENT_CACHE_ROW-wide row per bin. Any
            # other shape (e.g. a point-store cache from a sunnypilot install, or
            # an older row layout — load_cache checks the exact width) is
            # silently dropped and the bin relearns from the restored seed.
            if len(rows) == 1 and len(rows[0]) > 2:
              self.moment_bank.load_cache(i, rows[0])
        cloudlog.info("restored speed-bin torque params from cache")
    except Exception:
      cloudlog.exception("speed-dep: failed to restore cache")

  def _estimate_params_speed_binned(self):
    """Fit each bin straight from its moment matrix.

    Long-horizon smoothing happens in the data domain via the ESS window, but the
    output still goes through the bin's FirstOrderFilter with a short dedicated
    decay (~5 s wall-clock at the 4 Hz estimate cadence): it ramps the transition
    when a bin first validates or re-validates instead of stepping the live
    steering gain between two 250 ms messages, and it keeps any single degenerate
    fit from being applied whole. The filter is also fed while the bin is not yet
    valid, so by the time the valid flag flips the filtered value has already
    converged near the fit and the fallback→learned switch is seamless.

    latAccelFactor is additionally floored: it is a divisor in the controller,
    and the relaxed sanity window's lower clip bound is 0.0."""
    from openpilot.selfdrive.locationd.torqued import FRICTION_FACTOR

    bank = self.moment_bank
    valid_all = bank.is_valid()
    slopes, frictions, fit_ok = bank.fit(FRICTION_FACTOR)
    results = []
    for i in range(bank.n):
      if not fit_ok[i]:
        if valid_all[i]:
          # Ill-conditioned despite enough evidence: drop this bin's history and rebuild.
          cloudlog.warning(f"speed-dep moment: bin {i} not identifiable with valid weight, resetting bin")
          bank.reset_bin(i)
        self._speed_bin_last_valid[i] = False
        results.append((i, False))
        continue

      valid = bool(valid_all[i])
      factor_lo, factor_hi = self.speed_bin_lat_accel_factor_bounds[i]
      fric_lo, fric_hi = self.speed_bin_friction_bounds[i]
      factor_lo = max(factor_lo, MOMENT_MIN_LAT_ACCEL_FACTOR)
      factor_hi = max(factor_hi, factor_lo)
      for key, value, lo, hi in (('latAccelFactor', float(slopes[i]), factor_lo, factor_hi),
                                 ('frictionCoefficient', float(frictions[i]), fric_lo, fric_hi)):
        self.speed_bin_filtered[i][key].update_alpha(MOMENT_FILTER_DECAY)
        self.speed_bin_filtered[i][key].update(float(np.clip(value, lo, hi)))
      self._speed_bin_last_valid[i] = valid
      results.append((i, valid))
    return results

  def _extend_msg(self, ltp, with_points):
    """Appends per-bin values to the cereal msg. with_points=True on cache writes (every 60s)."""
    if not self.speed_binned:
      return
    bin_results = self._estimate_params_speed_binned()
    n_bins = len(self.speed_bin_bounds)

    lat_factors, frictions, valid_flags = [], [], []
    bin_points = [] if with_points else None
    for i in range(n_bins):
      lat_factors.append(float(self.speed_bin_filtered[i]['latAccelFactor'].x))
      frictions.append(float(self.speed_bin_filtered[i]['frictionCoefficient'].x))
      _, valid = bin_results[i]
      valid_flags.append(valid)
      if with_points:
        # speedBinPoints is List(List(List(Float32))): per bin, a list of rows.
        # The moment learner writes a single MOMENT_CACHE_ROW-wide row.
        bin_points.append([self.moment_bank.to_cache(i)])

    ltp.speedBinCenters = self.speed_bin_centers
    ltp.speedBinLatAccelFactors = lat_factors
    ltp.speedBinFrictions = frictions
    ltp.speedBinValid = valid_flags
    if with_points:
      ltp.speedBinPoints = bin_points
