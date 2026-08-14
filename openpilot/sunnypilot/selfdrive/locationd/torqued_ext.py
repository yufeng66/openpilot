"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import numpy as np

from opendbc.car.structs import car
from openpilot.cereal import log

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

RELAXED_MIN_BUCKET_POINTS = np.array([1, 200, 300, 500, 500, 300, 200, 1])

ALLOWED_CARS = ['toyota', 'hyundai', 'rivian', 'honda']

# Default speed bins — used when car has no speed_dependent.toml entry.
DEFAULT_SPEED_BIN_BOUNDS = [(5, 8), (8, 12), (12, 18), (18, 24), (24, 29), (29, 35), (35, 40)]
DEFAULT_SPEED_BIN_CENTERS = [6.5, 10.0, 15.0, 21.0, 26.5, 32.0, 37.5]

# --- Moment-matrix learner (experimental, SpeedDependentTorqueMomentToggle) ---
# Tuned offline against 39k C3 points (untracked/learner_analysis_2026-08-09).
# MOMENT_ESS_CAP is in POINTS, not seconds: one observation can take at most
# ceiling/(cap+ceiling) of the state, which is what bounds a bad sample's impact.
# At the device's 20 Hz livePose rate a saturated bin holds ~5 min of in-bin
# driving (half-life ~3.5 min); the offline replay ran on 4x-decimated qlogs, so
# its wall-clock horizon looked 4x longer for the same cap.
MOMENT_ESS_CAP = 6000.0
MOMENT_DENSITY_CEILING = 7.0    # max inverse-density up-weight for rare steer ranges
MOMENT_DENSITY_FLOOR = 0.1      # min down-weight for over-represented steer ranges
MOMENT_SPEED_KERNEL_H = 3.0     # m/s, Gaussian speed kernel width
MOMENT_KERNEL_MIN = 0.01        # ignore bins further than ~3 kernel widths away
MOMENT_MIN_ESS = 60.0           # readiness: accumulated weight
MOMENT_MIN_XSTD = 0.06          # readiness: steer spread, i.e. the fit is identifiable
MOMENT_CACHE_ROW = 15           # 6 unique moments + total weight + 8 density counters


class SpeedBinMoment:
  """Running second-moment matrix of p = [steer, 1, lateral_accel] for one speed bin.

  Upstream fits latAccelFactor with a total-least-squares SVD of the stacked point
  matrix A; the slope it takes is the smallest right-singular vector of A, which is
  identically the smallest eigenvector of A^T A. So the fit only ever needed the 3x3
  moment matrix, never the points — friction (the spread perpendicular to the fitted
  line) falls out of the same moments. That makes this O(1) in memory: ~10k stored
  points per bin collapse to 15 floats.

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
    self.dens = np.zeros(len(self.bounds))

  def _density_weight(self, steer):
    """Continuous replacement for upstream's fixed-capacity steer buckets: weight a
    point by how under-represented its steer range currently is. Returns 0 outside
    the tracked range, which upstream drops as well."""
    for i, (lo, hi) in enumerate(self.bounds):
      if lo <= steer < hi:
        total = self.dens.sum()
        mean_d = total / len(self.dens) if total > 0 else 1.0
        w = float(np.clip(mean_d / max(self.dens[i], 0.5),
                          MOMENT_DENSITY_FLOOR, self.density_ceiling))
        self.dens[i] += 1.0
        if self.dens.sum() > self.ess_cap:
          self.dens *= self.ess_cap / self.dens.sum()
        return w
    return 0.0

  def add(self, steer, lateral_acc, kernel_weight):
    w = self._density_weight(steer) * kernel_weight
    if w <= 0.0:
      return
    p = np.array([steer, 1.0, lateral_acc])
    self.M += w * np.outer(p, p)
    self.S += w
    if self.S > self.ess_cap:
      f = self.ess_cap / self.S
      self.M *= f
      self.S = self.ess_cap

  def x_std(self):
    if self.S <= 0.0:
      return 0.0
    m = self.M / self.S
    return math.sqrt(max(m[0, 0] - m[0, 1] ** 2, 0.0))

  def is_valid(self):
    """Enough evidence, and enough steer spread for the slope to be identifiable."""
    return self.S >= MOMENT_MIN_ESS and self.x_std() >= MOMENT_MIN_XSTD

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
            float(self.S)] + [float(d) for d in self.dens]

  def load_cache(self, row):
    row = list(row)
    if len(row) != MOMENT_CACHE_ROW or len(self.dens) != MOMENT_CACHE_ROW - 7:
      return False
    if not all(math.isfinite(v) for v in row):
      return False
    m00, m01, m02, m11, m12, m22, s = row[:7]
    if s <= 0.0 or s > self.ess_cap * 1.01:
      return False
    self.M = np.array([[m00, m01, m02], [m01, m11, m12], [m02, m12, m22]])
    self.S = float(s)
    self.dens = np.array(row[7:], dtype=float)
    return True

class TorqueEstimatorExt:
  """SP extension mixed into TorqueEstimator via multiple inheritance.

  Adds per-speed-bin learning on top of upstream's single-value torqued.
  Gated by SpeedDependentTorqueToggle + EnforceTorqueControl (both offroad-only).

  Data flow:
    1. torqued calls _on_torque_point() for each quality-filtered sample → routed to speed bin
    2. _estimate_params_speed_binned() runs independent SVD fit per bin (same algo as upstream)
    3. Per-bin validity uses upstream's is_valid() — requires sufficient steer-range coverage
    4. _extend_msg() writes per-bin values to cereal; invalid bins fall back to seed/global values
    5. The lateral controller interpolates latAccelFactor and friction by speed each frame

  Bin configuration:
    - Cars with a speed_dependent.toml entry use per-car bin centers (and derived bounds)
    - Cars without an entry use DEFAULT_SPEED_BIN_BOUNDS and seed all bins with global offline values
  """

  def __init__(self, CP: car.CarParams):
    self.CP = CP
    self._params = Params()
    self.frame = -1

    self.enforce_torque_control_toggle = self._params.get_bool("EnforceTorqueControl")  # only during init
    self.use_params = self.CP.brand in ALLOWED_CARS and self.CP.lateralTuning.which() == 'torque'
    self.use_live_torque_params = self._params.get_bool("LiveTorqueParamsToggle")
    self.custom_torque_params = self._params.get_bool("CustomTorqueParams")
    self.torque_override_enabled = self._params.get_bool("TorqueParamsOverrideEnabled")
    # Not restricted by ALLOWED_CARS brand list. Requires EnforceTorqueControl:
    # the speed-dep settings live behind the Enforce-gated torque customization
    # panel, so the feature must not run while its controls are unreachable.
    self.speed_binned = (self.CP.lateralTuning.which() == 'torque'
                         and self.enforce_torque_control_toggle
                         and self._params.get_bool("SpeedDependentTorqueToggle"))
    # Experimental estimator swap: same bins and same cereal output, but each bin
    # keeps a running moment matrix instead of a point store. Strictly a sub-mode
    # of speed-dep, so it cannot arm on its own.
    self.moment_learner = (self.speed_binned
                           and self._params.get_bool("SpeedDependentTorqueMomentToggle"))
    # Defaults — overwritten by TorqueEstimator.__init__ before initialize_custom_params runs
    self.min_bucket_points = RELAXED_MIN_BUCKET_POINTS
    self.factor_sanity = 0.0
    self.friction_sanity = 0.0
    self.offline_latAccelFactor = 0.0
    self.offline_friction = 0.0

  def initialize_custom_params(self, decimated=False):
    self.update_use_params()

    if self.enforce_torque_control_toggle:
      if self._params.get_bool("LiveTorqueParamsRelaxedToggle"):
        self.min_bucket_points = (RELAXED_MIN_BUCKET_POINTS / (10 if decimated else 1)).tolist()
        self.factor_sanity = 0.5 if decimated else 1.0
        self.friction_sanity = 0.8 if decimated else 1.0

      if self._params.get_bool("CustomTorqueParams"):
        self.offline_latAccelFactor = float(self._params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
        self.offline_friction = float(self._params.get("TorqueParamsOverrideFriction", return_default=True))

    # Init speed bins and restore cached values before first get_msg
    if self.speed_binned:
      self._post_reset()
      self._restore_ext_cache()

  def _update_params(self):
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.use_live_torque_params = self._params.get_bool("LiveTorqueParamsToggle")
      self.custom_torque_params = self._params.get_bool("CustomTorqueParams")
      self.torque_override_enabled = self._params.get_bool("TorqueParamsOverrideEnabled")

  def update_use_params(self):
    self._update_params()

    if self.enforce_torque_control_toggle:
      if self.custom_torque_params and self.torque_override_enabled:
        self.use_params = False
      else:
        self.use_params = self.use_live_torque_params

    self.frame += 1

  # --- Speed-binned learning hooks (called from TorqueEstimator) ---

  @staticmethod
  def _centers_to_bounds(centers):
    """Derive bin bounds from centers using midpoints between consecutive centers.
    First bin starts at default lower edge (5 m/s), last bin ends at default upper edge (40 m/s)."""
    bounds = []
    for i, c in enumerate(centers):
      lo = DEFAULT_SPEED_BIN_BOUNDS[0][0] if i == 0 else (centers[i - 1] + c) / 2
      hi = DEFAULT_SPEED_BIN_BOUNDS[-1][1] if i == len(centers) - 1 else (c + centers[i + 1]) / 2
      bounds.append((lo, hi))
    return bounds

  def _post_reset(self):
    """Initializes per-speed-bin buckets and filters. Must run after factor_sanity/offline values are set."""
    if not self.speed_binned:
      return

    from openpilot.selfdrive.locationd.torqued import TorqueBuckets, STEER_BUCKET_BOUNDS, \
      POINTS_PER_BUCKET, MIN_FILTER_DECAY
    from opendbc.sunnypilot.car.interfaces import get_speed_dep_config

    cfg = get_speed_dep_config().get(self.CP.carFingerprint, {})

    if 'speed_bp' in cfg:
      self.speed_bin_centers = list(cfg['speed_bp'])
      self.speed_bin_bounds = self._centers_to_bounds(self.speed_bin_centers)
    else:
      self.speed_bin_bounds = list(DEFAULT_SPEED_BIN_BOUNDS)
      self.speed_bin_centers = list(DEFAULT_SPEED_BIN_CENTERS)

    n_bins = len(self.speed_bin_bounds)

    # Exactly one store is authoritative; the other stays empty so a stray index
    # is a no-op rather than a silently half-fed learner.
    if self.moment_learner:
      self.speed_bin_points = []
      self.speed_bin_moments = [SpeedBinMoment(STEER_BUCKET_BOUNDS) for _ in range(n_bins)]
    else:
      self.speed_bin_points = [self._make_speed_bin_bucket(TorqueBuckets, STEER_BUCKET_BOUNDS, POINTS_PER_BUCKET) for _ in range(n_bins)]
      self.speed_bin_moments = []
    self._speed_bin_last_len = [0] * n_bins
    self._speed_bin_last_valid = [False] * n_bins

    # Seed values: from TOML if configured, otherwise global offline values for all bins
    ref_lafs = cfg.get('laf_bp', [self.offline_latAccelFactor] * n_bins)
    ref_frictions = cfg.get('friction_bp', [self.offline_friction] * n_bins)
    self.speed_bin_decays = [float(MIN_FILTER_DECAY)] * n_bins
    self.speed_bin_filtered = [
      {'latAccelFactor': FirstOrderFilter(ref_lafs[i], self.speed_bin_decays[i], DT_MDL),
       'frictionCoefficient': FirstOrderFilter(ref_frictions[i], self.speed_bin_decays[i], DT_MDL)}
      for i in range(n_bins)
    ]
    # SVD results get clipped to ±sanity% of seeds to prevent runaway learning
    self.speed_bin_lat_accel_factor_bounds = [
      ((1.0 - self.factor_sanity) * factor, (1.0 + self.factor_sanity) * factor)
      for factor in ref_lafs
    ]
    self.speed_bin_friction_bounds = [
      ((1.0 - self.friction_sanity) * f, (1.0 + self.friction_sanity) * f)
      for f in ref_frictions
    ]

  def _make_speed_bin_bucket(self, TorqueBuckets, STEER_BUCKET_BOUNDS, POINTS_PER_BUCKET):
    """Create a single speed-bin TorqueBuckets instance.
    Per-bucket minimums are scaled down from the global learner since each
    speed bin sees a fraction of the total data."""
    scaled_min = np.maximum(np.asarray(self.min_bucket_points) // len(self.speed_bin_bounds), 1)
    return TorqueBuckets(x_bounds=STEER_BUCKET_BOUNDS,
                         min_points=scaled_min,
                         min_points_total=int(scaled_min.sum()),
                         points_per_bucket=POINTS_PER_BUCKET,
                         rowsize=3)

  def _on_torque_point(self, steer, lateral_acc, vego):
    """Called from handle_log. Routes quality-filtered points to speed bins.

    Hard bin assignment for the point-store learner. The moment learner instead
    feeds every nearby bin with a Gaussian speed weight, so a bin is informed by
    its neighbours rather than starving until its own range is driven — the whole
    reason the top bin can report anything at all."""
    if not self.speed_binned:
      return
    if self.moment_learner:
      for i, center in enumerate(self.speed_bin_centers):
        k = math.exp(-0.5 * ((vego - center) / MOMENT_SPEED_KERNEL_H) ** 2)
        if k >= MOMENT_KERNEL_MIN:
          self.speed_bin_moments[i].add(steer, lateral_acc, k)
      return
    for i, (lo, hi) in enumerate(self.speed_bin_bounds):
      if lo <= vego < hi:
        self.speed_bin_points[i].add_point(steer, lateral_acc)
        break

  def _restore_ext_cache(self, cache_ltp=None):
    """Restores per-bin filter values and points from cache.
    Reads from Params when cache_ltp is not provided; that path also requires the
    global learner's restore key to match (same car, offline baseline, and learner
    version), so the speed bins live and die with the global cache. Passing
    cache_ltp directly bypasses the key gate (test seam)."""
    if not self.speed_binned:
      return
    try:
      from openpilot.selfdrive.locationd.torqued import MIN_FILTER_DECAY, VERSION
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
      # Reject cache from a different config (e.g. TOML update changed bin centers)
      if not np.allclose(list(cache_ltp.speedBinCenters), self.speed_bin_centers, atol=0.01):
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
            if self.moment_learner:
              # Only accept a moment cache; a point cache from the other mode is
              # silently dropped and the bin relearns from the restored seed.
              if len(rows) == 1 and len(rows[0]) == MOMENT_CACHE_ROW:
                self.speed_bin_moments[i].load_cache(rows[0])
            elif not any(len(r) == MOMENT_CACHE_ROW for r in rows):
              self.speed_bin_points[i].load_points(rows)
        # self.decay doesn't exist yet at init time (set by upstream reset()), fallback is intentional
        self.speed_bin_decays = [getattr(self, 'decay', MIN_FILTER_DECAY)] * n_bins
        cloudlog.info("restored speed-bin torque params from cache")
    except Exception:
      cloudlog.exception("speed-dep: failed to restore cache")

  def _estimate_params_moment(self):
    """Fit each bin straight from its moment matrix.

    No output low-pass here: the ESS window already does the smoothing, in the data
    domain rather than on the parameter. Adding upstream's FirstOrderFilter on top
    would throw away the convergence advantage that motivates this path, so the
    sanity-clipped fit is written to the filter's value directly."""
    from openpilot.selfdrive.locationd.torqued import STEER_BUCKET_BOUNDS, FRICTION_FACTOR

    results = []
    for i, mom in enumerate(self.speed_bin_moments):
      if not mom.is_valid():
        results.append((i, False))
        continue

      fit = mom.fit(FRICTION_FACTOR)
      if fit is None:
        # Ill-conditioned despite enough evidence: drop this bin's history and rebuild.
        cloudlog.warning(f"speed-dep moment: bin {i} not identifiable with valid weight, resetting bin")
        self.speed_bin_moments[i] = SpeedBinMoment(STEER_BUCKET_BOUNDS)
        self._speed_bin_last_valid[i] = False
        results.append((i, False))
        continue

      slope, friction_coeff = fit
      factor_lo, factor_hi = self.speed_bin_lat_accel_factor_bounds[i]
      fric_lo, fric_hi = self.speed_bin_friction_bounds[i]
      self.speed_bin_filtered[i]['latAccelFactor'].x = float(np.clip(slope, factor_lo, factor_hi))
      self.speed_bin_filtered[i]['frictionCoefficient'].x = float(np.clip(friction_coeff, fric_lo, fric_hi))
      self._speed_bin_last_valid[i] = True
      results.append((i, True))
    return results

  def _estimate_params_speed_binned(self):
    """Run independent SVD fit per speed bin. Resets bin on NaN with valid data."""
    if self.moment_learner:
      return self._estimate_params_moment()

    from openpilot.selfdrive.locationd.torqued import TorqueBuckets, STEER_BUCKET_BOUNDS, \
      POINTS_PER_BUCKET, FRICTION_FACTOR, FIT_POINTS_TOTAL, slope2rot, MIN_FILTER_DECAY, MAX_FILTER_DECAY

    results = []
    for i, bucket in enumerate(self.speed_bin_points):
      if not bucket.is_calculable():
        results.append((i, False))
        continue

      # Skip bins with no new data since last fit
      cur_len = len(bucket)
      if cur_len == self._speed_bin_last_len[i]:
        results.append((i, self._speed_bin_last_valid[i]))
        continue

      # Same total least squares SVD as upstream's estimate_params()
      points = bucket.get_points(FIT_POINTS_TOTAL)
      try:
        _, _, v = np.linalg.svd(points, full_matrices=False)
        slope, offset = -v.T[0:2, 2] / v.T[2, 2]  # slope = latAccelFactor
        _, spread = np.matmul(points[:, [0, 2]], slope2rot(slope)).T
        friction_coeff = np.std(spread) * FRICTION_FACTOR
        if not any(np.isnan(val) for val in [slope, friction_coeff]):
          factor_lo, factor_hi = self.speed_bin_lat_accel_factor_bounds[i]
          fric_lo, fric_hi = self.speed_bin_friction_bounds[i]
          self.speed_bin_decays[i] = min(self.speed_bin_decays[i] + DT_MDL, MAX_FILTER_DECAY)  # slow down filter over time
          self.speed_bin_filtered[i]['latAccelFactor'].update(np.clip(slope, factor_lo, factor_hi))
          self.speed_bin_filtered[i]['latAccelFactor'].update_alpha(self.speed_bin_decays[i])
          self.speed_bin_filtered[i]['frictionCoefficient'].update(np.clip(friction_coeff, fric_lo, fric_hi))
          self.speed_bin_filtered[i]['frictionCoefficient'].update_alpha(self.speed_bin_decays[i])
          self._speed_bin_last_len[i] = cur_len
          self._speed_bin_last_valid[i] = bucket.is_valid()
          results.append((i, self._speed_bin_last_valid[i]))
          continue
      except np.linalg.LinAlgError:
        pass

      # NaN with valid data = poisoned bucket, reset to recover
      if bucket.is_valid():
        cloudlog.warning(f"speed-dep: bin {i} produced NaN with valid data, resetting bin")
        self.speed_bin_points[i] = self._make_speed_bin_bucket(TorqueBuckets, STEER_BUCKET_BOUNDS, POINTS_PER_BUCKET)
        self.speed_bin_decays[i] = MIN_FILTER_DECAY
        self._speed_bin_last_len[i] = 0
      self._speed_bin_last_valid[i] = False
      results.append((i, False))
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
        # The point store writes N rows of [steer, latAccel]; the moment learner
        # writes a single MOMENT_CACHE_ROW-wide row, which is how restore tells
        # the two cache formats apart.
        bin_points.append([self.speed_bin_moments[i].to_cache()] if self.moment_learner
                          else self.speed_bin_points[i].get_points()[:, [0, 2]].tolist())

    ltp.speedBinCenters = self.speed_bin_centers
    ltp.speedBinLatAccelFactors = lat_factors
    ltp.speedBinFrictions = frictions
    ltp.speedBinValid = valid_flags
    if with_points:
      ltp.speedBinPoints = bin_points
