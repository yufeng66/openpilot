# Code review — lateral torque changes

**Status:** closed 2026-08-24 — all actionable findings (P1, P5, P7) fixed on both branches;
P2/P3/P6 were document-only. The review below is the original pre-fix text.

| | |
|---|---|
| Primary subject | `dp-testing` commit `c6f488a6c8` "feat(lat): moment-learner update, Friction Reduction, jerk torque controller" |
| Worktree | `/home/yfeng/openpilot-dp` |
| Follow-up subject | `pal23sp-hda2-big-testing` (Palisade 3X), worktree `/home/yfeng/openpilot`, HEAD `1faa0772d8` |
| Reference sources | sunnypilot originals under `/home/yfeng/openpilot/openpilot/sunnypilot/selfdrive/controls/lib/` |
| Reviewed by | manual pass + `/code-review max` agent, findings cross-checked and several reproduced by execution |

---

## 1. Headline

Two things to act on before the jerk controller is armed anywhere:

- **P1** — the shared PID integrator is written twice per frame under two different limit
  spaces, so anti-windup is anchored above the real actuation ceiling. Reproduced on
  **both** branches, worse on the Palisade branch.
- **P5 / P7** — cheap, self-inflicted, unrelated to P1. Fix whenever.

Everything else is either inherited from sunnypilot/twilsonco verbatim (leave it — it is
the tune that was validated on the road) or cosmetic.

**The moment-matrix bin learner itself is clean.** Only P7 comes from it, and it is latent.
Learner-on / jerk-off was verified good on the Palisade branch by execution.

---

## 2. How this was verified

```bash
# dp-testing
cd /home/yfeng/openpilot-dp && source .venv/bin/activate
python -m pytest --noconftest -o addopts="" -q \
  dragonpilot/selfdrive/controls/lib/tests/test_lat_torque_settings.py \
  dragonpilot/selfdrive/controls/lib/tests/test_latcontrol_torque_jerk_aware.py \
  dragonpilot/selfdrive/controls/lib/tests/test_latcontrol_torque_jerk_integration.py \
  dragonpilot/selfdrive/controls/lib/tests/test_speed_dep_helpers.py \
  dragonpilot/selfdrive/locationd/tests/test_torqued_speed_dep.py
# -> 112 passed
ruff check $(git show --name-only --format="" c6f488a6c8)   # -> All checks passed

# pal23sp-hda2-big-testing (imports need opendbc on the path; cereal has no `car` here,
# it lives in opendbc.car.structs)
source .venv/bin/activate
PYTHONPATH=/home/yfeng/openpilot:/home/yfeng/openpilot/opendbc_repo python <script>
```

Repro scripts are in Appendix A.

---

## 3. Findings — `dp-testing` `c6f488a6c8`

Provenance tags:
`[SP]` inherited verbatim from sunnypilot/twilsonco ·
`[PORT]` introduced by this port ·
`[LEARNER]` from the moment-matrix bin learner

### P1 `[PORT]` — integrator winds past the torque-space bound

`selfdrive/controls/lib/latcontrol_torque.py:97` + `dragonpilot/.../latcontrol_torque_jerk_aware.py:45-57`

`common/pid.py:51-53` freezes the integrator only when the resulting control exceeds
`pos_limit`:

```python
i_upperbound = self.i if test_control > self.pos_limit else self.pos_limit
self.i = np.clip(i, i_lowerbound, i_upperbound)
```

Per-frame sequence with the toggle armed:

1. `controlsd.py:101` `update_live_torque_params()` -> host `update_limits()` -> `pos_limit = steer_max * latAccelFactor` (2.5)
2. host `pid.update()` at ±2.5 — the guard never engages, `i` grows in a space 2.5x looser than where it is applied
3. `extension.update_limits()` -> ±1.0
4. extension `pid.update()` at ±1.0 — can only *freeze* `i` (`i_upperbound = self.i`), never reduce it

Measured (LAF 2.5, v 25 m/s, small steady tracking error, 3000 frames):

| ordering | final `i` | torque | first pinned at `steer_max` |
|---|---|---|---|
| dp (limits set inside `extension.update()`) | **1.0510** | −1.0000 | frame 319 |
| sunnypilot (`extension.update_limits()` from caller) | 0.4204 | −0.9997 | never |

The commit message's invariant — "each PID call is clipped in its own space" — is the wrong
one. The integrator is *shared*, so the tighter torque-space bound has to govern **both**
writes. Stock sunnypilot with its speed-dep off does exactly that and does not wind up.

`LatControlTorque` has no `reset()` override (`LatControl.reset()` only zeroes `sat_time`),
so the over-wound integrator survives the `self.LaC.reset()` at `controlsd.py:141`.

**Fix direction:** set the torque-space bound *before* the host's PID call — restore
sunnypilot's caller-side ordering, or call it at the top of `LatControlTorque.update()`.

### P2 `[SP]` — `lat_accel_friction_factor` latches 0.7 -> 1.0 permanently

`dragonpilot/selfdrive/controls/lib/latcontrol_torque_ext_base.py:168-170`

```
init                       = 0.7
after ONE sign-flip plan   = 1.0
after 500 more clean plans = 1.0   <-- never returns to 0.7
friction_input(err=0.5): 0.5 now vs 0.35 as designed   (+43%)
```

`update_calculations` writes an `__init__`-time constant with no reset path.
`sign(0.0) == 0.0` counts as a flip, so near-zero plan noise on a straight road latches it
within seconds. The 0.7 initializer is effectively dead after the first few frames.

Byte-identical to sunnypilot, so it **is** the validated tune. **Do not fix** — document it
as a third inherited quirk alongside the two the commit already calls out.
`test_sign_flip_in_the_window_zeroes_the_jerk_term` currently pins the latched value as if
intended.

### P3 `[SP]` — `model_valid` checks the wrong array

`latcontrol_torque_ext_base.py:138-140`

Gates on `len(orientation.x) >= CONTROL_N` (17) but consumes `acceleration.y` against
`t_diffs` (32) and `T_IDXS` (33). A 17-length plan passes the gate and raises
`ValueError: operands could not be broadcast together with shapes (16,) (32,)` out of
`Controls.state_control()`. Every shipped model publishes 33, so this is hardening, not a
live bug. Guard should be `len(acceleration.y) == len(ModelConstants.T_IDXS)`.

### P4 `[PORT]` — per-car torque conversion is lost

`latcontrol_torque_ext_base.py:98`

sunnypilot routes through `CI.torque_from_lateral_accel_in_torque_space()`; the port
hardcodes `la / latAccelFactor`. dp's `opendbc/car/gm/interface.py:67-75` *does* return a
non-linear sigmoid+linear callback for cars in `NON_LINEAR_TORQUE_PARAMS`. GM is not in the
settings `brands` list nor in torqued's `ALLOWED_CARS`, and `latcontrol_torque_jerk_aware.py:39`
reads the param with no brand check — so it needs the param set out-of-band to bite. Low
reachability; deliberate and documented in the commit.

### P5 `[PORT]` — both new settings descriptions can never be translated

`dragonpilot/settings/min-feat.lat.jerk-torque.py:9`, `min-feat.lat.torqued-sd.py:21`

`dragonpilot/selfdrive/ui/update_translations.py:18` extracts with
`xgettext -L Python --keyword=tr`, which records string *literals*. `tr("A " + "B")` does not
yield the runtime-joined msgid, so `multilang.tr()` misses and the text stays English in
every locale. These two are the **only** `tr(... + ...)` call sites in all of `dragonpilot/`.

Fix: drop the `+`, use implicit adjacent concatenation.

*Not confirmed locally* — xgettext is not installed in WSL and no `.pot` is checked in.
Structural evidence only. Verify on a machine with gettext before/after.

### P6 `[SP]` — `pid_log.error` silently changes units when armed

`latcontrol_torque_jerk_aware.py:96`

Becomes torque while `desiredLateralAccel` / `actualLateralAccel` in the same entry stay
m/s², and `pid_log.version` is still `VERSION = 1`. Any log analysis or replay comparison
mixes two unit systems across drives depending on a param the log does not carry. Bump the
version or add a flag.

### P7 `[LEARNER]` — density-ceiling latents

`dragonpilot/selfdrive/locationd/torqued_ext.py:58` and `:127`

- `MOMENT_DENSITY_CEILING_V = (MOMENT_SPEED_BIN_CENTERS[0], MOMENT_SPEED_BIN_CENTERS[-1])`
  anchors the ramp to the **default** grid, but `_post_reset()` can build the bank from
  `SPEED_DEP_CAR_CONFIG[...]['speed_bp']`. The docstring says "anchored at the grid ends";
  it is anchored at the default grid's ends. A per-car grid spanning e.g. 10-30 m/s would
  get ~7.9 at its low bin (no longer bit-identical to the validated flat behaviour) and
  ~13.6 at its top. Fix: `np.interp(centers, (centers[0], centers[-1]), ...)`.
- `SpeedBinMoment` still defaults to flat `MOMENT_DENSITY_CEILING = 7.0` while
  `SpeedBinMomentBank` defaults to the ramp. `SpeedBinMoment` is both the parity oracle and
  the cache-row decoder for offline analysis, so any `learner_analysis` replay that does not
  hand-pass `density_ceiling=` silently reproduces the pre-change weighting for highway
  bins. `test_torqued_speed_dep.py:728` now has to pass it by hand. Give it the same
  `None -> moment_density_ceiling(...)` sentinel.

Both latent today (`SPEED_DEP_CAR_CONFIG` is `{}`). The offline-replay half is the one that
actually matters.

### P8 `[SP]` mostly — cleanup

- `_desired_curvature`, `_actual_curvature` (`ext_base.py:112-113`) — initialized, never written or read
- `_roll_compensation`, `_gravity_adjusted_lateral_accel` — written in `update()`, never read (passed as parameters instead)
- `actual_lateral_jerk`, `lateral_jerk_setpoint`, `lateral_jerk_measurement` — computed every frame, consumed by nothing (NNFF leftovers). `test_actual_jerk_tracks_steering_rate_and_speed` pins a value that does not reach the output. Side effect: the `self.actual_lateral_jerk = 0.0` half of the P2 latch block is dead too, so `lat_accel_friction_factor = 1.0` is that branch's only live effect.
- `jerk_aware.py:86` `if not self._jerk_aware_enabled: return` — unreachable, `update()` already returned at line 64 `[PORT]`
- unused `CI` arg (`ext_base.py:86`) and `self.params` (`jerk_aware.py:36`) `[PORT]`
- `update_calculations` recomputes the 33-element `np.diff`/`tolist` chain at 100 Hz for a 20 Hz signal. Only `predicted_lateral_jerk` can be hoisted into `update_model_v2` — `desired_lateral_jerk` genuinely changes per frame.
- `get_friction_in_torque_space` is exactly `opendbc.car.lateral.get_friction(...) / latAccelFactor` (np.interp is linear in `fp`), so the deadzone/threshold logic is duplicated.

---

## 4. Findings — `pal23sp-hda2-big-testing` (Palisade)

**Same class of defect as P1, worse form — but only when `LateralJerkTorqueController` is on.**

Reproduced on the branch:

| config | max \|i\| | final torque | exceeds `steer_max`? |
|---|---|---|---|
| stock sunnypilot (speed-dep off, jerk on) | 0.4204 | −0.9997 | no |
| **learner on + jerk on (Palisade config)** | **1.9077** | **−2.4870** | **yes, 2.5x** |
| learner on + **jerk off** | 1.0509 | −0.9996 | **no — clean** |
| dp-testing for comparison (jerk on) | 1.0510 | −1.0000 | no (pinned at 1.0) |

### Mechanism

dp splits the bounds (host ±2.5 / extension ±1.0), so the integrator over-winds but the
output is still clipped to `steer_max`. The Palisade branch puts **both** writes at
lat-accel bounds:

1. `openpilot/selfdrive/controls/controlsd.py:100` `extension.update_limits()` -> ±1.0
2. `LatControlTorque.update()` line 2: `if self.extension.update_override_torque_params(...): self.update_limits()` -> back to **±2.5**
3. both PID writes then run at ±2.5

So the torque-space PID is never bounded by `steer_max` at all and returns −2.49.

Clearest statement of the bug: `i = 1.05` is a *correct* integrator value in lat-accel space
(bound 2.5); the stock loop divides by `latAccelFactor` and gets 0.42 torque. Arming the
jerk controller makes that same number get consumed directly as torque, where the bound is
1.0. Same value, wrong space.

### It fires every frame, not intermittently

`update_override_torque_params` (`latcontrol_torque_ext_override.py:38`) only returns True
when the interpolated values differ — but:

```
stored friction (capnp float32) = 0.11999999731779099
interp friction (python float64)= 0.12
fric differs? True   <- changed=True on this alone, every frame
```

`torque_params` is a capnp builder, so `friction` round-trips through float32 while the
interp result is float64. For any friction not exactly representable in float32 that
comparison can never be equal — so the limit reset fires every frame of every drive, at
constant speed, with a flat friction table.

### Real-world consequences

The car is not handed a >100% command — the Hyundai carcontroller clips to `STEER_MAX`
downstream. What would actually be observable:

- **Anti-windup never engages** — `pos_limit` is anchored 2.5x above the real ceiling, so
  `i` winds freely to ~2x `steer_max`.
- **Saturation alerts** — `pid_log.saturated` uses `steer_max - abs(output_torque) < 1e-3`,
  true from the moment |output| ≥ 1.0, so `sat_time` accumulates toward `steerLimitTimer`.
- **Curve-exit overshoot** — the wound integrator keeps commanding full torque after the
  error resolves.

### Provenance nuance

The flaw is sunnypilot's own ordering and is **latent** on stock sunnypilot (speed-dep off
-> `update_override_torque_params` returns False -> limits stay ±1.0). It only goes live
because `_speed_dep_active` stays true, which is the fork's moment learner. So on this
branch it *is* learner-activated — unlike on dp, where the port's re-ordering causes it.
Not caused by the moment-matrix math either way.

### Fix direction (different from dp's)

`update_override_torque_params` should not re-widen the host limits while the extension is
armed, and that float32 comparison should use a tolerance regardless.

### This is live on the car, not hypothetical

Session memory records the Palisade reference config for the drive that was liked:
`EnforceTorqueControl=1`, `TorqueControlTune=1.0` (v1, not v0), **`LateralJerkTorqueController=1`**,
`FrictionReduction=1`, NNLC off. Combined with the moment learner running (14/15 bins valid
as of 2026-08-23), that is exactly the "learner on + jerk on" row above.

`EnforceTorqueControl=1` does not change the outcome: `update_override_torque_params` still
returns the `changed` flag set by the speed-dep block regardless of that toggle's branch.

Worth re-confirming against the device before acting, since that config is from an earlier
session:

```bash
ssh c3x 'cat /data/params/d/LateralJerkTorqueController; echo; cat /data/params/d/dp_lat_torqued_sd'
```

---

## 5. Verified clean

- **Port fidelity.** `latcontrol_torque_ext_base.py` is byte-identical to sunnypilot's below
  the header/`__init__`. `get_friction_in_torque_space` matches
  `opendbc/sunnypilot/car/lateral_ext.py` exactly and its docstring is right (stock opendbc
  `get_friction` *does* multiply by `latAccelFactor`; the torque-space twin does not). The
  13-arg `extension.update(...)` call — including the doubled `measurement` — maps
  identically to sunnypilot v1. The shared-PID choice is faithful: sunnypilot's
  `PIDController(KP, KI)` in `__init__` is dead there too, overwritten by `self._pid = pid`.
- **ESS cap 6000 -> 24000.** Cache-compatible in the direction taken (`load_cache` rejects
  `s > ess_cap * 1.01`; old rows carry S ≤ 6000). Half-life arithmetic in the comments checks
  out: ~3.5 min at 6000, ~13.9 min at 24000, ~20 min saturation at 20 Hz.
- **The ramped ceiling does NOT relax bin readiness.** Simulated it because it looked like a
  risk — `S` and `x_std` both get easier, but the binding gate is `dens[:,1:-1] >= 3.0`, and
  `dens` advances by raw kernel weight, independent of the ceiling. Identical validity timing
  at ceiling 7 vs 15 (320 points either way on a narrow highway distribution; 87 on a wider
  one). **Do not act on this.**
- **Friction Reduction.** `friction_scale` clamps to [0,9]; learned bins only; inert with the
  learner off; correctly threaded controlsd -> `interp_live_torque_params` -> `build_speed_dep_bp`.
- **Params plumbing.** `Params.get(..., return_default=True)` returns a typed int for INT
  keys, so `int(... or 0)` is safe. Settings schema fields are all in `_KNOWN_ITEM_KEYS`.
  Toggle `brands` list matches torqued's `ALLOWED_CARS` exactly
  (`toyota, hyundai, rivian, honda, volkswagen`), so `useParams` is true for every car that
  can enable the toggle. `controlsd` really is `started`-gated, so "read once per drive" holds.
- 112 tests pass across the five touched suites; ruff clean on all 13 changed files (the 45
  tree-wide ruff hits are pre-existing, e.g. `selfdrive/controls/tests/test_alka.py`).

---

## 6. Discounted / disputed

Raised by the review agent, judged not actionable as stated:

- **Friction discontinuity at learned/seeded boundaries.** The seed-path half is right but
  inert (`SPEED_DEP_CAR_CONFIG` is `{}`). The claim that "the same applies to the
  `any(valid_bp)` branch" is **wrong** — that branch keeps learned bins only and scales all
  of them, so there is no discontinuity there.
- **`self.LaC.extension` guarded on `lateralTuning.which()` rather than `isinstance`.** Real
  but **pre-existing** — upstream openpilot guards `update_live_torque_params` identically.
  Not introduced by this commit.
- **`sign()` duplicating `np.sign`.** Byte-faithful to sunnypilot and faster on Python
  floats. Leave it. (The `get_friction_in_torque_space` duplication half is fair — see P8.)

---

## 7. Next actions

1. **P1, both branches** — fix before the jerk toggle is armed anywhere. dp: restore
   caller-side `update_limits()` ordering. Palisade: stop `update_override_torque_params`
   re-widening while armed, plus a tolerance on the float32 compare.
   Regression test: pin `|pid.i| <= steer_max` across a long steady-error run.
2. **Check the device** — is `LateralJerkTorqueController` actually on? (command in §4)
3. **P5, P7** — cheap, self-inflicted, safe to fix any time.
4. **P2, P3, P6** — document, do not fix. P2 especially: fixing it moves off the road-validated tune.

---

## Appendix A — repro scripts

All four verified re-runnable 2026-08-23, reproducing the numbers quoted above.
**Use the venv of the branch being tested** — A.1/A.2 need `/home/yfeng/openpilot-dp/.venv`,
A.3/A.4 need `/home/yfeng/openpilot/.venv`. Running A.3 under the dp venv fails with
`OSError: .../common/libparams_c.so: cannot open shared object file`.

### A.1 dp-testing: P1 ordering comparison

```
cd /home/yfeng/openpilot-dp && source .venv/bin/activate && python <<'EOF'
import sys, numpy as np
sys.path.insert(0,'.')
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.modeld.constants import ModelConstants

P='dragonpilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware.Params'
LAF, FRIC, SR, WB = 2.5, 0.12, 15.0, 2.7

def make_CP():
    CP=car.CarParams.new_message(); CP.steerLimitTimer=0.4; CP.steerActuatorDelay=0.12; CP.steerRatio=SR
    t=CP.lateralTuning.init('torque'); t.latAccelFactor=LAF; t.latAccelOffset=0.0; t.friction=FRIC
    t.steeringAngleDeadzoneDeg=0.0; return CP.as_reader()
def make_CI():
    CI=MagicMock(); CI.torque_from_lateral_accel.return_value=lambda la,tp: la/tp.latAccelFactor
    CI.lateral_accel_from_torque.return_value=lambda t,tp: t*tp.latAccelFactor; return CI
def make_lac(on):
    with patch(P) as mp:
        mp.return_value.get_bool.side_effect=lambda p: on
        return LatControlTorque(make_CP(), make_CI(), DT_CTRL)
def VM():
    vm=MagicMock(); vm.calc_curvature.side_effect=lambda a,v,r: a/(SR*WB); return vm
def flat_model():
    n=len(ModelConstants.T_IDXS)
    return SimpleNamespace(acceleration=SimpleNamespace(y=[0.0]*n), orientation=SimpleNamespace(x=[0.0]*33))

def run(sunnypilot_order, angle_deg=1.0, dc=0.0006, frames=3000, v=25.0):
    lac=make_lac(True); lac.extension.update_model_v2(flat_model())
    CS=SimpleNamespace(vEgo=v,aEgo=0.0,steeringAngleDeg=angle_deg,steeringRateDeg=0.0,steeringPressed=False)
    imax=0.0; sat=None
    for f in range(frames):
        lac.update_live_torque_params(LAF,0.0,FRIC)
        if sunnypilot_order: lac.extension.update_limits()
        t,_,_=lac.update(True, CS, VM(), SimpleNamespace(roll=0.0,angleOffsetDeg=0.0), False, dc, False, 0.15)
        imax=max(imax, abs(lac.pid.i))
        if sat is None and abs(t)>=lac.steer_max-1e-9: sat=f
    return lac.pid.i, imax, t, sat

for lbl, order in (("dp", False), ("sunnypilot", True)):
    i, imax, t, sat = run(order)
    print(f"{lbl:<12} final i={i:8.4f}  max|i|={imax:8.4f}  torque={t:8.4f}  sat@{sat}")
EOF
```

Expected: `dp` -> i 1.0510, sat@319. `sunnypilot` -> i 0.4204, sat None.

### A.2 dp-testing: P2 friction-factor latch

```
cd /home/yfeng/openpilot-dp && source .venv/bin/activate && python <<'EOF'
import sys, numpy as np
sys.path.insert(0,'.')
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from openpilot.selfdrive.modeld.constants import ModelConstants
from dragonpilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware import LatControlTorqueJerkAware

P='dragonpilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware.Params'
lac = SimpleNamespace(torque_params=SimpleNamespace(latAccelFactor=2.5, friction=0.12, latAccelOffset=0.03), steer_max=1.0)
with patch(P) as mp:
    mp.return_value.get_bool.side_effect = lambda p: True
    ext = LatControlTorqueJerkAware(lac, SimpleNamespace(steerActuatorDelay=0.12), MagicMock())

def model(a): return SimpleNamespace(acceleration=SimpleNamespace(y=list(a)), orientation=SimpleNamespace(x=[0.0]*33))
def CS(): return SimpleNamespace(vEgo=25.0,aEgo=0.0,steeringRateDeg=10.0,steeringPressed=False)
def VM():
    vm=MagicMock(); vm.calc_curvature.return_value=0.001; return vm

n=len(ModelConstants.T_IDXS)
clean=list(np.linspace(0.0,3.0,n))
flip=[0.0]*n
for i in range(5,n): flip[i] = 1.0 if i%2 else -1.0

print("init                       =", ext.lat_accel_friction_factor)
ext.update_model_v2(model(clean)); ext.update_calculations(CS(), VM(), 0.0)
print("after clean plan           =", ext.lat_accel_friction_factor)
ext.update_model_v2(model(flip));  ext.update_calculations(CS(), VM(), 0.0)
print("after ONE sign-flip plan   =", ext.lat_accel_friction_factor)
for _ in range(500):
    ext.update_model_v2(model(clean)); ext.update_calculations(CS(), VM(), 0.0)
print("after 500 more clean plans =", ext.lat_accel_friction_factor)
ext.lookahead_lateral_jerk = 0.0
print("friction_input(err=0.5) now         =", ext.update_friction_input(1.0, 0.5))
ext.lat_accel_friction_factor = 0.7
print("friction_input(err=0.5) as designed =", ext.update_friction_input(1.0, 0.5))
EOF
```

### A.3 Palisade branch: speed-dep on/off comparison

```
cd /home/yfeng/openpilot && source .venv/bin/activate
PYTHONPATH=/home/yfeng/openpilot:/home/yfeng/openpilot/opendbc_repo python <<'EOF'
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.modeld.constants import ModelConstants

LAF, FRIC, SR, WB = 2.5, 0.12, 15.0, 2.7
PATCHES = ['openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware.Params',
           'openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc.Params',
           'openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override.Params']
def flags(k): return {'LateralJerkTorqueController': True}.get(k, False)
def make_CP():
    CP=car.CarParams.new_message(); CP.steerLimitTimer=0.4; CP.steerActuatorDelay=0.12; CP.steerRatio=SR
    t=CP.lateralTuning.init('torque'); t.latAccelFactor=LAF; t.latAccelOffset=0.0; t.friction=FRIC
    t.steeringAngleDeadzoneDeg=0.0; return CP.as_reader()
def make_CI():
    CI=MagicMock()
    CI.torque_from_lateral_accel.return_value=lambda la,tp: la/tp.latAccelFactor
    CI.lateral_accel_from_torque.return_value=lambda t,tp: t*tp.latAccelFactor
    CI.torque_from_lateral_accel_in_torque_space.return_value=(
        lambda inp,tp,gravity_adjusted: inp.lateral_acceleration/float(tp.latAccelFactor))
    return CI
def make_lac():
    ps=[patch(p) for p in PATCHES]
    for p in ps:
        m=p.start(); m.return_value.get_bool.side_effect=flags; m.return_value.get.side_effect=lambda k,**kw: None
    sp=MagicMock(); sp.neuralNetworkLateralControl.model.path=''
    lac=LatControlTorque(make_CP(), sp, make_CI(), DT_CTRL)
    for p in ps: p.stop()
    return lac
def VM():
    vm=MagicMock(); vm.calc_curvature.side_effect=lambda a,v,r: a/(SR*WB); return vm

def run(speed_dep_on, frames=3000):
    lac=make_lac(); ext=lac.extension
    n=len(ModelConstants.T_IDXS)
    ext.update_model_v2(SimpleNamespace(acceleration=SimpleNamespace(y=[0.0]*n),
                                        orientation=SimpleNamespace(x=[0.0]*33)))
    ext.update_lateral_lag(0.15)
    if speed_dep_on:
        ext._speed_dep_active=True; ext._speed_dep_speed_bp=[5.,20.,40.]
        ext._speed_dep_lat_accel_factor_bp=[LAF]*3; ext._speed_dep_friction_bp=[FRIC]*3
    CS=SimpleNamespace(vEgo=25.,aEgo=0.,steeringAngleDeg=1.0,steeringRateDeg=0.,steeringPressed=False)
    P=SimpleNamespace(roll=0., angleOffsetDeg=0.); imax=0.; sat=None; t=0.
    for f in range(frames):
        lac.update_torque_parameters(LAF,0.,FRIC)
        lac.extension.update_limits()
        t,_,_=lac.update(True, CS, VM(), P, False, 0.0006, None, False, 0.15)
        imax=max(imax,abs(lac.pid.i))
        if sat is None and abs(t)>=lac.steer_max-1e-9: sat=f
    return lac,imax,t,sat

for lbl,on in (("speed-dep OFF", False), ("speed-dep ON ", True)):
    lac,imax,t,sat=run(on)
    print(f"{lbl}: limits=(+{lac.pid.pos_limit:.2f},{lac.pid.neg_limit:.2f}) max|i|={imax:.4f} "
          f"torque={t:.4f} exceeds_steer_max={abs(t)>lac.steer_max+1e-9} sat@{sat}")
EOF
```

Expected: OFF -> max|i| 0.4204, torque −0.9997, no exceed. ON -> max|i| 1.9077, torque −2.4870, exceeds.

### A.4 Palisade branch: the float32 compare

```
cd /home/yfeng/openpilot && source .venv/bin/activate
PYTHONPATH=/home/yfeng/openpilot:/home/yfeng/openpilot/opendbc_repo python <<'EOF'
import numpy as np
from opendbc.car.structs import car
CP=car.CarParams.new_message(); t=CP.lateralTuning.init('torque')
t.latAccelFactor=2.5; t.friction=0.12; t.latAccelOffset=0.0
tp=CP.as_reader().lateralTuning.torque.as_builder()
new_fric=float(np.interp(25.0,[5.,20.,40.],[0.12,0.12,0.12]))
print("stored friction (capnp float32) =", repr(tp.friction))
print("interp friction (python float64)=", repr(new_fric))
print("fric differs? ", new_fric != tp.friction)
EOF
```
