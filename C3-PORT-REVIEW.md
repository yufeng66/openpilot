# Running current sunnypilot master on the comma three (C3/tici)

**Date:** 2026-07-20 (rev 4 — final; dragonpilot C3 payload fully identified)
**Workspace:** `~/sunnypilot-c3/` (isolated — your `~/openpilot` Palisade fork is untouched)
**Deliverable branch:** `wip-master-c3-tici` in `~/sunnypilot-c3/sunnypilot` (sunnypilot master @ 2c334ede4, 2026-07-20 + 4 commits)

## TL;DR — the final, correct picture

- A C3 **cannot** boot the current AGNOS boot.img (the kernel lost `comma_tici.dts` in
  `agnos-kernel-sdm845@b90695f10`, Aug 2025), but current master userspace **requires** the
  current AGNOS system partition (baked Python deps).
- **Dragonpilot's keystone trick** (base64-obfuscated `restore_partitions()` in their
  `agnos.py`, comment: "Implementation by Rick … please provide proper credit"): on
  non-tizi devices, substitute the **AGNOS 12.8 `boot` + `abl` partitions** (from an added
  `agnos_tici.json`) into the otherwise-stock 18.4 manifest. Result: **hybrid AGNOS — 12.8
  kernel/bootloader (has the tici device tree, device reports `"comma tici"`) + 18.4 system
  partition (new userspace deps)**. `/VERSION` comes from the system partition (18.4), matches
  `AGNOS_VERSION`, so no reflash loop. All images are stock comma CDN artifacts — a composition
  comma never shipped.
- Because the device reports **tici**, sunnypilot's hardwared gate applies: the port branch must
  be named `*-tici` (channel_type derivation) and the restored `device_type == "tici"` code is
  load-bearing. Plain sunnypilot master would not engage on a C3.
- Nobody ships custom-built AGNOS images. comma's clone/authorization lock is userspace
  (registration-required-for-onroad), not the OS.

## The 4 commits on `wip-master-c3-tici`

1. `Revert "camerad: remove AR0231 (#36070)"` — early-C3 cameras, probe-based (dragonpilot
   ships the same restoration).
2. `Revert "remove tici-specific code (#36078)"` — `device_type == "tici"` branches: amplifier
   CONFIGS (C3 mono speaker EQ), USB IRQ affinity, NVMe storage alert, params key, ui_state,
   pins, tici_reset. Active at runtime (device reports tici).
3. `tici fixups` — soundd volume, UI FPS cap.
4. `agnos: hybrid manifest for C3` — static version of dragonpilot's trick: stock 18.4 manifest
   with `boot`+`abl` replaced by AGNOS 12.8's entries (this branch is C3-only, so no runtime
   selection needed; credited to Rick Lan / dragonpilot in the commit message).
   **⚠ Never install this branch on a 3X/C4 — the mixed manifest is only correct for a C3.**

Sources: sunnypilot's `sync-20251218-tici` (commits `5ac160222`, `3ef5e531e`, `a3797eb7a`) for
1–3; `dragonpilot/dragonpilot@pre-build` `agnos_tici.json` for 4.

## Dragonpilot's exact C3 payload (fully identified)

Located by layer-diffing their assembled branch (root `6adb63b91` = byte-identical comma
0.11.1 → core squash `9ca16517e` → **min squash `6a8db4193` ← everything C3 is here** → brand
squash `f00bae913` → fixes). **No public feature branch carries it** — the hw/C3 source branch
is unpublished; only squashed outputs (`pre-build`, `testing`) contain it. Payload:

| Piece | Size | Purpose |
|---|---|---|
| `agnos.py` `restore_partitions` + `agnos_tici.json` | ~30 lines + manifest | THE keystone: hybrid 12.8-boot/18.4-system AGNOS (obfuscated, credit requested) |
| `system/camerad/` AR0231 revert (+ spectra.cc, bps_blobs.h) | ~463 lines | early-C3 / clone cameras |
| `panda_tici/` (264 files) + `selfdrive/pandad_tici/` (14 files) | vendored old panda+pandad | F4 ("dos") clone pandas + aux/second USB panda |
| `launch_chffrplus.sh` additions | 158 lines | panda MCU fingerprint (→`TICI_DOS`/`TICI_TRES`), aux-panda USB-OTG probe, `mount_nvme`, `set_lite_hw` (amp probe → `LITE`), model-selector env |
| tici HAL mods | small | hardware.py, amplifier.py, pins.py, modem.py, power_monitoring.py, hardwared.py (incl. commented-out `registered_device` onroad gate), soundd/application/ui_state |
| clone shims | ~20 lines | registration `LITE` skip, no-IMEI fallback → `UnregisteredDevice` |

O3/O3L/O3XL = clone hardware (README: 副廠硬體). Code-fingerprinted differences vs real C3:
F4 internal panda (some), missing amp (`LITE`), missing modem/IMEI, NVMe, AR0231-era cameras.
comma's OS boots faithful clones; comma's userspace registration gate is what excludes them.

## AGNOS state of the world

| Who | What a C3 actually runs | Images |
|---|---|---|
| comma master / release-tizi | n/a (18.5 / 18.4, tizi+mici only) | comma CDN |
| comma release-tici LTS | pure 12.8 | comma CDN |
| sunnypilot master | n/a for C3 (stock 18.4 manifest won't boot one) | comma CDN |
| sunnypilot `*-tici` (frozen Oct 2025) | pure 12.8 | comma CDN |
| dragonpilot 0.11.1 | **hybrid: 12.8 boot/abl + 18.4 rest** (runtime substitution) | comma CDN |
| our `wip-master-c3-tici` | **hybrid, static manifest** (same composition) | comma CDN |

Future cliff: comma's vamOS lists only C3X/C4; also the hybrid depends on 12.8's kernel staying
ABI-compatible with new system partitions. If that ever breaks, the real fix is building AGNOS
with the dts revert (agnos-builder + `b90695f10` revert; sunnypilot's dormant
`agnos-builder@comma_three` branch is prior art).

## What's left

1. On-device sanity (RAV4 C3 on dragonpilot, ssh): `cat /VERSION` → expect 18.4;
   `cat /sys/firmware/devicetree/base/model` → expect `comma tici`. (Confirms the hybrid.)
2. Push `wip-master-c3-tici` to a personal fork **keeping the `-tici` suffix**; install on the
   C3; compiles on device. Because the device already carries dp's hybrid boot chain and
   `/VERSION` is 18.4, no AGNOS reflash should trigger; a fresh/stock-14.x device would get the
   hybrid flashed by our manifest.
3. Watch-items: pandad USB timing on first boot (old workaround dropped; re-add
   `no_internal_panda_count` reset from `3ef5e531e` if the internal panda is flaky in the first
   minute); registration must succeed (real comma device — fine).
4. RAV4 = TSS2 regular CAN, internal tres panda suffices. The Palisade CAN FD case needs aux
   red panda support — upstream removed it; dragonpilot solves via `pandad_tici`; sunnypilot
   state unknown. Separate project.

## Workspace map

- `~/sunnypilot-c3/sunnypilot` — remotes: `origin`=sunnypilot, `commaai`=comma; local refs
  `dp-devel-staging`, `dp-pre-build`, `comma-release-tizi`; branch `wip-master-c3-tici` (local).
- `~/sunnypilot-c3/agnos-builder`, `~/sunnypilot-c3/panda` (+`sp` remote).
- Scratchpad: `dp-delta.txt` (base vs comma: 0 modified), `dp-stack-delta.txt` (557 paths),
  `min-layer.txt` (the C3 layer), `dp-launch.diff`, `agnos_tici.json`.
- Key refs: `sync-20251218-tici` reverts `5ac160222`/`3ef5e531e`/`a3797eb7a`; kernel dts removal
  `agnos-kernel-sdm845@b90695f10`; dp layers `6adb63b91`→`9ca16517e`→`6a8db4193`→`f00bae913`.
