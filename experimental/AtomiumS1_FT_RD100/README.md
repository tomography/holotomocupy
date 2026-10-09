# AtomiumS1_FT_RD100 — single-distance FT, ±100 px random displacement, 4.5 nm voxels

`Atomium_S1_FT_4K_RD100_004p5nm_0001`, ESRF ID16A visit **blc17322**, beamtime
20260825, collected 2026-08-29. On eagle at

```
/eagle/APS_IRI/vnikitin/20260829/AtomiumS1/
```

12000 projections over 180°, **one** propagation distance, 4096² frames at a
4.5 nm voxel. Same sample, optic and energy as every other AtomiumS1 scan —
**the only difference is the commanded random-displacement amplitude**, ±100 px
here. The FT arm of the sweep is RD000 / RD025 / RD100 / RD300 = ±0 / ±25 /
±100 / ±300 detector px, and the point of this directory is that comparison.

**[`../AtomiumS1_FT_RD300/README.md`](../AtomiumS1_FT_RD300/README.md) is the
reference for how the pipeline works.**
**[`../AtomiumS1_FT_RD000/README.md`](../AtomiumS1_FT_RD000/README.md) covers
everything this scan shares with the rest of the FT arm** — the `ewoks`
fallback, the absent ESRF drop, the shift model, the two-pass ladder and the
re-submission hazard. This README records only what is specific to ±100 px.

## What arrived

```
<pfile>_1_/           12064 files: 12003 EDF + 40 ref + 21 dark + sidecars
<pfile>/projections/  <pfile>_0001.{nx,txt,json}   12003-row shift table
```

12003 projection frames = 12000 scan frames + three post-scan retakes at
ω = 180 / 90 / 0. The shift table has the matching 12003 rows, the last three
explicitly `0.000 -0.000` — the piezo is parked for the anchor frame and both
retakes. No padding needed.

Measured amplitude in the table, detector px:

| | min | max | ptp |
|---|---|---|---|
| x | −99.97 | +99.97 | 199.95 |
| y | −100.00 | +99.95 | 199.95 |

so "RD100" is the half-amplitude, as the name implies.

No master `<pfile>.nx`; all 124 NXtomo virtual sources fail to resolve, so
`Layout` falls back to **`ewoks`** — geometry from the NXtomo, frames from the
EDFs. Frames spot-checked non-zero with
[`../AtomiumS1_HT_RD300/check_frames_nonzero.py`](../AtomiumS1_HT_RD300/check_frames_nonzero.py).

### Geometry, from the `.info` sidecar

| plane | SourceDistance (mm) | Distance (mm) | PixelSize (µm) | norm_mag |
|---|---|---|---|---|
| 1 | −3.69806 | 1209.28 | 0.0045000 | 1.000000 |

Energy 17.1 keV, detector pixel 1.476015 µm, voxel 4.5000 nm, `Dim` 4096,
`TOMO_N` 12000, `ScanRange` −180. Identical to the other FT scans to five
decimals, which is the point.

## There is no ESRF processing for this scan

Only FT_RD300, HT_RD025 and HT_RD300 have a `<pfile>_` drop directory. **This
scan has none**: no `rhapp.mat`, no `quali.mat`, no `reference_motion.mat`, no
`naburec/`, no `correct_motion.txt`. Nothing is inherited and nothing can be
validated against ESRF — every number is measured by this pipeline. Read the
step-3 and step-4b log lines rather than assuming.

## The shift model, as configured

```python
shifts_final = random_shifts + motion_shifts + rhapp_shifts
# + the step-4b rotation axis, added into shifts_final[..., 1]
```

| term | value here |
|---|---|
| `random_shifts` | **±100 px**, from the 12003-row table |
| `motion_shifts` | **0** — `motion_src=none`, step 7 owns the drift |
| `rhapp_shifts` | **0** — `ndist=1`, rhapp is a plane-to-plane difference (`steps15.py:732-734`) |
| step-4b axis | measured on bin-0 Paganin phase |

`rotation_center_shift=0` with `center_src=measured` is required, not a gap —
step 4b folds the measured axis into `cshifts_final` and steps 4/5/6 would add
a configured value on top; `steps15.py:79` hard-aborts on the inconsistent
pairing, and all five `config_step6_*.conf` here are `0` to match.

Sample drift is not estimated in step 3 at all. The retake-based estimator in
[`estimate_quali_motion.py`](estimate_quali_motion.py) is present and
`motion_src=quali` switches it on, but **it does not work and must be left
off**: on raw ID16A frames the correlation is pinned to zero lag out to roughly
±50 px, so it cannot measure even a *known* shift of 26–55 px. See ["What the
retake estimator actually does" in the HT_RD300
README](../AtomiumS1_HT_RD300/README.md#what-the-retake-estimator-actually-does--why-motion_srcnone).
This affects neither `rhapp` (signal at 270–390 object px, far outside that
zone) nor the step-4b axis (measured on phase, not raw frames) nor step 7
(measured on a finished volume).

## Running it

```bash
qsub polaris_run.sh           # 2 nodes, 18 h, preemptable: the full two-pass ladder
```

The ladder, the reason there are two passes, and the re-submission hazard are
in [`../AtomiumS1_FT_RD000/README.md`](../AtomiumS1_FT_RD000/README.md#the-two-pass-ladder)
— they are byte-identical here.

## Differences from the FT_RD300 configs

`config_steps15.conf` differs in **`path_out` and `pfile` only**.

| key | value |
|---|---|
| `pfile` | `Atomium_S1_FT_4K_RD100_004p5nm_0001` |
| `start_step` | `1` (convert from EDF; this scan has never been converted) |
| `ref_dist` | `0` — forced, `ndist=1` |
| `rho` | `1.0, 0.00625, 0.32, 0.0` (the FT ladder) |
| rhapp | zero by construction at `ndist=1` |
| motion | **none** — `motion_src=none`; step 7 owns the whole drift |
| axis | measured at step 4b; `rotation_center_shift=0` everywhere |
| correct3d | not a step-3 term; step 7 writes `correct_correct3D_extra.txt` and step 6 applies it |
