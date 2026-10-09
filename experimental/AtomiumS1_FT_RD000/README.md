# AtomiumS1_FT_RD000 — single-distance FT, no random displacement, 4.5 nm voxels

`Atomium_S1_FT_4K_RD000_004p5nm_0002`, ESRF ID16A visit **blc17322**, beamtime
20260825, collected 2026-08-29. On eagle at

```
/eagle/APS_IRI/vnikitin/20260829/AtomiumS1/
```

12000 projections over 180°, **one** propagation distance, 4096² frames at a
4.5 nm voxel. Same sample, optic and energy as every other AtomiumS1 scan —
**the only difference is the commanded random-displacement amplitude, which
here is zero.** This is the baseline of the RD sweep: RD000 / RD025 / RD100 /
RD300 are ±0, ±25, ±100 and ±300 detector px.

**[`../AtomiumS1_FT_RD300/README.md`](../AtomiumS1_FT_RD300/README.md) is the
reference for how the pipeline works and why each shift term exists.** This
README records only what is different here.

## What arrived

```
<pfile>_1_/           12064 files: 12003 EDF + 40 ref + 21 dark + sidecars
<pfile>/projections/  <pfile>_0001.{nx,json}      -- and NO .txt shift table
```

12003 projection frames, not 12000: the scan writes the 12000 scan frames plus
three post-scan retakes at ω = 180 / 90 / 0.

There is no master `<pfile>.nx`, so the link-table branch in
[`esrf_layout.py`](esrf_layout.py) finds nothing and the plain `_000k` naming is
used. All 124 NXtomo virtual sources fail to resolve — our eagle copies flatten
the ESRF `<sample>` level — so `Layout` falls back from `nxvds` to **`ewoks`**:
geometry from the NXtomo, frames from the EDFs. Both are expected, and
[`preflight.py`](preflight.py) prints them rather than letting them pass
silently. Frames were spot-checked non-zero with
[`../AtomiumS1_HT_RD300/check_frames_nonzero.py`](../AtomiumS1_HT_RD300/check_frames_nonzero.py);
`preflight.py` deliberately reads none.

**There is no shift table**, and that is correct rather than a missing file:
zero displacement was commanded, so the piezo never moved and ESRF wrote
nothing. `preflight.py` recognises the RD000 case by name and says so:

```
WARNING: plane 1: no shift table at .../..._0001.txt
         -- RD000, so zero displacement was commanded and step 3 will use zeros
```

### Geometry, from the `.info` sidecar

| plane | SourceDistance (mm) | Distance (mm) | PixelSize (µm) | norm_mag |
|---|---|---|---|---|
| 1 | −3.69806 | 1209.28 | 0.0045000 | 1.000000 |

Energy 17.1 keV, detector pixel 1.476015 µm, voxel 4.5000 nm, `Dim` 4096,
`TOMO_N` 12000, `ScanRange` −180.

## There is no ESRF processing for this scan

Only three of the seven AtomiumS1 scans have an ESRF processing drop — the
`<pfile>_` directory with no trailing digit. They are FT_RD300, HT_RD025 and
HT_RD300. **This scan has none**: no `rhapp.mat`, no `quali.mat`, no
`reference_motion.mat`, no `naburec/`, no `correct_motion.txt`.

So nothing here is inherited from ESRF and nothing can be validated against
them. Every number this pipeline uses it measures itself. That is the point of
the sweep — these are the scans the self-contained estimators exist for — but
it also means a mistake has no external witness, so read the step-3 and step-4b
log lines rather than assuming.

## The shift model, as configured

```python
shifts_final = random_shifts + motion_shifts + rhapp_shifts
# + the step-4b rotation axis, added into shifts_final[..., 1]
```

**On this scan three of those four terms are identically zero**, which makes it
the simplest configuration in the sweep:

| term | value here | why |
|---|---|---|
| `random_shifts` | **0** | no displacement commanded, no shift table |
| `motion_shifts` | **0** | `motion_src=none` — see below |
| `rhapp_shifts` | **0** | `ndist=1`, and rhapp is a *plane-to-plane* difference, so there is nothing to difference (`steps15.py:732-734` logs `ndist=1, rhapp is zero by construction`) |
| step-4b axis | measured | the only non-zero term |

So `shifts.png` is flat apart from the axis, and that is the correct result —
not a sign the figure failed.

**`motion_src=none`.** Sample drift is *not* estimated in step 3; step 7 fits
it out of a finished volume instead and writes `correct_correct3D_extra.txt`,
which **step 6** applies. The retake-based estimator in
[`estimate_quali_motion.py`](estimate_quali_motion.py) is still present and
`motion_src=quali` switches it on, but **it does not work and must be left
off** — on raw ID16A frames the correlation is pinned to zero lag out to
roughly ±50 px, so it cannot measure even a *known* shift. See ["What the
retake estimator actually does" in the HT_RD300
README](../AtomiumS1_HT_RD300/README.md#what-the-retake-estimator-actually-does--why-motion_srcnone).
On this scan it could not be validated anyway: there is no `quali.mat`.

**`rotation_center_shift=0` with `center_src=measured`** — this pairing is
required, not a gap. Step 4b measures the axis on bin-0 Paganin phase and folds
it into `cshifts_final`; steps 4, 5 and 6 would then add any configured value
*on top*. `steps15.py:79` hard-aborts on the inconsistent combination, and all
five `config_step6_*.conf` here carry `rotation_center_shift=0` to match, which
nothing checks automatically.

## Running it

```bash
qsub polaris_run.sh           # 2 nodes, 18 h, preemptable: the full two-pass ladder
```

[`preflight.py`](preflight.py) runs first and fails the job on a half-copied
scan — `Layout` infers `ndist` by globbing `{pfile}_[0-9]_/`, so a scan caught
mid-copy would otherwise reconstruct at the wrong number of planes with no
error anywhere. It also checks frame counts, `.info` agreement and the geometry
cross-check.

### The two-pass ladder

```
PASS 1   steps15
         step6  config_step6_bin2_nopos.conf   iter    0 -> 1024   lam_laplacian 5e-5
         step6  config_step6_bin1_nopos.conf   iter 1024 -> 1280   lam_laplacian 2.5e-5
STEP 7   step7  config_step6_bin1_nopos.conf   --iter 1280 --bin 1
         -> correct_correct3D_extra.txt, next to the configs
PASS 2   step6  config_step6_bin2.conf         iter    0 -> 1024   <- applies the correction
         step6  config_step6_bin1.conf         iter 1024 -> 1280
         step6  config_step6_bin0.conf         iter 1280 -> 1536   lam_laplacian 0
```

`_nopos` is pass 1's own set of configs, and it differs from pass 2's in
exactly three keys: `rho=1.0,0.00625,0,0.0` (the third component is the
position direction, so **positions are frozen**), `correct3d_extra=0`, and a
`path_out` of `*_rec6_paper_nopos` rather than `*_rec6_paper`. Pass 1 therefore
produces a volume whose drift has *not* been refined away, which is what step 7
needs in order to measure that drift; the two trees do not collide.

Two passes because step 7's per-angle drift is only read at **bin 2** — the one
rung with `start_iter=0`, where `Reader.read_pos` adds it to `cshifts_final`.
bin 1 and bin 0 inherit it through the checkpoint; re-adding would double count.
The same three configs serve both passes: `find_latest_checkpoint` returns
`None` whenever `start_iter=0`, so pass 2's bin 2 ignores what pass 1 left and
re-seeds from the step-5 Paganin+FBP volume. During pass 1 the correction file
does not exist yet, so bin 2 logs "positions unchanged" and skips it.

### Re-submitting a finished run silently loses work

On a second submission `correct_correct3D_extra.txt` already exists, so pass 1
applies it; step 7 then re-measures from an already-corrected volume and
**overwrites** the file with the residual. Step 6 *adds* the file to
`cshifts_final` rather than accumulating, so the original correction is lost,
not doubled — the second run is worse than the first and nothing errors.

`polaris_run.sh` refuses to start pass 1 when the file is present. After a
mid-pass-2 preemption, moving the file aside is the **wrong** answer: comment
out pass 1 and step 7 and resubmit pass 2 alone. Resume itself is manual —
`find_latest_checkpoint` globs `checkpoint_*{start_iter:04}.h5`, so set
`start_iter` to the highest checkpoint on disk by hand.

## Differences from the FT_RD300 configs

`config_steps15.conf` differs in **`path_out` and `pfile` only**. Everything
else — `ntheta`, `ndist`, `paganin=35`, `ref_dist=0`, the `nz/n/nobj` ladder,
`niter`/`start_iter`, `nchunk`, `mask*`, `psf_sigma`, `rho`, `lam_*` — is
copied verbatim, because the geometry and the BH ladder are identical and the
whole point is that only the displacement amplitude varies.

| key | value |
|---|---|
| `pfile` | `Atomium_S1_FT_4K_RD000_004p5nm_0002` |
| `start_step` | `1` (convert from EDF; this scan has never been converted) |
| `ref_dist` | `0` — forced, `ndist=1` |
| `rho` | `1.0, 0.00625, 0.32, 0.0` (the FT ladder; HT uses different values) |
| rhapp | zero by construction at `ndist=1` |
| motion | **none** — `motion_src=none`; step 7 owns the whole drift |
| axis | measured at step 4b; `rotation_center_shift=0` everywhere |
| correct3d | not a step-3 term; step 7 writes `correct_correct3D_extra.txt` and step 6 applies it |
