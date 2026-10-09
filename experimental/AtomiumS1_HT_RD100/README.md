# AtomiumS1_HT_RD100 — 4-distance HT, ±100 px random displacement, 4.5 nm voxels

`Atomium_S1_HT_4K_RD100_004p5nm_0002`, ESRF ID16A visit **blc17322**, beamtime
20260825, collected 2026-08-29. On eagle at

```
/eagle/APS_IRI/vnikitin/20260829/AtomiumS1/
```

4000 projections over 180°, **four** propagation distances, 4096² frames at a
4.5 nm voxel. Same sample, optic and energy as
[`../AtomiumS1_HT_RD300`](../AtomiumS1_HT_RD300) and
[`../AtomiumS1_HT_RD025`](../AtomiumS1_HT_RD025) — **the only difference is the
commanded random-displacement amplitude**, ±100 px here against ±25 and ±300 in
the siblings. The point of this directory is that comparison, so everything
that does not have to differ is copied from the RD300 sibling rather than
re-tuned.

**[`../AtomiumS1_HT_RD300/README.md`](../AtomiumS1_HT_RD300/README.md) is the
reference for how the pipeline works and why each shift term exists.** This
README records only what is different here.

## What arrived

```
<pfile>_1_ .. _4_/    4069 files each: 4003 EDF + 40 ref + 21 dark + sidecars
<pfile>/projections/  <pfile>_0001..0004.{nx,txt,json}   4003-row shift tables
```

4003 projection frames, not 4000: the scan writes the 4000 scan frames plus
three post-scan retakes at ω = 180 / 90 / 0.

**All four shift tables have the full 4003 rows**, which is worth stating
because HT_RD300 does not — its tables stop short at `ntheta` and step 3 pads
them with zeros. Nothing is padded here. The row count is checked per plane,
since the four planes are four separate scans and need not agree.

Measured amplitude in the plane-1 table, detector px:

| | min | max | ptp |
|---|---|---|---|
| x | −99.94 | +99.98 | 199.93 |
| y | −99.85 | +99.97 | 199.82 |

so "RD100" is the half-amplitude, as the name implies.

There is no master `<pfile>.nx` for this scan, so the link-table branch in
[`esrf_layout.py`](esrf_layout.py) finds nothing and the plain `_000k` naming is
used. That is the normal case; RD300 is the exception, where planes 3 and 4
were re-taken as scan 0006 and only the master link table records it.

All 44 NXtomo virtual sources fail to resolve — our eagle copies flatten the
ESRF `<sample>` level — so `Layout` falls back from `nxvds` to **`ewoks`**:
geometry from the NXtomo, frames from the EDFs. Expected, and
[`preflight.py`](preflight.py) prints it rather than letting it pass silently.
Frames on all four planes were spot-checked non-zero with
[`../AtomiumS1_HT_RD300/check_frames_nonzero.py`](../AtomiumS1_HT_RD300/check_frames_nonzero.py);
`preflight.py` deliberately reads none.

### Geometry, from the `.info` sidecars

| plane | SourceDistance (mm) | Distance (mm) | PixelSize (µm) | norm_mag |
|---|---|---|---|---|
| 1 | −3.69813 | 1209.30 | 0.0045000 | 1.000000 |
| 2 | −3.85678 | 1209.14 | 0.0046931 | 0.958865 |
| 3 | −4.49138 | 1208.51 | 0.0054653 | 0.823384 |
| 4 | −5.80903 | 1207.19 | 0.0070686 | 0.636618 |

Energy 17.1 keV, detector pixel 1.476015 µm, voxel 4.5000 nm, `Dim` 4096,
`TOMO_N` 4000, `ScanRange` −180. Matches RD025 and RD300 to five decimals,
which is the point.

## There is no ESRF processing for this scan

Only three of the seven AtomiumS1 scans have an ESRF processing drop — the
`<pfile>_` directory with no trailing digit. They are FT_RD300, HT_RD025 and
HT_RD300. **This scan has none**: no `rhapp.mat`, no `quali.mat`, no
`reference_motion.mat`, no `naburec/`, no `correct_motion.txt`.

That matters more on HT than on FT, because HT has a real `rhapp` term:

* **`rhapp` is measured, not validated.** [`estimate_rhapp.py`](estimate_rhapp.py)
  measures it from the data as usual, but there is no `rhapp.mat` to run
  `--validate` against. The estimator itself is validated on the scans that do
  have one (corr 0.875–0.998 against `rhapp.mat`, at parity with ESRF's own
  per-angle scatter), so the method is trusted — this scan simply has no
  independent witness.
* **The rotation axis has no ESRF number either.** There is no `naburec/*.conf`
  and no PyHST `.par`, so `esrf_meta.nabu_axis` finds nothing and the
  cross-check log line that fires on RD025/RD300 will be absent. Step 4b
  measures the axis regardless; here it is the only source.

Read the step-3 and step-4b log lines rather than assuming.

## The shift model, as configured

```python
shifts_final = random_shifts + motion_shifts + rhapp_shifts
# + the step-4b rotation axis, added into shifts_final[..., 1]
```

| term | value here |
|---|---|
| `random_shifts` | **±100 px**, from the four 4003-row tables |
| `motion_shifts` | **0** — `motion_src=none`, step 7 owns the drift |
| `rhapp_shifts` | **measured** per plane by `estimate_rhapp.py`; zero at `ref_dist` by construction |
| step-4b axis | measured on bin-0 Paganin phase |

**`motion_src=none`.** Sample drift is *not* estimated in step 3; step 7 fits
it out of a finished volume and writes `correct_correct3D_extra.txt`, which
**step 6** applies. The retake-based estimator in
[`estimate_quali_motion.py`](estimate_quali_motion.py) is present and
`motion_src=quali` switches it on, but **it does not work and must be left
off** — on raw ID16A frames the correlation is pinned to zero lag out to
roughly ±50 px, so it cannot measure even a *known* shift of 26–55 px. See
["What the retake estimator actually does" in the HT_RD300
README](../AtomiumS1_HT_RD300/README.md#what-the-retake-estimator-actually-does--why-motion_srcnone).
On this scan it could not be validated anyway: there is no `quali.mat`. Note
this is a statement about *raw-frame* correlation only — `rhapp` sits at
270–390 object px, far outside the contaminated zone, and the step-4b axis and
step 7 work on phase and on a finished volume respectively.

**`rotation_center_shift=0` with `center_src=measured`** — this pairing is
required, not a gap. Step 4b measures the axis on bin-0 Paganin phase and folds
it into `cshifts_final`; steps 4, 5 and 6 would then add any configured value
*on top*. `steps15.py:79` hard-aborts on the inconsistent combination, and all
five `config_step6_*.conf` here carry `rotation_center_shift=0` to match, which
nothing checks automatically.

**`ref_dist = 2`** (0-based), matching the RD025 and RD300 siblings so the three
amplitudes stay comparable. ESRF's `reference_plane` is not available for this
scan, and it would not matter: step 3 re-differences rhapp against `ref_dist`
itself, and what is left is a per-angle global shift that step 7 fits.

## Running it

```bash
qsub polaris_run.sh           # 2 nodes, 18 h, preemptable: the full two-pass ladder
```

[`preflight.py`](preflight.py) runs first and fails the job on a half-copied
scan — `Layout` infers `ndist` by globbing `{pfile}_[0-9]_/`, so a scan caught
mid-copy would otherwise reconstruct at the wrong number of planes with no
error anywhere. It also checks frame counts, `.info` agreement across planes,
the geometry cross-check, and the shift-table row counts.

### The two-pass ladder

```
PASS 1   steps15
         step6  config_step6_bin2_nopos.conf   iter    0 -> 1024   lam_laplacian 5e-5
         step6  config_step6_bin1_nopos.conf   iter 1024 -> 1280   lam_laplacian 1.25e-5
STEP 7   step7  config_step6_bin1_nopos.conf   --iter 1280 --bin 1
         -> correct_correct3D_extra.txt, next to the configs
PASS 2   step6  config_step6_bin2.conf         iter    0 -> 1024   <- applies the correction
                                                                   rho 1.0,0.05,0.04,0
         step6  config_step6_bin1.conf         iter 1024 -> 1280   rho 1.0,0.05,0.02,0
         step6  config_step6_bin0.conf         iter 1280 -> 1536   rho 1.0,0.05,0.01,0
                                                                   lam_laplacian 0
```

`_nopos` is pass 1's own set of configs, differing from pass 2's in exactly
three keys: the third `rho` component — the position direction — is **0, so
positions are frozen**; `correct3d_extra=0`; and `path_out` is
`*_rec6_paper_nopos` rather than `*_rec6_paper`. Pass 1 therefore produces a
volume whose drift has *not* been refined away, which is what step 7 needs in
order to measure that drift; the two trees do not collide.

Two passes because step 7's per-angle drift is only read at **bin 2** — the one
rung with `start_iter=0`, where `Reader.read_pos` adds it to `cshifts_final`.
bin 1 and bin 0 inherit it through the checkpoint; re-adding would double count.
The same three configs serve both passes: `find_latest_checkpoint` returns
`None` whenever `start_iter=0`, so pass 2's bin 2 ignores what pass 1 left and
re-seeds from the step-5 Paganin+FBP volume. During pass 1 the correction file
does not exist yet, so bin 2 logs "positions unchanged" and skips it — the
sequencing is self-enforcing on a first run.

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

## Differences from the RD300 configs

Only the identity keys and `start_step`. Everything else — `ntheta`, `ndist`,
`rhapp_bin`, `correct3d_bin`, the `nz/n/nobj` ladder, `niter`/`start_iter`,
`nchunk`, `mask*`, `psf_sigma`, `rho`, `lam_*`, `ref_dist`, `paganin` — is
copied verbatim, because the geometry and the BH ladder are identical.

| key | value |
|---|---|
| `pfile` | `Atomium_S1_HT_4K_RD100_004p5nm_0002` |
| `start_step` | `1` (convert from EDF; this scan has never been converted — RD300 is at `3` because it already has its converted h5) |
| `ref_dist` | `2` (0-based), matching the siblings |
| `paganin` | `35` |
| rhapp | measured by `estimate_rhapp.py`; no `rhapp.mat` here, so no `--validate` target |
| motion | **none** — `motion_src=none`; step 7 owns the whole drift |
| axis | measured at step 4b; `rotation_center_shift=0` everywhere, and no ESRF number to cross-check against |
| correct3d | not a step-3 term; step 7 writes `correct_correct3D_extra.txt` and step 6 applies it |
