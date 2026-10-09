# AtomiumS1_HT_RD025 — 4-distance HT, ±25 px random displacement, 4.5 nm voxels

`Atomium_S1_HT_4K_RD025_004p5nm_0003`, ESRF ID16A visit **blc17322**, beamtime
20260825, collected 2026-08-29. On eagle at

```
/eagle/APS_IRI/vnikitin/20260829/AtomiumS1/
```

4000 projections over 180°, four propagation distances, 4096² frames at a 4.5 nm
voxel. Same sample, optic and energy as
[`../AtomiumS1_HT_RD300`](../AtomiumS1_HT_RD300) — **the only difference is the
commanded random-displacement amplitude, ±25 px here against ±300 px there.**
The point of this directory is that comparison, so everything that does not have
to differ is copied from the RD300 sibling rather than re-tuned.

**[`../AtomiumS1_HT_RD300/README.md`](../AtomiumS1_HT_RD300/README.md) is the
reference for how the pipeline works and why each shift term exists.** This
README records only what is different here, what is inherited and unverified,
and what is missing.

## What arrived

```
<pfile>_1_ .. _4_/    4069 files each: 4003 EDF + 40 ref + 21 dark + sidecars
<pfile>/projections/  <pfile>_0001..0004.{nx,txt,json}   4003-row shift tables
<pfile>_/             ht_<pfile>.m, rhapp.mat, rhappnofit.mat, shift_show
```

4003 projection frames, not 4000: the scan writes the 4000 scan frames plus
three post-scan retakes at ω=180 / 90 / 0.

There is **no master `<pfile>.nx`** for this scan, so the link-table branch in
[`esrf_layout.py`](esrf_layout.py) finds nothing and the plain `_000k` naming is
used. That is the normal case; RD300 is the exception, where planes 3 and 4 were
re-taken as scan 0006 and only the master link table records it.

All 44 NXtomo virtual sources fail to resolve — our eagle copies flatten the
ESRF `<sample>` level — so `Layout` falls back from `nxvds` to **`ewoks`**:
geometry from the NXtomo, frames from the EDFs. Expected, and
[`preflight.py`](preflight.py) prints it rather than letting it pass silently.

### Geometry, from the `.info` sidecars

| plane | SourceDistance (mm) | Distance (mm) | PixelSize (µm) | norm_mag |
|---|---|---|---|---|
| 1 | −3.69806 | 1209.28 | 0.0045000 | 1.000000 |
| 2 | −3.85670 | 1209.12 | 0.0046931 | 0.958865 |
| 3 | −4.49129 | 1208.48 | 0.0054652 | 0.823385 |
| 4 | −5.80891 | 1207.17 | 0.0070686 | 0.636619 |

Energy 17.1 keV, detector pixel 1.476015 µm, voxel 4.5000 nm, `Dim` 4096.
Identical to RD300 to five decimals, which is the point.

## What is inherited and unverified

**The rotation axis is measured, not inherited.** As configured,
`center_src=measured` and **`rotation_center_shift=0`** — step 4b measures the
axis on bin-0 Paganin phase and folds it into `cshifts_final`. The pairing is
required rather than a gap: steps 4, 5 and 6 would each add a configured value
*on top* of the measured one, so `steps15.py:79` hard-aborts if
`center_src=measured` is combined with a non-zero `rotation_center_shift`, and
all five `config_step6_*.conf` here carry `0` to match — which nothing checks
automatically.

ESRF's own number for the sibling RD300 scan is −29.48, from its nabu
configuration. **There is no nabu drop for RD025** — `<pfile>_/` holds only the
Octave pipeline's rhapp output, no `_rec_` tree — so there is nothing here even
to cross-check the measurement against, and the step-4b measurement is the only
source. Expect no `rotation axis: nabu ...` line in the step-3 log.

## What is missing

**`correct_motion.txt` does not exist for this scan**, at any plane — and it
is not wanted. **Step 3 carries no drift term at all: `motion_src=none`**, so
`shifts_final = random + rhapp` and step 7 fits the whole drift out of a
finished volume.

[`estimate_quali_motion.py`](estimate_quali_motion.py) is still here, and
`motion_src=quali` in `config_steps15.conf` switches it on — but **it is not
validated and should be left off**. Measured on HT_RD300, where four truth
points exist, it reads ≈ 0 wherever the true drift is below ~20 object px, and
four separate attempts to filter the offending zero-lag component away all
failed; see ["What the retake estimator actually does" in the HT_RD300
README](../AtomiumS1_HT_RD300/README.md#what-the-retake-estimator-actually-does--why-motion_srcnone).
`quali.mat` is present here, so `--validate` has something to check against
even though `reference_motion.mat` is not — and the warning below is why this
scan was expected to be the hard case.

> *Research log, from when the estimator was expected to ship. It does not run
> as configured, so none of these warnings will appear in the step-3 log.*
>
> **±25 px is the hard case for this estimator, and it says so.** The method
> separates the sample peak from the detector-fixed illumination peak by
> *predicting* where the illumination will correlate — at `r_retake − r_scan`,
> known from the shift table — and masking a 24 px disc there. That works when
> the commanded displacement is large. At ±25 px the predicted lag is itself
> within ~25 px of zero, so the disc would cover the sample peak too; the code
> detects this, **skips the mask, and flags the point unverifiable**. The same
> warning fires on an RD000 scan, where the illumination lag *is* zero lag.
>
> This is the amplitude at which the previous `estimate_motion.py` failed
> outright — a single ~3 px spike at exactly zero lag and no sample peak
> anywhere in ±25 px. Two guards now stand between that failure and a silent
> wrong answer: the unwhitened 1-D profile cross-check, which warns when it
> disagrees with the 2-D peak, and the cross-crop scatter limit, which **drops**
> a point whose crops disagree by more than 5 px rather than taking their
> median. A dropped point degrades the fit to degree 1; two dropped points stop
> step 3.
>
> So read the warnings in the step-3 log on this scan. The drift here is small
> (sub-px to ~4 bin-2 px in `quali.mat`), which makes it the null-ish case that
> shows the estimator does not *invent* motion — and step 7 still owns whatever
> is left over either way.

**`ref_dist = 2` does not match ESRF's `reference_plane = 1`** for this scan
(RD300 says 3). Harmless as configured — step 3 re-differences rhapp against
`ref_dist` itself, there is no `correct_motion.txt` written for their plane to
inherit an offset from, and what is left is a per-angle global shift that step 7
fits. Kept at 2 so RD025 and RD300 stay comparable.

## Running it

```bash
qsub polaris_run_debug.sh     # 10 nodes, 1 h, debug-scaling: preflight + steps15 + as much of bin 2 as fits
qsub polaris_run.sh           # 2 nodes, 18 h, preemptable: the full two-pass ladder
```

[`preflight.py`](preflight.py) runs first in both and fails the job on a
half-copied scan — `Layout` infers `ndist` by globbing `{pfile}_[0-9]_/`, so a
scan caught mid-copy would otherwise reconstruct at the wrong number of planes
with no error anywhere. It also checks frame counts, `.info` agreement across
planes, and the geometry cross-check, and warns loudly about the missing
`correct_motion.txt`.

### The two-pass ladder

```
PASS 1   steps15
         step6  config_step6_bin2_nopos.conf   iter    0 -> 1024   lam_laplacian 5e-5
         step6  config_step6_bin1_nopos.conf   iter 1024 -> 1280   lam_laplacian 1.25e-5
STEP 7   step7  config_step6_bin1_nopos.conf   --iter 1280 --bin 1
         -> correct_correct3D_extra.txt, next to the configs
PASS 2   step6  config_step6_bin2.conf         iter    0 -> 1024   <- applies the correction
         step6  config_step6_bin1.conf         iter 1024 -> 1280
         step6  config_step6_bin0.conf         iter 1280 -> 1536   lam_laplacian 0
```

`_nopos` is pass 1's own set of configs, differing from pass 2's in exactly
three keys: the third `rho` component — the position direction — is **0, so
positions are frozen**; `correct3d_extra=0`; and `path_out` is
`*_rec6_paper_nopos` rather than `*_rec6_paper`. Pass 1 therefore produces a
volume whose drift has *not* been refined away, which is what step 7 needs in
order to measure it; the two trees do not collide.

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
out pass 1 and step 7 and resubmit pass 2 alone.

## Differences from the RD300 configs

Only the identity keys and the amplitude-dependent prose. Everything else —
`ntheta`, `ndist`, `rhapp_bin`, `correct3d_bin`, the `nz/n/nobj` ladder,
`niter`/`start_iter`, `nchunk`, `mask*`, `psf_sigma`, `rho`, `lam_*` — is
copied verbatim, because the geometry and the BH ladder are identical.

| key | value |
|---|---|
| `pfile` | `Atomium_S1_HT_4K_RD025_004p5nm_0003` |
| `start_step` | `1` (convert from EDF; this scan has never been converted) |
| rhapp | measured by `estimate_rhapp.py`; `rhapp.mat` is a `--validate` target, not an input |
| motion | **none** — `motion_src=none`; step 7 owns the whole drift |
| correct3d | not a step-3 term; step 7 writes `correct_correct3D_extra.txt` and step 6 applies it |
