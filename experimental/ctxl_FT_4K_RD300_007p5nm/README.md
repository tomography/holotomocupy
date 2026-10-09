# ctxl — cortex tissue, SINGLE-distance FT, ±300 px random displacement, 7.5 nm voxels

ESRF ID16A, proposal **ihls3888**, scan taken **2026-08-30 10:53 → 13:26**
(2 h 33 min, from the mtimes of `scan0008/balor_*.h5`).
Cortex tissue, **one** propagation distance, 12000 projections over 180°,
±300 px random sample displacement, 4096² detector at 17.1 keV.

Raw data: `/eagle/APS_IRI/vnikitin/20260829/ctxl/ctxl_FT_4K_RD300_007p5nm_0003`
plus `/eagle/APS_IRI/vnikitin/20260829/RAW_DATA/ctxl/ctxl_FT_4K_RD300_007p5nm_0003`

Same sample, same plane and the same optic as the 4-distance scan in
[`../ctxl_HT_4K_RD300_007p5nm`](../ctxl_HT_4K_RD300_007p5nm) — that folder is the
reference for everything the two share, and this one only writes down what
differs. It has never been reconstructed; every number below was measured from
the raw data on 2026-09-01 or carried over from a named sibling and marked so.

> **The pixels are not where the HT folder's are.** This scan was never
> converted to EDF and Peter's octave pipeline was never run on it. It is the
> *third* directory flavour, `nxvds`: geometry in an NXtomo, pixels only in the
> raw balor HDF5 behind a virtual dataset. All of that is confined to
> [`esrf_layout.py`](esrf_layout.py); nothing in the raw tree has to be copied
> or renamed. See **Where the pixels are** below.

> ## !! SUPERSEDED IN PART BY THE 2026-09-27 ESRF DROP
>
> ESRF has since converted this scan to EDF and run Peter's octave pipeline on
> it.  The drop landed on eagle on **2026-09-27** and carries, for the first
> time for this scan: `correct_motion.txt`, `correct_correct3D.txt` (+ the
> vertical-only `_v` variant, whose zeroed column is the HORIZONTAL one --
> file col0, as on AtomiumS1), `quali.mat`, `reference_motion.mat`,
> `shrink_list.mat`, `shapp.mat`, a PyHST `.par` and a full `naburec/`.
> The copy finished 2026-09-27 18:02; everything below is measured against it.
> **The config files have been updated; this README has NOT been
> rewritten.**  What it still says that is now wrong:
>
> | README says | now |
> |---|---|
> | `rotation_center_shift` **+9.80** (retakes) | **+15.47**, from ESRF's own `naburec/nabu_final_cm.conf` (`rotation_axis_position = 1031.235375` on the 2048-wide `_rec_.nx`, `(1031.235375 - 1023.5) x 2`).  The retake estimate +9.82 +- 0.41 stands as an independent measurement 5.7 px away; ESRF's own number wins, as it did on AtomiumS1_HT.  The `.par`'s 1023.931433 is stale, again exactly as on AtomiumS1_HT; on the 1-based PyHST centre 1024.5 it gives **-1.14**, 16.6 px away. |
> | `correct_motion.txt` "estimated here" | ESRF's own file, in RAW detector px (amplitude test slope 1.00005 / 1.00001 against the random `.txt`), so `correct_motion_bin` stays unset.  Validated against `reference_motion.mat`'s `ref_v`/`ref_h` to 5e-7. |
> | `correct3D` "borrowed from the AtomiumS1 FT scan" | ESRF's own file.  `correct3d_bin` stays **2**.  The retired transplant was good to ~1 px rms (corr 0.984 / 0.956, max abs diff 2.42 px). |
> | shrinkage "none - measured as absent" | `shrink_list.mat` EXISTS (row 0: h 4356 ppm, v 6221 ppm) and **Peter applied it** -- `<pfile>_rec_.info` has `correct_shrink = 2`.  So `rho[tp]=0` freezing tp at ESRF's value is what reproduces his reconstruction, not a bug.  Our own retake bound (2 sigma <= ~530 ppm) still says those numbers are grid noise from `find_shrink.m`'s 5000 ppm quantum -- that is a disagreement with ESRF, not a reason to drop the file.  `shapp.mat` is the same shrinkage as a per-angle ramp saturating at exactly those two values; it is non-linear (a straight-line fit leaves 9.4% / 6.8% of the range as residual) where `init_tp_from_shrink()` ramps linearly, so we match his endpoints but not his path. |
> | "never converted to EDF, Peter's pipeline never run" | both are now false; the `nxvds` layout still describes the RAW tree. |
>
> **Re-copied unnested and verified complete on 2026-09-27 18:02**: 12066 EDFs,
> `quali.mat`, `shrink_list.mat`, a populated `naburec/`, and
> `<pfile>/projections/` with the NXtomo and the +-300 px random `.txt`.
> Everything steps 1-5 and step 0 need is on disk.  See
> [Matching Peter's final reconstruction](#matching-peters-final-reconstruction).

| | ctxl FT (here) | `../ctxl_HT_4K_RD300_007p5nm` | `../Y350a_largedisp_006nm` |
|---|---|---|---|
| layout | **nxvds (2026, no EDF)** | ewoks (2026, EDF) | bliss (2025) |
| distances | **1** | 4 | 1 |
| projections | **12000** | 4000 | 3000 |
| energy | 17.1 keV | 17.1 keV | 33.35 keV |
| voxel size | 7.500 nm | 7.500 nm | 6.000 nm |
| random displacement | ±300 px | ±300 px | ±300 px |
| `nobj` | **4800** | 5056 | 4800 |
| `rotation_center_shift` | **+15.47 px** (ESRF nabu; was +9.80) | −15.77 px (nabu) | −37.50 px |
| `paganin` | **60** | 60 | 40 |
| `rhapp` | **N/A — one plane** | yes | N/A |
| `correct_motion.txt` | **from ESRF** (2026-09-27; was estimated here) | yes, from ESRF | estimated locally |
| shrinkage | *measured* as absent; ESRF's `shrink_list.mat` disagrees | none — measured as absent | none — measured as absent |
| probe | **measurable, `step0.py` works** | not possible (no RAW_DATA) | not wired up |

## `holotomocupy.esrf_meta` -- written 2026-09-27

`steps15.py:43` is `from holotomocupy import esrf_meta`.  The module had never
existed; it now does, at `src/holotomocupy/esrf_meta.py`.  It reads back what
ESRF's own pipeline recorded about a scan and never raises on a missing or
malformed file -- every function returns `None` or a note instead.

| function | called from | returns |
|---|---|---|
| `drop_dirs(path, pfile)` | -- | `<pfile>_`, `<pfile>_1_`, plus one nested level |
| `find_drop_file(path, pfile, name)` | :657 | `(abspath or None, note)` |
| `bin_from_pixelsize(px, vox, tol=0.02)` | :604, :610, :678, :685 | `(bin or None, why)` |
| `rec_info(path, pfile)` | :543 | `<pfile>_rec_.info` as a dict, incl. `pixelsize_m`, `width` |
| `mat_pixelsize(matpath)` | :542 | `pixelsize` out of an Octave ASCII `.mat`, in m |
| `reference_motion(path, pfile)` | :544 | `ref_v` / `ref_h` / `reference_plane_1based` / `ref_dist` |
| `nabu_axis(path, pfile, voxelsize)` | :206 | `{'rcs','source','note','candidates'}`, raw px |
| `pyhst_axis(path, pfile, voxelsize)` | :207 | same shape, from `<pfile>_rec_.par` |

Only **`find_drop_file` is load-bearing** -- without it `correct3d_shifts`
silently becomes zeros.  The other seven are diagnostics, because
`rotation_center_shift`, `rhapp_bin`, `correct3d_bin` and `ref_dist` are all
pinned in the config.  They are worth having anyway: they are what caught the
stale `.par` here and settled the AtomiumS1_HT axis.

**Centre convention, settled by validation.**  nabu's centre is `(N-1)/2`
(0-based), PyHST's is `(N+1)/2` (1-based); `rotation_center_shift =
(axis_pos - centre) x bin_factor`, used as printed, never negated.  Validated
against four independent ESRF truths, all reproduced exactly:

| scan | nabu | PyHST `.par` |
|---|---|---|
| ctxl_FT (this one) | **+15.47** (7 confs span 0.114) | -1.14, stale |
| ctxl_HT | -15.77 on the pi/2-padded 3216-wide grid | -18.87, recorded |
| AtomiumS1 FT | +10.74 | -- |
| AtomiumS1_HT | -29.48 | +44.20, 73.7 px stale |

`load_octave_text_mat` and `load_shrink_from_mats` already existed in
`src/holotomocupy/reader.py` and do the `.mat` parsing; `esrf_meta` is the
drop-layout knowledge on top.

## `esrf_layout.py` -- written 2026-09-27, third flavour `nxvds`

`steps15.py:51` (`Layout`) and `step0.py:44` (`NxFrames`) had no file behind
them; both now come from this folder's own `esrf_layout.py`.  It is
[../ctxl_HT_4K_RD300_007p5nm/esrf_layout.py](../ctxl_HT_4K_RD300_007p5nm/esrf_layout.py)
verbatim -- the `read_info` / `read_nx_geometry` helpers diff clean against it
-- plus a third flavour and the frame readers `steps15.py` step 1 needs.

**This scan has BOTH.**  `<pfile>_1_/` holds 12003 projection EDFs, 40 flats
and 20 darks, *and* `<pfile>/projections/<pfile>_0001.nx` is an NXtomo whose
`instrument/detector/data` is a virtual dataset over **124 raw balor HDF5
files** under `RAW_DATA/`.  When a virtual NXtomo is present the flavour is
`nxvds` and the readers go through it; a scan with the NXtomo but no VDS stays
`ewoks` and reads EDFs, so the same file still drives ctxl_HT unchanged.

| added | why |
|---|---|
| `NxFrames` | rebases the 124 virtual sources onto our tree; `../AtomiumS1_FT_RD300/nx_frames.py` is a standalone copy |
| `_nx_is_virtual` | picks `nxvds` vs `ewoks`; False on an unreadable `.nx`, so a half-copied drop degrades to EDF |
| `frame_shape(k)` | `steps15.py:175`, without reading a frame |
| `read_proj` / `read_refs` / `read_darks` | `steps15.py:297-317`; VDS for `nxvds`, fabio otherwise |
| `nx(k)` / `_flat_blocks(k)` | cached `NxFrames`; flats split into start/end batches by contiguous `image_key == 1` runs |
| `angles(k)` | `sample/rotation_angle` at `image_key == 0` for `nxvds`, `angles_file.txt` otherwise |

**Verified against the real drop, 2026-09-27.**  `flavour=nxvds`, ndist 1,
ntheta 12000, nref 20, ndark 20, frame 4096x4096, voxelsize 7.5 nm,
`info_check` clean.  All 124 virtual sources resolve -- **0 missing** -- and
the VDS is **bit-for-bit identical to the EDFs** at projections 0 / 6000 /
12002, flat 0 and dark 0.  `angles` matches `angles_file.txt` to 5e-7 deg and
is monotonic over the 12000 scan angles; the trailing 180 / 90 / 0 are the
post-scan retakes.

`nx(k)` **raises** when any virtual source is unresolved rather than letting
HDF5 serve its fill value: an unrebased source reads as zeros with no error
and no warning, and every downstream step would run to completion on them.

## State on eagle, and what has to be recomputed

`<path_out>/` was last written **2026-09-08**, before this drop:

| dataset | there? | still valid? |
|---|---|---|
| `pdata0`, `pdata0_0/_1/_2` | yes | **yes** -- at `ndist=1` step 4 only bins `pdata0`; the shifts enter `srdata` (a local, never written) and the padding, not the output |
| `/exchange/cshifts_final` | yes | **no** -- pre-drop `correct_motion.txt` / `correct_correct3D.txt` |
| `/exchange/shrink` | yes, identically **0** | **no** -- `shrink_list.mat` now exists |
| `_obj.h5:/exchange/obj_init_re60_2`, `obj_init_imag60_2` | yes | **no** -- FBP'd with the old shifts and the old axis.  bin 1 / bin 0 inits were never written, which is fine: the ladder upsamples from the bin-2 checkpoint |
| `_proj.h5:/exchange/proj_bin0`, `proj_bin2` | yes | diagnostic only |
| `_srdata.h5:/exchange/srdata_bin2` | yes | **no** |

So `start_step` moves 5 -> **3**.  The axis is the one exception that would
NOT have needed it: `reader.read_pos` adds `rotation_center_shift` when step 6
runs (`reader.py:276`), so +15.47 reaches step 6 from the config directly.
There is no `stop_step`, so step 3 drags step 4 along -- ~2.1 TB of I/O that
changes nothing at `ndist=1`.

## Matching Peter's final reconstruction

Peter's FINAL rec is `naburec/nabu_final_cm.conf` and its `_even` / `_odd`
twins.  They drove the only three full-recon slurm jobs that ever ran on this
scan (36740674 / 5 / 6), producing `<pfile>_rec_cm_.vol`, `_rec_cm_even_.vol`
and `_rec_cm_odd_.vol`.  `nabu_final.conf` points at a `.../correct.txt` that
does not exist anywhere in the drop and never ran -- ignore it.

The corrections are split across two stages, which is the thing to get right:

**Stage 1, Peter's octave pipeline** (`ht_ctxl_FT_..._0003.m` -> the `_rec_`
EDFs -> `nxtomomill edf2nx` -> `<pfile>_rec_.nx`, 12001 x 2048 x 2048 float32,
15 nm).  This is where the big shifts go.  `<pfile>_rec_.nx`'s
`sample/x_translation`, `y_translation` and `z_translation` are **all exactly
zero**, and nabu's only per-angle shift input is the +-21 px
`correct_correct3D.txt` -- so the +-300 px random walk and the drift are
already baked into the `_rec_` frames, via `correct_motion.txt`.  Same stage
applies the shrinkage (`correct_shrink = 2`, i.e. `find_shrink.m` trials ->
`shapp.mat`) and Paganin at `delta_beta = 60`, and bins 2x2.

**Stage 2, nabu FBP** on that `.nx`: `binning = 1`, `binning_z = 1`,
`rotation_axis_position = 1031.235375`, `axis_correction_file =` (empty),
`translation_movements_file = ../correct_correct3D.txt` -- the FULL file, both
columns, **not** the `_v` variant.  The even/odd twins differ only by
`projections_subsampling = 2:0` / `2:1`.

Term by term, against what this folder does:

| term | Peter | here | match |
|---|---|---|---|
| +-300 px random walk | inside `correct_motion.txt`, applied to the EDFs | `random_shifts`, read from `projections/*.txt` | yes |
| drift | the rest of `correct_motion.txt` | `motion_shifts = correct_motion/mag - random_shifts`, so the walk is not counted twice | yes |
| `rhapp` | not used (ndist=1) | `rhapp.mat` absent -> zeros | yes |
| shrinkage | `shapp.mat`, a per-angle ramp saturating at `shrink_list.mat` row 0 (h 4356, v 6221 ppm) | `init_tp_from_shrink()` ramps LINEARLY to the same endpoints; `rho[tp]=0` holds it | endpoints yes, path approximate |
| 3-D residual | `correct_correct3D.txt`, full, on the 15 nm grid | same file, `correct3d_bin=2` | yes |
| rotation axis | 1031.235375 on 2048 px | `rotation_center_shift=+15.47` raw px | yes |
| delta/beta | 60 | `paganin=60` | yes |
| half-sets | `projections_subsampling = 2:0` / `2:1` | `ntheta=6000` + `start_theta=0` / `1` | yes |

**No config value needs changing.**  The single real difference is the
shrinkage *path*: Peter's ramp is non-linear (a straight-line fit through
`shapp.mat` leaves 9.4% vertical / 6.8% horizontal of the range as residual),
ours is linear to the same endpoint.  `Rec` has no richer tp model, and with
`rho[tp]=0` the term is frozen either way, so the residual is a fixed
sub-pixel warp, not a drift.

## Which scan

The beamline made three attempts:

| attempt | state |
|---|---|
| `_0001` | **aborted at 82.845°** — 5524 of 12000 projections, no end flats, no retakes. Under 180° means no tomogram, no opposed pair for the axis and no retakes for drift or shrinkage. `python show_geometry.py config_steps15.conf --pfile ctxl_FT_4K_RD300_007p5nm_0001` prints the ABORTED banner. |
| `_0002` | aborted immediately; no scan directories at all. |
| `_0003` | **complete** — 12000 projections 0…179.985°, frame 12000 at exactly 180.000°, post-scan retakes at 90° and 0°, 20 darks, 20 start flats, 20 end flats. **This is the one reconstructed here.** |

`pfile` in [`config_steps15.conf`](config_steps15.conf) selects the attempt and
everything downstream follows — the scripts read `ntheta`, `ndist` and the frame
count out of the scan itself.

## Where the pixels are

There is no `<pfile>_1_/` EDF directory, no `.info` sidecar, and no `<pfile>_/`
octave directory (hence no `rhapp.mat`, no `correct_motion.txt`, no `quali.mat`,
no `ht_<pfile>.m`). What exists is the NXtomo nxtomomill wrote,

```
<pfile>/projections/<pfile>_0001.nx
```

which carries the geometry, `sample/rotation_angle` and `image_key`, and whose
`instrument/detector/data` is an HDF5 **virtual** dataset pointing at

```
RAW_DATA/ctxl/<pfile>/scanNNNN/balor_XXXX.h5
```

The stored source paths are relative and assume a `PROCESSED_DATA` level our
copy of the tree does not have. **HDF5 answers an unresolvable virtual source
with the fill value — silent zeros, no error**, which is exactly how the HT
folder's probe attempt failed. `esrf_layout.NxFrames` therefore re-bases the
sources onto the real `RAW_DATA` root itself and exposes a `missing` list; every
script that reads frames checks it and refuses to run on zeros.

Frame counts and timing, from the balor mtimes:

```
scan0008  120 files  2026-08-30 10:53:19 -> 13:26:41   projections + retakes
scan0009    1 file   2026-08-30 13:28:22               end flats
scan0010    1 file   2026-08-30 13:28:32
```

## Geometry

`python show_geometry.py config_steps15.conf`:

```
ndist=1   ntheta=12000   detector 4096x4096   nref=20   ndark=20
energy = 17.1 keV        detector pixel = 1.47601 um
focus->detector = 1212.9965 mm
z1     = 6.163535 mm     mag = 196.80x    voxel = 7.500 nm
propagation distance = 6.1322 mm,  norm_magnifications = [1.0]
```

Identical to **plane 1 of the HT scan**, as it should be — same optic
(`Optic_used=1.47601`), same energy, same focal position. There is no `.info`
sidecar to cross-check the voxel size against, so `info_check` falls back to the
NXtomo's own `sample/x_pixel_size`; that is a weak check, since the same writer
produced both. The strong check is that these numbers reproduce plane 1 of the
HT scan to seven digits, and *that* scan's four planes reproduced four
independent `.info` `PixelSize` values exactly.

![scan overview](scan_overview.png)

## Shifts

At `ndist=1` step 3 reduces to almost nothing, and it is worth being explicit
about what that means.

| term | here |
|---|---|
| random displacement | `<pfile>/projections/<pfile>_0001.txt`, 12003 rows, ±300 px, zero at frames 0 / 6000 / 11999 and at the three retakes |
| `rhapp` | **not applicable** — `rhapp.mat` registers distance planes against each other and there is one plane. Step 3 logs `rhapp.mat not found, using zeros`; that warning is expected and means nothing is wrong. |
| `correct_motion.txt` | **not provided by ESRF** — measured here, checked in as [`correct_motion.txt`](correct_motion.txt) |
| `correct3D` | **borrowed from the AtomiumS1 FT scan** — ESRF produced none for this scan, so `<pfile>_0003_/correct_correct3D.txt` is Peter's `Atomium_S1_FT_4K_RD300_004p5nm_0001_` file copied row for row and scaled by 4.5/7.5 = 0.6. Same 12001-angle grid, so no interpolation. See [Borrowed `correct_correct3D.txt`](#borrowed-correct_correct3dtxt) below. |

`correct_correct3D_extra.txt` is **ours and no longer a step-3 input**: it is
written by [`step7.py`](step7.py) into this directory and added to the
positions by step 6, so refining it costs a step-6 rerun and not a steps15
one. Same columns, same 12001 rows and the same binned pixels as Peter's
file, so one `correct3d_bin=2` covers both. It exists because the `correct3D`
above is borrowed and only good to ~1 px rms. `correct3d_extra=0` in
`config_step6_*.conf` ignores the file without deleting it, for the
with/without pair.

**`correct_motion.txt` REPLACES the random displacement, it does not add to
it.** Step 3 computes

```
motion_base  = raw_motion / norm_mag[ref] - random_shifts[:, ref]
shifts_final = random + rhapp + motion_base + (correct3d + correct3d_extra)
```

With one plane (`norm_mag[0] = 1`, `ref_dist = 0`) and no rhapp this telescopes
to `shifts_final = raw_motion + correct3d + correct3d_extra`. So the file must hold the random
displacement **plus** the drift, in raw detector pixels. `estimate_motion.py`
writes exactly that; the same arrangement is used in `../Y350a_largedisp_006nm`.

**Install it before running:**

```bash
cp correct_motion.txt <path>/<pfile>/projections/
```

Without it step 3 logs `correct_motion.txt not found, using zeros`, which is
still a valid run — it just leaves the 1.8 px of vertical drift in.

### How the drift was measured

`python estimate_motion.py config_steps15.conf --template 48 --zero-x --out correct_motion.txt`,
from the two post-scan retakes, over six crops 1024…2304:

```
retake 12001 (omega=90) vs frame 6000:   dy = +0.95   dx = +0.10   (MAD 0.02 px)
retake 12002 (omega= 0) vs frame    0:   dy = -0.83   dx = +0.04   (MAD 0.02 px)
fitted drift:  y  1.809 px ptp, rms 0.472      x  ZEROED
```

![drift](motion_estimate.png)

The raw evidence, one row per retake — the frame at the start of the scan, its
counterpart at the end, their difference, and what the measured drift on its own
would produce (same colour scale), with the phase-correlation surface the number
actually comes from:

```bash
python fig_retake_diff.py config_steps15.conf
```

![retake differences](retake_diff.png)

```
 retake  frame   omega     dy      dx    +-dy   +-dx   rms A   rms B   offset  rms B-A  rms drift
  12001   6000   90.00  +0.95   +0.10   0.01   0.02   0.1384  0.1378  +0.0975   0.0596     0.0545
  12002      0    0.00  -0.83   +0.04   0.02   0.02   0.0817  0.0835  -0.0089   0.0606     0.0509
```

Two things in that table are worth reading carefully. The `offset` column is a
flat transmission difference between the two ends of the scan (+0.098 at ω=90),
which is not drift and is removed before the panels are drawn. And **the
difference does not shrink when the drift is taken out** — shifting the retake
back by the measured 0.8–1.0 px raises the pixel-wise rms by a few percent at
every smoothing scale, and the two pairs do not even agree on which sign helps.
That is not a failed measurement: `rms(B−A) ≈ 0.06` against a frame contrast of
0.08–0.14 means the two frames have largely decorrelated at pixel scale over
2 h 33 min, and the drift is a small coherent component on top of that. Phase
correlation finds it to a MAD of 0.02 px anyway, because correlation is not an
rms. Judge the measurement from the correlation surfaces in column 5, not from
the difference panels.

The vertical figure reproduces the HT scan's own retake measurement (1.572 px
ptp) on the same sample over a comparable duration — that is the cross-check.

`--zero-x`: the horizontal drift measures 0.10 px ptp, i.e. **0.75 nm** at a
7.5 nm voxel, and the horizontal column is exactly degenerate with the rotation
axis. Zeroing it keeps that degeneracy out of the file, so
`rotation_center_shift` is the same number whether `correct_motion.txt` is
installed or not — verified: column 0 of the written file is bit-for-bit column
0 of the raw displacement table. (This is the opposite of the HT scan, whose
ESRF-supplied file carries 2.1 px in that column that the retakes do *not* see —
their axis correction folded in.)

## Rotation centre — **+9.80 px**, measured from the post-scan retakes

This scan has no `naburec/` and no PyHST `.par`, so there is nothing here that
ESRF states outright, and all four configs used to carry **−15.77** — the value
[`../ctxl_HT_4K_RD300_007p5nm`](../ctxl_HT_4K_RD300_007p5nm) gets from
`naburec/nabu_final_cm.conf`, `(1599.613741 − 1607.5) × 2 = −15.7725` raw px,
borrowed by instruction under the rule "when a scan has no axis from nabu, use
its sibling's".

**Measured 2026-09-08, that is 25.6 px wrong.** `estimate_center.py` on the
post-scan retakes gives **+9.82 ± 0.41 px**, and the FT stage position is simply
different from the HT one — exactly as on AtomiumS1, where the HT scan sits at
+44.20 and its FT sibling at +10.74. This scan's +9.8 lands on AtomiumS1_FT's
+10.74, so **both** FT scans of this beamtime put the axis near +10. All four
configs now carry **+9.80**.

```
python estimate_center.py config_steps15.conf --retakes on --bands 4

  pair 12002/12000 (180.000 deg) retake band 0: dy=+0.20  shift= +9.45 px
  pair 12002/11999 (179.985 deg) retake band 0: dy=+6.89  shift= +9.45 px
  pair 12002/11999 (179.985 deg) retake band 1: dy=+4.00  shift=+10.44 px
  pair 12002/11999 (179.985 deg) retake band 2: dy=+8.08  shift= +9.93 px
  rotation_center_shift = +9.82 +- 0.41 px   (4 of 8 kept)
```

### The sign is used as printed, not negated

Checked against the two scans where ESRF states the axis outright, with the same
command:

| scan | `estimate_center --retakes on` | ESRF truth | error |
|---|---|---|---|
| ctxl_HT | −20.09 ± 1.12 | nabu −15.77 | −4.32 |
| AtomiumS1 FT | +13.34 ± 1.60 | nabu +10.74 | +2.60 |

Same sign both times, ~4 px accuracy, no consistent bias — a negation would have
put those two 30 and 24 px out. So the honest band here is about **+6 to +14**.
Neither control applies `correct_correct3D.txt` either, and its horizontal
column averages the same +1.64 raw px on ctxl_HT as it does here, so that term
is already inside those errors rather than unaccounted for.

### Corroboration, and what does not work

On the unbinned Paganin frames in `<pfile>_proj.h5:/exchange/proj_bin0` (1200
stitched frames, every 10th angle, 4800 × 4800), an opposed-pair mirror over 10
pairs with unwhitened 1-D column-profile matching gives **+7.3 ± 3.7**, and the
correlation-free centre-of-mass landmark **−0.9**. Soft — the NCC peak is a
32 px plateau on data this smooth — but consistent with +9.8. Note the opposed
partner there is `(j, 1199−j)`: the saved stack spans 0…179.85°, so `(i, i+600)`
is 90°, not 180°.

Three things that do **not** work on this scan, so as not to retry them:

* an **FBP focus sweep** on a slice built from `proj_bin0` — tv, g95 and entropy
  all run to the edges of a −256…+44 sweep with no interior vertex, because
  `paganin=60` leaves nothing to focus on;
* the **mirror on angular differences** (`A_j − A_{j+1}` mirrors onto
  `B_{1199−j} − B_{1198−j}`, which cancels the static halo exactly) — sound in
  principle, pure stitch noise at 0.15° spacing;
* **2-D Hann-windowed correlation** on `proj_bin0`, which reads −8.3 because the
  mirror-symmetric halo pulls the peak toward zero shift.

Re-quoting the axis costs only `start_step=5` — steps 1–4 never see it — and the
value must change in all four configs together.

### Why the retake pair, and why the scan pair is useless here

This is the one place this scan behaves differently from every other folder in
the repository, so the reasoning is spelled out — it is what makes the 25.6 px
gap worth taking seriously rather than dismissing.

A finished ID16A scan writes three extra frames after the last scan point, at
ω = 180, 90 and 0. That gives two ways to build an exactly-opposed pair:

* **(frame 0, frame 12000)** — the pair the HT folder used. Nominally 180.000°
  apart, but separated by the **whole scan**, so it carries every bit of drift,
  thermal movement and beam wander accumulated over 2 h 33 min.
* **(frame 12002, frame 12000)** — ω=0 retake against ω=180 retake. Also exactly
  180.000° apart, and **written two frames apart** at the very end of `scan0008`
  — under a minute, against 2 h 33 min.

On this scan that distinction is the whole ballgame. The scan pair scatters over
160 px across bands and crops (−80.6, −127.0, −8.5 px at crop 2048): the
correlation locks onto residual illumination rather than the sample. The retake
pair repeats to about 1 px.

`python estimate_center.py config_steps15.conf --pairs 1`, three horizontal
bands, at three crops:

```
crop 2048   +9.51 +- 0.50   (5 of 9 bands kept)
crop 2560   +8.78 +- 1.45   (6 of 9)
crop 3072   +8.98 +- 1.65   (6 of 9)
```

In **all three**, every band that survived the |dy| and 3-MAD rejections came
from a retake pair, and every band of the (0, 12000) scan pair was thrown out
automatically. Mean of the three crops = **+9.09**, the same answer the
2026-09-08 `--retakes on --bands 4` run gives to 0.7 px.

![rotation centre](center_estimate.png)

Convention, as in `../Y350a_HT/config_steps15.conf`:
`rotation_center_shift = c − n/2`, with `c` the axis column on the bin-0
detector grid.

The retake machinery is new in this folder's `estimate_center.py`
(`--retakes {auto,on,off}`, `--search`, `opposed_pairs`, `local_peak`). It is
skipped automatically when a scan has no retakes, or when the displacement table
has no rows for them — the 2025 tables stop at `ntheta+1` rows, so those scans
fall back to the scan pair, which is what they always used. Nothing in the older
folders changes behaviour.

**Still refine it.** ±0.5 px at 7.5 nm is ±3.8 nm, and the estimate comes from
two frames. After steps 1–5, run
`python step5_center_sweep.py config_steps15.conf` on the FBP volume and update
all four configs if it disagrees.

## Shrinkage — measured here as absent, but ESRF's value is what we use

> **2026-09-27:** the drop brought `shrink_list.mat` (row 0: h 4356 ppm,
> v 6221 ppm) and Peter applied it (`<pfile>_rec_.info`: `correct_shrink = 2`).
> `load_shrink_from_mats` therefore no longer returns zeros, and `rho[tp]=0`
> freezes tp on ESRF's ramp rather than on nothing.  That is deliberate -- it
> is what reproduces his final rec.  The measurement below still says those
> numbers are indistinguishable from zero at our noise level; treat it as a
> disagreement with ESRF that costs a fixed sub-pixel warp, not as a reason to
> delete the file.  See [Matching Peter's final reconstruction](#matching-peters-final-reconstruction).

`python estimate_shrink.py config_steps15.conf --grid 5`, all 25 blocks fitting
at every crop:

```
omega=90 vs frame 6000 (dt=0.50):   A_y = +196 +- 281 ppm    A_x =   +5 +- 807 ppm
omega= 0 vs frame    0 (dt=1.00):   A_y = +183 +- 173 ppm    A_x =  -26 +-  98 ppm
```

The two pairs agree to 0.0σ and every term is within 2σ of zero. Hence
`rho[tp] = 0` in all three step-6 configs — the linear shrink parameters A, B
are not refined. `../Y350a_HT` is the counter-example where they are.

![shrinkage](shrink_estimate.png)

## `nobj` = 4800

The grid must hold the sample **plus** the whole displacement sweep. At
`ndist=1`, `norm_mag = 1` exactly, so the commanded ±300 *detector* px are ±300
*object* px:

```
4096 + 2*300 = 4696  ->  4800 = 75*64
```

4800/2 = 2400 and 4800/4 = 1200, so it bins cleanly over all three levels.

This is the one place the single-distance scan is *easier* than the 4-distance
one: with a single plane there is no demagnified plane whose detector footprint
overflows the grid, so the 352 px margin comfortably exceeds the 300 px sweep and
`Rec._build_data_mask` should keep ≈1.000 of the detector pixels at every angle.
Anything much below 1.0 means something else is wrong — most likely a shift file
in the wrong units. (`mask_oob` is set, but at `ndist=1` with this margin it is a
non-event.)

## `paganin` = 60

`delta/beta = 60`, the same value as the HT scan of the **same sample at the
same energy** — and, as there, a deliberate override of the `delta_beta = 20`
Peter's HT driver used. 20 leaves the step-5 volume too noisy to be a useful
starting point; 60 smooths it without flattening the boundaries step 6 then
sharpens. There is no driver for this scan to read a value from in any case.

Single-distance Paganin is a much weaker starting point than the 4-distance
version — one propagation distance cannot separate phase from absorption, so the
transport filter is doing the whole job. Expect the bin-2 stage of step 6 to have
more work to do than it did on the HT scan, and look at the initial volume before
trusting the ladder:

```bash
python show_slices.py config_steps15.conf --init
```

`paganin` must match in all four configs: step 5 writes
`/exchange/obj_init_re{paganin}_{bin}` and step 6 reads that exact name.

## Probe — **available here**, unlike the HT scan

The HT folder cannot retrieve a probe: its NFP virtual datasets point into a
`RAW_DATA/` tree that was never copied. **This scan's was.** The NFP companion
`projections/<pfile>_NFP_before_0001.nx` (20 darks + 50 frames) resolves through
`NxFrames`, verified locally: `missing = []`, 2 sources, dark mean 99.8, NFP
frame means ≈888, geometry identical to the projection scan (z1 6.163535 mm,
voxel 7.49999578 nm, mag 196.80×), positions y [−99.0, +102.0] px /
x [−106.9, +99.1] px → `pos_range` 115 → `nobj` 4352.  The piezo positions
come from `sample/x_translation` / `y_translation` of that NXtomo (mm), not
from a PCIe header as in the 2025 bliss route.  Some ESRF trees also carry a
`<pfile>_NFP_after_0001`; this scan wrote only `_before` — point `nx_file` at
the other one to compare.  `rho` in `config_step0.conf` is copied from
`../Siemens/config_step0.conf`, the only other NXtomo NFP config in the tree.

[`step0.py`](step0.py) here is `../Siemens/step0.py` (which already speaks
NXtomo, via `parse_args_step0_nx` and `read_nxtomo_meta`) with the direct
virtual-dataset reads swapped for `NxFrames`. Run it, then uncomment `prb_file`
in `config_step6_bin2.conf`:

```bash
# uncomment the step0.py mpiexec line, then:
qsub polaris_run.sh                        # writes <path_out>/nfp_results.h5
```

Until then step 6 starts from a flat probe and refines it (`rho[1] = 0.05`).
Bins 1 and 0 inherit the probe from the previous level's checkpoint, so
`prb_file` only ever goes in the bin-2 config.

## Step 7 — refining the per-angle drift

[`step7.py`](step7.py) re-projects a step-6 checkpoint and searches for the
per-angle shift that minimises the entropy of the FBP. One GPU, no MPI:

```bash
python step7.py config_step6_bin2.conf --dry-run      # geometry and units
python step7.py config_step6_bin2.conf                # ~20 min
```

It writes `correct_correct3D_extra.txt` into this directory, in Peter's layout
and binned pixels. Step 6 adds it to the positions it reads
(`correct3d_extra=1` in `config_step6_*.conf`), so the next reconstruction is
a **step-6 rerun** — steps15 is not involved and `cshifts_final` does not
change. `correct3d_extra=0` ignores the file without deleting it, which is how
the with/without pair gets reconstructed.

**What it cannot tell you.** The volume it re-projects was reconstructed with
the current shifts, so part of the answer is the metric's own bias rather than
a residual misalignment, and a rigid object translation is invisible to it by
construction. Run `tests/find_shifts_extra/test_find_shifts_null.py` at the
same grid, angle count and slab, and treat an answer of that size as noise.

## Running it on Polaris

```bash
ssh polaris
cd /eagle/APS_IRI/vnikitin/holotomocupy_gpu_reduced/experimental/ctxl_FT_4K_RD300_007p5nm
source ../polaris_env.sh
qsub polaris_run.sh                                # steps15 + bin2 + bin1 + bin0
qstat -u $USER                                     # watch it
tail -f slurm-*.out                                # the job log, in this directory
```

There used to be a `check_data_read.py` pre-flight here -- it resolved the
paths step 3 would read, checked the shift files' shapes and units, the bin
factors, `ref_dist`, the drift and `rotation_center_shift`, and
`polaris_run.sh` ran it before the healthcheck. It is gone, exactly as in
[../ctxl_HT_4K_RD300_007p5nm](../ctxl_HT_4K_RD300_007p5nm/README.md): the
script was never committed and the `python ../check_data_read.py` line only
aborted the job. The checks that still matter are made by `steps15.py` itself
through [`esrf_meta.py`](../../src/holotomocupy/esrf_meta.py), and they appear
in the run log.

[`polaris_run.sh`](polaris_run.sh) is one literal `mpiexec` line per stage, in
sequence. To run only part of it — the NFP probe alone, steps 1–5 already done,
or resuming after a preemption — **comment out the lines you do not want and
uncomment the opt-in ones you do** (`step0.py` is commented out by default).
Each line ends in `|| exit $?`, so a failed stage stops the job instead of
letting the next level seed itself from a checkpoint that was never written.

Recommended first pass:

```bash
# uncomment step0.py, comment out steps15 + the three step-6 lines:
qsub polaris_run.sh                                   # 0. optional, then uncomment prb_file
# leave only steps15 uncommented:
qsub polaris_run.sh
python step5_center_sweep.py config_steps15.conf      # 1. confirm +9.80 from the FBP volume
python show_slices.py config_steps15.conf --init      # 2. eyeball the Paganin volume
# leave only the three step-6 lines uncommented:
qsub polaris_run.sh
```

**The drift file is already installed** — `correct_motion.txt` has been at
`<path>/<pfile>/projections/` since 2026-09-01 (12003 rows), and step 3 logs
the path it resolves it to. It is **ours**, measured from
this scan's post-scan retakes by [`estimate_motion.py`](estimate_motion.py);
its horizontal column is identically zero, so all of the x correction here is
the random displacement.

**ESRF supplied nothing for this scan.** Unlike the `ctxl_HT` sibling and
`../AtomiumS1_FT_RD300`, eagle held no `<pfile>_0003_` octave directory at all: no driver
`ht_<pfile>.m`, no `<pfile>_rec_.info`, no `rhapp.mat`, no
`correct_correct3D.txt`, and no `naburec/` — so no Peter shifts to install and
no ESRF rotation axis to read out. `rhapp.mat` missing is expected anyway — one
distance, nothing to register — and step 3 logs *"using zeros"*.

### Borrowed `correct_correct3D.txt`

ESRF produced none for this scan. The file in the drop is **the other FT scan's**
— `AtomiumS1/Atomium_S1_FT_4K_RD300_004p5nm_0001_/correct_correct3D.txt`,
Peter's, 12001 rows — copied in **row for row** and multiplied by
**4.5/7.5 = 0.6**.

**Row for row**, because the two FT scans share an angle list: `angles_file.txt`
is 0…180° in 0.0149…° steps, 12001 unique angles plus the three retakes, the
same in both. Row *i* is the same angle in both files, so nothing is
interpolated.

**× 0.6**, because the file is in the pixels of ESRF's own 2×2-binned
reconstruction grid, and that grid is 9 nm for AtomiumS1 against 15 nm here.
Read at face value the correction would be applied at 167 % of its true size.

The factor is measurable, not just nominal. Peter made a `correct_correct3D.txt`
for both ctxl_HT and AtomiumS1 FT; least-squares fitting one against the other:

| column | scale | residual |
|---|---|---|
| horizontal | **1.66667** (= 7.5/4.5 exactly) | 0.0040 px rms, of a 34 px range |
| vertical | 1.612 | 0.65 px rms, of a 20.5 px range |

The horizontal column of the two is **one physical curve in two unit systems**,
which is the evidence that correct3D is a property of the *rotation* — stage
wobble, axis tilt — and repeats between scans on this stage. The result lands at
20.510 × 12.325 binned px, against ctxl_HT's own 20.503 × 12.337; that agreement
is the check that the units came out right.

**What it still assumes.** The vertical column is the weak half: ~3 % of its
range is genuinely per-scan by the fit above, and whatever part of correct3D is
slow drift over scan time does not repeat at all and is being transplanted as if
it did. Treat it as a starting point for step 6 to refine (`rho[pos] > 0` at
every level), not as a measurement of this scan. 41 × 25 raw px of ptp sits
inside the 352 px grid margin, so a wrong transplant costs convergence speed,
not validity. **Delete it the moment Peter drops a real one**, and check
`<pfile>_rec_.info` then — `PixelSize / voxel size` is `correct3d_bin`.

Its horizontal mean is +1.64 raw px, which is degenerate with
`rotation_center_shift` — and the +9.80 above does *not* include it:
`estimate_center.py` works on raw holograms with only the random displacement
undone. That is the same footing as the two ESRF controls it was checked
against, whose `correct_correct3D.txt` horizontal means are the same +1.64, so
the term sits inside their ~4 px errors rather than unaccounted for.

**Superseded.** Until this replaced it, the file was ctxl_HT's 4001-row original
resampled onto these 12001 angles. Same physical curve, but it needed an
interpolation this one does not, and its vertical column was an HT scan's rather
than an FT scan's. Kept in the drop for comparison:
`correct_correct3D.txt.bak_20260907-220522`, `correct_correct3D_HT_source.txt`,
`correct_correct3D.txt.bak_premean`. Current provenance, with the numbers, is in
`README_provenance.txt`; the verbatim source is
`correct_correct3D_S1FT_source.txt`.

`rotation_center_shift` **is no longer borrowed.** It was, from ctxl_HT's
`naburec/`, under the same rule as `correct_correct3D.txt` — but unlike
`correct_correct3D.txt` this scan has its own measurement, **+9.82 ± 0.41 px**
from [`estimate_center.py`](estimate_center.py) on the post-scan retakes, and
that is 25.6 px from the borrowed −15.77. A day between the scans (HT Aug 31,
FT Sep 01) and a re-centred stage account for it, and AtomiumS1 shows the same
HT/FT split (+44.20 vs +10.74). All four configs now carry **+9.80**; see
[Rotation centre](#rotation-centre--980-px-measured-from-the-post-scan-retakes)
above.

The three step-6 levels share one `path_out`, so each seeds itself from the
previous level's checkpoint via `start_iter`, and `Reader.read_checkpoint`
upsamples obj/prb/pos onto the finer grid. Iteration numbering is cumulative:

| config | bin | n | nobj | start_iter | niter | nchunk | `lam_laplacian` | `rho` |
|---|---|---|---|---|---|---|---|---|
| `config_step6_bin2.conf` | 2 (4×4) | 1024 | 1200 | 0 | 1025 | 32 | 1e−4 | 1.0, 0.0125, 0.04, 0 |
| `config_step6_bin1.conf` | 1 (2×2) | 2048 | 2400 | 1024 | 1281 | 8 | 5e−5 | 1.0, 0.0125, 0.04, 0 |
| `config_step6_bin0.conf` | 0 (1×1) | 4096 | 4800 | 1280 | 1537 | 4 | 0 | 1.0, 0.0125, 0.04, 0 |

`checkpoint_step=32` divides every `start_iter` and every `niter−1`, so the
handoff checkpoints are guaranteed to exist. Preemption is survivable: a
resubmit loses at most 32 iterations. `estimate_rho=True` only at bin 2.

**Walltime.** No timings exist for this scan. Against the measured `../Y350a_HT`
ladder (4 distances, ntheta=4000, n=4096, nobj=4608; 13 min + 22 min + 1 h 47 min
≈ 2 h 25 min on 2 nodes / 8 ranks), this scan is 3× the angles but ¼ the
distances, ≈0.8× overall — and the ladder here is 1536 iterations, not 1024, so
scale by 1.5 → **≈3 h 30 min of step 6**. Steps 1–5 read 402 GB and
write ~1.1 TB and have never been timed on an `nxvds` scan — that is the real
unknown, and why the script asks for 18 h. Trim it once the first `.out` exists.

**Disk**, in the steps15 `path_out`:

| file | size |
|---|---|
| `<pfile>.h5` (12000 × 4096² × 2 B × 1 dist) | 402 GB |
| bin-0 pdata (× 4 B × 1 dist) | 805 GB |
| `<pfile>_obj.h5` (1200³ × 4 B × 2, bin 2 only) | 14 GB |

≈1.2 TB as configured, against 231 TB free on eagle as of 2026-09-01.

`start_level_rec=2` means step 5 writes a Paganin+FBP init **only for bin 2** —
the first rung of the step-6 ladder. Bins 1 and 0 are seeded from the checkpoint
the level below wrote, so they never need one; the same setting is used by
ctxl_HT and AtomiumS1_HT. Step 4 still writes all three binned pdata levels.
Set `start_level_rec=0` if you want full-resolution FBP volumes to look at —
that replaces the 14 GB above with 885 GB (4800³ × 4 B × 2) for ≈2.1 TB total.

## Files

| file | what |
|---|---|
| [`esrf_layout.py`](esrf_layout.py) | the only place that knows bliss from ewoks from **nxvds**; filenames, geometry, `NxFrames` VDS re-basing |
| [`config_steps15.conf`](config_steps15.conf) | steps 1–5 |
| [`config_step6_bin{2,1,0}.conf`](config_step6_bin2.conf) | the BH ladder, 7.5 nm, `model=intensity` |
| [`config_step6_binned_bin{2,1}.conf`](config_step6_binned_bin2.conf) | the **binned** (15 nm) arm: bin 2 -> bin 1 only, `model=amplitude`, `path_out` `..._rec6_binned` |
| [`config_step0.conf`](config_step0.conf) | NFP probe retrieval |
| [`correct_motion.txt`](correct_motion.txt) | **measured here** — random displacement + drift, 12003×2; already installed in `<pfile>/projections/` |
| [`polaris_run.sh`](polaris_run.sh) | PBS job, one `mpiexec` line per stage; comment out what you do not want |
| [`show_geometry.py`](show_geometry.py) | derived geometry and the per-level config blocks |
| [`scan_overview.py`](scan_overview.py) | the overview figure above |
| [`estimate_center.py`](estimate_center.py) | rotation centre from opposed projections — **retake pairs**, `--retakes/--search` |
| [`estimate_motion.py`](estimate_motion.py) | drift from the post-scan retakes; writes `correct_motion.txt` |
| [`fig_retake_diff.py`](fig_retake_diff.py) | each retake against its counterpart at the start of the scan, and the difference |
| [`estimate_shrink.py`](estimate_shrink.py) | shrinkage from the post-scan retakes |
| [`step5_center_sweep.py`](step5_center_sweep.py) | centre refinement from the FBP volume |
| [`show_slices.py`](show_slices.py) | the step-5 initial volume |
| [`show_iter.py`](show_iter.py) | one step-6 checkpoint: slices, probe, positions, convergence |
| [`steps15.py`](steps15.py) | steps 1–5 driver |
| [`step6.py`](step6.py) | BH reconstruction driver |
| [`step0.py`](step0.py) | NFP probe retrieval — NXtomo + `NxFrames`, **works here** |
