# AtomiumS1_FT_RD300 — SINGLE-distance, ±300 px random displacement, 4.5 nm voxels

ESRF ID16A, proposal **blc17322**, beamtime 20260825.
Atomium sample S1, **one** propagation distance, 12000 projections over 180°,
±300 px random sample displacement, 4096² detector at 17.1 keV, **4.500 nm**
voxels.

**This folder processes the FT scan**, `Atomium_S1_FT_4K_RD300_004p5nm_0001`
(`.info` `Date` = 2026-09-01 11:24:59), with `steps15.py` / `step6.py` and
`config_steps15.conf` / `config_step6_bin{2,1,0}.conf`.

There is a second scan of the same sample in the same directory taken 25
minutes later, `Atomium_S1_NFP_4K_RD300_004p5nm_0001` (11:49:16). **It is not
processed from this folder.** The 2-D single-angle driver that used to
reconstruct it — `nfp2d.py`, `config_nfp2d{,_handyn}.conf`,
`polaris_run_nfp2d.sh`, and `config.parse_args_nfp2d` — was removed on
2026-09-10, as was the tomographic path for it before that. That scan is not
what `step0.py` in this folder reads: `step0.py` retrieves the probe from the
FT scan's own 70-frame NFP *companion*, `<pfile>_NFP_before_0001.nx`, which is a
different acquisition — see below.

Raw data: `/eagle/APS_IRI/vnikitin/20260829/AtomiumS1/`

Everything below is about the **FT** scan.

**The closest sibling is [`../ctxl_FT_4K_RD300_007p5nm`](../ctxl_FT_4K_RD300_007p5nm),
not the other Atomium folders.** Same scan recipe — one distance, 12000
projections over 180°, ±300 px, 4096², post-scan retakes — on a different
sample at a different energy and voxel size. Every script here is that folder's
copy, and every long note there applies unless this one says otherwise. What
`../AtomiumL1_HT` and `../AtomiumS2` share is only the sample name: they are
4-distance 33.35 keV scans on 2048² detectors from 2024/2025.

This scan has never been reconstructed. Every number below was **measured from
the raw data on 2026-09-03/04** or carried over from a named sibling and marked
so.

> **The pixels are ordinary EDF files.** Unlike the ctxl FT scan — which was
> never converted and had to read through an HDF5 virtual dataset into
> `RAW_DATA/` — this one is the `ewoks` flavour: EDF frames in `<pfile>_1_/`,
> geometry in the NXtomo. Nothing under `RAW_DATA/` is needed for the
> reconstruction itself. The one thing that does need it is NFP probe
> retrieval — see `step0.py` below.

| | AtomiumS1 (here) | `../ctxl_FT_4K_RD300_007p5nm` | `../ctxl_HT_4K_RD300_007p5nm` |
|---|---|---|---|
| layout | **ewoks (2026, EDF)** | nxvds (2026, no EDF) | ewoks (2026, EDF) |
| distances | **1** | 1 | 4 |
| projections | **12000** | 12000 | 4000 |
| energy | 17.1 keV | 17.1 keV | 17.1 keV |
| voxel size | **4.500 nm** | 7.500 nm | 7.500 nm |
| random displacement | ±300 px | ±300 px | ±300 px |
| `nobj` | **4096** (= n, no margin) | 4736 | 5056 |
| `rotation_center_shift` | **+13.55 px** (retake pair) | +9.09 px (retake pairs) | −19.22 px (weak) |
| `paganin` | **35** (set by hand, not measured) | 20 | 20 |
| `correct_motion.txt` | **measured here**, checked in — 114 px vertical, x zero | measured there | from ESRF |
| shrinkage | none — *measured* as absent | none — measured as absent | none — measured as absent |
| probe | `step0.py` present, needs 2 `RAW_DATA` files | measurable, `step0.py` works | not possible |

## What is on disk

Verified 2026-09-03, after the copy to eagle finished:

```
<pfile>_1_/          12003 projections, 4096² uint16, ~378 GB
                     ref0000_00{00..19}.edf    20 start flats
                     ref12000_00{00..19}.edf   20 end flats
                     dark00{00..19}.edf        20 darks
                     angles_file.txt, <pfile>_1_.info
                     correct_motion.txt        ← installed here 2026-09-03
<pfile>/projections/ <pfile>_0001.nx           geometry (NXtomo)
                     <pfile>_0001.txt          displacement table, 12003×2
                     <pfile>_NFP_before_0001.nx
```

There is **no `<pfile>_/` octave directory** — Peter's pipeline was never run on
this scan. So: no `rhapp.mat` (not needed at `ndist=1`), no `quali.mat`, no
`ht_<pfile>.m` driver (which is why `paganin` had to be argued rather than read),
and no `correct_motion.txt` from ESRF. The drift was measured here instead.

### Probe retrieval — `step0.py`

`step0.py` / `config_step0.conf` / `nx_frames.py` were added 2026-09-12 and are
the `../ctxl_FT_4K_RD300_007p5nm` drivers with the paths changed; the
reconstruction code is byte-identical. The NFP companion `<pfile>/projections/<pfile>_NFP_before_0001.nx` is on eagle — 70
frames, 20 darks + 50 NFP, geometry bit-for-bit the projection scan's (17.1 keV,
`z1` 3.698060 mm, focus→detector 1212.9760 mm, 328.00×, 4.49999929 nm) — but its
pixels are a virtual dataset into `RAW_DATA/`, which must be on eagle under
`/eagle/APS_IRI/vnikitin/20260829/RAW_DATA/`, beside `ctxl`:

```
Atomium_S1/<pfile>/scan0004/balor_0000.h5   20 darks    ~0.6 GiB
Atomium_S1/<pfile>/scan0005/balor_0000.h5   50 NFP frames  ~1.6 GiB
```

`<pfile>` is the **FT** scan, `Atomium_S1_FT_4K_RD300_004p5nm_0001`. Both it and
the sample directory name (`Atomium_S1`, with the underscore) are read out of
the NXtomo's virtual-source paths, so neither is ours to choose.

> **These are not in the `Atomium_S1_NFP_*` directories.** The `NFP_before`
> frames were taken as scans 4 and 5 of the FT acquisition itself — that is why
> the `.nx` lives in that scan's `projections/`. The standalone
> `Atomium_S1_NFP_4K_*` datasets are a different measurement and will not
> satisfy this file. Copying those instead leaves the guard failing with no
> obvious reason why.

HDF5 answers an unresolvable virtual source with **silent zeros**, so
`nx_frames.NxFrames` refuses rather than quietly returning a black probe. Check
where it stands without submitting a job — it prints the resolved source files,
marks the missing ones, says where each should be copied, and exits non-zero:

```
python nx_frames.py /eagle/APS_IRI/vnikitin/20260829/AtomiumS1/Atomium_S1_FT_4K_RD300_004p5nm_0001/projections/Atomium_S1_FT_4K_RD300_004p5nm_0001_NFP_before_0001.nx
```

Once those two files are on eagle, uncomment the `nfp` line in `polaris_run.sh`
(or `mpirun -n 4 python step0.py config_step0.conf`), then uncomment `prb_file`
in `config_step6_bin2.conf` only — bins 1 and 0 inherit the probe from the
previous level's checkpoint. The piezo raster is ±115 px (`y` −99.0…+102.0,
`x` −106.9…+99.1), the same raster ctxl scanned at 7.5 nm, so `nobj` comes out
at 4352 for `n=4096`.

Without it step 6 starts from a flat probe and refines it (`rho[prb] = 0.05`),
same as `../ctxl_HT_4K_RD300_007p5nm`. This is the only stage in the folder that
needs `RAW_DATA/` at all; `steps15.py` reads the EDFs.

#### Intensity misfit and the PSF (changed 2026-09-12)

`RecNFP` used to minimise `|| |psi| - d ||²` against amplitudes. It now
minimises `|| K|psi|² - d ||²` against **intensity**, with `K` one Gaussian on
the detector intensity — the same model, and now literally the same code path,
as step 6's `Rec.F0`; NFP simply has no out-of-grid detector mask, so `W = 1`.
`tests/nfp/test_f0_intensity.py` checks the two agree exactly (rel `0.0` at
every sigma, `w` term included) and Taylor-tests the derivatives against a
float64 numpy reference.

Two consequences worth knowing before comparing anything:

* **`err` is not comparable with NFP runs made before this date**, and neither
  are the `rho` that were tuned against the old misfit. `estimate_rho=True` in
  `config_step0.conf` is what makes that tolerable — it re-searches `rho[prb]`
  and `rho[pos]` against the new misfit at run time.
* every `step0.py` in `experimental/` now loads `raw / mean` instead of
  `sqrt(raw / mean)`. Feeding the old amplitude into the new misfit is wrong
  but not loud, so the change was made everywhere at once rather than here
  alone.

`psf_sigma=1.5053` in `config_step0.conf` is the 16 nm focal spot at
`M = 328.00` with a 1.476 µm pixel: `sigma = s*(M-1)/2.3548/detector_pixelsize`.
It is the same number step 6's sweep uses at bin 0 because this is the same
optic at the same plane, and it is **unbinned** here — the NFP frames are read
at the full 4096, so there is no bin factor to divide by. It is an imposed
width, not a fit; a wrong value is absorbed by the retrieved probe, which
matters because that probe is then handed to step 6 through `prb_file=`. To
check it, re-run at `0.0` and `3.0` and compare the **probe**, not `err` — `err`
moves when the model changes and says nothing across models.

`energy=` is now an optional key in `config_step0.conf` too (commented out by
default = take the 17.1 keV the `.nx` records). `step0.py` logs which of the two
it used.

## Frame numbering

```
    0 .. 11999   the scan, 0.000 .. 179.985° in 0.015° steps
       12000     the last scan point, 180.000002°
       12001     post-scan retake at  90°  (repeats frame 6000)
       12002     post-scan retake at   0°  (repeats frame    0)
```

The displacement table has a row for all 12003 and is **zero** at frames
0 / 6000 / 11999 and at all three retakes, so each retake repeats its
counterpart at the same commanded position. That is what makes the drift, the
shrinkage and the rotation axis measurable from this scan alone.

## Geometry

`python show_geometry.py config_steps15.conf`, read off the real data:

| | value |
|---|---|
| energy | 17.1 keV |
| detector pixel | 1.476015 µm |
| focus → detector | 1212.976 mm |
| z1 | 3.698060 mm |
| magnification | 328.00× |
| **voxel** | **4.49999929 nm** |
| propagation distance | 3.68679 mm |
| `norm_magnifications` | [1.0] |
| detector | 4096², 20 flats, 20 darks |

The `.info` sidecar's `PixelSize = 0.0045 µm` reproduces the derived value and
is written by a different part of the beamline software than the NXtomo, so
that agreement **is** an independent check on the distances.

## Rotation centre — magnitude **13.5 px**, config value **−13.00**

> **The config takes the negative of what `estimate_center.py` prints.**
> `estimate_center.py` measures the axis as `c − n/2` on the raw detector
> frames and gets **+13.55**; that number is right and everything below it
> stands. But step 5 applies `rotation_center_shift` with the *opposite* sign
> to that definition, so the configs here carry **`rotation_center_shift =
> −13.00`**. This was settled on 2026-09-04 by restitching the opposed pair at
> every candidate value (`fig_center_sweep.py`): the residual bottoms at −14.1
> / −13.6 and the lag crosses zero at −13.9 / −13.5, while the whole of
> +8 … +20 is monotone with no turnover at all. See the note at the top of
> `config_step6_bin2.conf`, and the **open question** at the end of this
> section about the other nine experiments.

Measured from the post-scan retakes with `estimate_center.py`, and this sample
made it harder than the sibling did.

The **scan pair (0, 12000)** is useless, exactly as on ctxl FT: its three bands
read −101.5 / −71.0 / +51.6 px at crop 2048 and swing to −115.0 / −16.9 / +20.5
at crop 3072, with `|dy|` up to 100 px on a pair that is physically 180° apart.
`phase_corr` is locking onto residual illumination, not the sample.

**What is different here is that the band aggregate fails too.** Averaging the
three bands over five crops gives anything from −17 to +14 px. The sample sits
in the middle of the field, so bands 0 and 2 (top and bottom thirds) are
illumination-dominated and wander with the crop. The **central band of the
exactly-opposed retake pair (12002, 12000)** does not:

| crop | shift | dy |
|---|---|---|
| 2048 | +13.48 | +0.79 |
| 2560 | +13.55 | +0.80 |
| 3072 | +13.56 | −0.16 |
| 3584 | +13.59 | −0.02 |

Four crops over a 1.75× range, spread **0.11 px**, with `|dy|` under a pixel
throughout. The second retake pair (12002, 11999 — 0.015° off exact)
corroborates at crop 2048 (+13.55, matching to 0.07 px) but breaks down above
it (+99.4 at crop 2560), so it is supporting evidence in the small-crop regime
only.

**Why it is no longer +12.93.** That value, measured 2026-09-03, was taken
against the *previous* `correct_motion.txt`, which carried +0.596 px of
horizontal drift at the end of the scan; without that file the same bands read
+13.53. A constant in the horizontal column is *exactly degenerate* with the
axis, so those were never two measurements — only two conventions. The motion
file has since been rebuilt and **its horizontal column is identically zero**,
which moves the axis back by that same 0.6 px. Re-measuring against the file
now installed gives +13.55, which matches the old no-motion value to 0.02 px.
The rule is unchanged: the axis and `<path>/<pfile>_1_/correct_motion.txt` are
one setting, never changed apart.

**And +13.55 is the axis for the whole scan, not just for the end where the
retakes sit** — because the scan turns out to have no horizontal drift at all
(next section). Had there been one, the retake pair would have measured the
axis at the end only.

0.11 px of crop-to-crop spread is not a 0.11 px error bar — it says the central
band is self-consistent, not that it is right, and the bands either side
disagree by whole pixels. At 4.5 nm voxels one pixel is 4.5 nm.

**Open question — the other nine experiments.** If this inverted sign is the
pipeline's convention rather than something local to this scan, then every
config whose axis came out of `estimate_center.py` is off by *twice* its value:
ctxl_HT −19.22, ctxl_FT +9.09, AtomiumL1_HT −14.376, AtomiumS2 −27.00,
Y350a_dist1234 −8.78, Y350a_HT −10.39, Y350a_largedisp −10.97, Y350c 145.07,
y350a_80um 35.66. The fix belongs in exactly one of two places and they are not
equivalent: negating what `estimate_center.py` prints invalidates only the
configs derived from it, while flipping the sign in `steps15.py` invalidates
*every* config including the hand-tuned ones. **Not decided — do not touch
another folder until it is.**

### `step5_center_sweep.py` cannot refine it, but the opposed pair can

The obvious next step, and what the sibling folder's README tells you to do, is
to refine the centre on the step-5 volume with `step5_center_sweep.py`. **That
does not work on this scan**, and it is worth writing down why so nobody spends
another hour on it.

The way out is not to score the *volume* at all. `fig_center_sweep.py` scores
the opposed-pair residual instead: no Paganin filter in the loop for the
`srdata` row, and a lever arm of two raw px of lag per raw px of axis. That is
what produced the sign correction above, and it is the tool to reach for next
time.

Two sweeps were run (jobs 7590148 and 7590168, 24 s and 35 s): +3…+23 step 1,
then −100…+100 step 5. Scored on the full frame with three focus metrics:

| | result |
|---|---|
| gradient energy | broad peak, parabolic vertex **−9.14 px** |
| Laplacian | **monotone** to the −100 edge |
| total variation | **monotone** to the −100 edge |
| gradE peak-to-trough | **0.55 %** of the mean over a 200 px sweep |

Three metrics on the same data, one broad peak and two non-peaks 90 px away.
That is not a measurement.

The reason is that step 5 is a **single-distance Paganin** reconstruction, and
Paganin is a low-pass of characteristic length
`sqrt((δ/β)·λ·z/4π)` = `sqrt(32 · 0.725 Å · 3.687 mm / 4π)` = **825 nm =
183 unbinned voxels** — about 180× the precision wanted from the centre. The
focus curve is correspondingly broad and shallow, exactly as measured. Sweeping
at bin 0 does not help: the blur is physical, not a sampling limit.

**There is also a trap in how you score it.** Changing the assumed centre
*translates* the reconstructed features — measured here at ~1:1, 62 px of travel
across the 200 px sweep — so a fixed sub-window measures the object sliding
through the window, not its sharpness. Scoring the central 50 % put the gradient
peak at −10 px and the Laplacian peak at **+40**, anti-correlated with each
other. [`score_sweep.py`](score_sweep.py) therefore scores the full frame,
prints the feature centre-of-mass travel, and warns when the contrast is under
2 %; [`inspect_sweep.py`](inspect_sweep.py) shows the first/last/difference
triptych that makes the translation obvious.

So the sweep rules out a *grossly* wrong centre — it is consistent with +13.55
in the sense that it cannot distinguish +13.55 from −9 — and that is all it can
do. **The retake measurement stands as the number**, because it is a direct
geometric measurement of two frames 180° apart and does not go through the
Paganin filter at all.

**The real check is after bin 2.** The step-6 object is reconstructed without
the Paganin low-pass and is far sharper; look for split double-edges in
`show_iter.py` and adjust all four configs together if any appear. Note that
`rotation_center_shift` is *not* only a step-5 seed — `reader.py:233` adds it to
every shift step 6 reads — but step 6 also optimises per-projection positions
(`rho[pos]` = 0.04/0.02/0.01), which can absorb a residual constant offset.

### Re-measured on the corrected projections — **−13.3 ± 0.7**, so −13.00 stands

The section above measured the axis before the drift was known. Once the new
`correct_motion.txt` (12:19) and `correct_correct3D.txt` (23:16) were installed
and steps 3–5 re-run from them (`start_step=3`, `_proj.h5` written 23:30, still
with `rotation_center_shift=-13`), the question is worth asking again on the
corrected projections: **it comes back −13.3 ± 0.7, i.e. unchanged.** Two
independent methods, neither of which is the retake pair:

| method | answer | scatter |
|---|---|---|
| opposed-pair mirror on the stitched projections, `mirror_center_proj.py` | **−13.3** | ±0.7 over 6 pairs, crop 0.95 |
| registered focus sweep, entropy, `sweep_center_registered.py` | −15.2 | ±0.4 over 4 rows |
| the same sweep, total variation | −18.6 | |
| the same sweep, 95th-percentile gradient | −16.9 | |

The mirror number is the one to quote: it is a correlation of two frames 180°
apart, so no Paganin low-pass, no focus metric, and no baseline. Its scatter
*falls* as the crop grows (2.1 px at crop 0.45 → 0.7 px at 0.95), which is what
a real peak does and noise does not. All six pairs pass the `|dy| ≤ 3 px` gate
at `|dy| ≤ 1 px` — and since pair (0, 11999) straddles the entire scan, that is
also a **direct confirmation that the new `correct_motion.txt` removed the
drift**: on the uncorrected data that pair carried all of it.

#### Three ways this measurement lies, all of them found here

**1. The first moment is biased on this object — do not use it.**
`estimate_center_com.py` fits `m(θ) = c + x₀cos θ + y₀sin θ` to the sinogram's
centre of mass. It is exact for a compact object, it uses all 1500 angles, and
it does not measure sharpness, so it looks like the right tool. It is not: it
returns **c = 518.3, i.e. `rotation_center_shift = +12.34`**, 25 px from every
other method. Reconstructing at +12.34 and at −13 side by side
(`arbitrate_center.py`, `arbitrate_center.png`) settles it in one look — +12.34
has vertical streaks across the whole slice, a doubled central blob and visibly
softer rings. The sinogram is *not* truncated (edges sit at +0.002/−0.015
against −0.56 in the middle) and the moment does follow a clean sinusoid, so the
usual explanations do not apply; the estimator is simply biased by this object's
broad Paganin background, and re-projecting the reconstruction to make a
synthetic with a known axis reproduces a bias of the same few-px order. The
script is kept only because the *calibration* inside it is reusable — see below.

**2. A 180° axis error translates the slice, so a fixed crop measures the
translation.** The README above already found this at ~1:1 across a 200 px
sweep. The reason is geometric: over 180° a point smears into a **semicircle**,
not a circle, and a semicircle's centroid is offset from its centre by `2d/π ≈
0.64 d`. `sweep_center_registered.py` cross-correlates every candidate back onto
the `d=0` reconstruction before scoring — the measured travel is `(−5.2, +0.9)`
px at `d = +4.75` — and with that one line the three metrics stop minimising 9
raw px apart.

**3. Sub-pixel Fourier shifts smooth the image, so a sweep on a fine grid
measures the interpolation.** After registration, `tv` and the gradient metric
were still oscillating with period 4 — minimising at exactly `d = +4, +1, −2`,
the **integer** sinogram shifts, because an integer shift does no interpolation
and so looks sharper. Entropy is immune (it is a histogram statistic). The fix
is to put every candidate on a grid of constant fractional `d`: with
`rotation_center_shift = rcs₀ − 4k` all shifts are integer, and the two such
grids run here (integer `d` and half-integer `d`) agree to **0.2 px** on all
three metrics — the sign that the artefact is gone.

#### The pipeline's axis column is exactly 512.0, measured not assumed

Turning a measured axis column into a config value needs to know which column
the pipeline's own FBP puts the axis on. Rather than derive it from the usfft
gridding, `estimate_center_com.py:pipeline_axis_column` forward-projects a
single delta at `(n/2, n/2)` with the same `holotomocupy.tomo.Tomo` and fits the
same three parameters: **c = 512.0000 with zero residual and zero cos/sin
amplitude**, so the grid centre *is* the axis and there is no half-pixel to
argue about. That calibration is what lets `mirror_center_proj.py` convert its
column to `rotation_center_shift = rcs₀ − (c − 512)·2^bin`, and the fact that it
then lands within 2 px of the focus sweep — which assumes no convention at all,
being anchored on `rcs₀` — cross-validates both.

#### No axis tilt

Entropy vertices per row (384/512/640/768): −15.4, −15.2, −15.4, −14.5 — no
monotone drift, consistent with the 0.04° bound in the tilt section below. Row
256 is an outlier (−28.6) on all three metrics; that slice has little structure.

**Bottom line:** the best combined estimate is **−14 ± 2**, the mirror
measurement alone is **−13.3 ± 0.7**, and the installed **−13.00 is inside both**.
There is no justification for changing the configs, particularly as step 6
optimises per-projection positions and can absorb a constant offset of this size.

## Drift — 114 px vertical, nothing horizontal

> **Superseded as of 2026-10-07: step 3 now measures this itself.**
> `estimate_quali_motion.py` runs between the random shifts and the rhapp
> search, correlating each post-scan retake against its in-scan twin and
> fitting a mean-removed quadratic through the three points — see the
> ["Sample drift" section of the HT_RD300
> README](../AtomiumS1_HT_RD300/README.md#sample-drift-from-the-post-scan-retakes)
> for the method, which is shared. **No ESRF shift file is read by the pipeline
> any more**, `correct_motion.txt` included; it is a `--validate` target now.
>
> It is the same retake pair the failed attempt below used, and it avoids that
> trap by flat-field correcting first, subtracting a static template, and
> **predicting** where the detector-fixed illumination will correlate
> (`r_retake − r_scan`, known from the shift table) so that peak can be masked
> rather than hoped away. The sections below are the investigation that
> established the 114 px number and are kept as the independent cross-check it
> is validated against.

The installed `correct_motion.txt` carries a **vertical** drift ramping to
114 raw px over the scan and an **identically zero** horizontal column. Both
halves of that were measured; the sections below give the measurements
(vertical from the Paganin projections, cross-checked on the raw holograms;
horizontal from the retakes, [below](#horizontal-drift--measured-on-purpose-and-it-is-zero)).

**The first attempt got it backwards, and the trap is worth knowing.**
`estimate_motion.py config_steps15.conf --template 48` correlates each retake
against its twin and returned the opposite picture — dy ≈ −0.1 px, dx ≈ +0.7,
"almost all horizontal", with a crop-to-crop MAD of 0.06 px that made it look
like a 5σ measurement:

| retake | vs frame | dy (wrong) | dx (wrong) |
|---|---|---|---|
| 12001 (ω=90) | 6000 | −0.07 | +0.71 |
| 12002 (ω=0) | 0 | −0.15 | +0.72 |

Every one of those numbers is the detector-fixed residual, not the sample.
Frames 0, 6000 and both retakes are the only frames commanded to **zero**
displacement, so in a retake/twin pair the flat-field residual is aligned at
lag 0 in both frames and pins the correlation there — a sharp one-pixel spike
sitting on top of the broad, real, off-centre hump. Tight crop-to-crop
agreement measures the stability of that spike, not of the sample. The true
vertical motion in those same two pairs is −117 and −58 px.

So **do not use `estimate_motion.py` on this scan.** Use
`estimate_drift_proj.py` (vertical, from the Paganin projections) and
`estimate_retake_drift.py` (horizontal, with the static template removed and
the zero lag vetoed). `--zero-x` is not needed as a flag here because the
horizontal column comes out zero on its own merits.

`correct_motion.txt` (12003 × 2) is checked into this folder and **already
installed** at

```
<path>/<pfile>_1_/correct_motion.txt
```

**Note the directory.** Step 3 reads it from `lay.dname(ref_dist)`, which for an
EDF flavour is the *distance* directory `<pfile>_1_/` — **not**
`<pfile>/projections/`, which is what the ctxl FT README says only because that
scan is `nxvds` and has no distance directory at all. Copying it to
`projections/` here would leave step 3 logging "not found, using zeros" with the
file sitting right there.

At `ndist=1` step 3 forms `(correct_motion − random[ref_dist])` and adds the
random displacement straight back, so the file **replaces** the random
displacement rather than adding to it, and must contain displacement + drift in
raw detector pixels. `estimate_motion.py` writes exactly that.

## Drift, checked a second way — from the projections, and it is 114 px

`estimate_motion.py` above works from the two post-scan retakes and reports
**0.15 px** of vertical motion across the whole scan. The truth is **114 px**,
so that number is simply wrong — and the reason matters, because the first
explanation written here was also wrong.

**Discredited, kept as a warning:** this section used to argue the retakes were
*structurally blind* — that each retake is compared with the scan frame at the
same ω, so a monotone drift is disguised as an angle-dependent effect and
cancels. **That reasoning is false.** Frame 0 and retake 12002 are both at
ω = 0 with the same (zero) commanded displacement, taken ~70 min apart. Nothing
cancels: their difference *is* the total drift, with no model in between. The
retake pair is the single most direct measurement in the scan, and re-measuring
it on the raw holograms returns **+118.2 px** (see the cross-check below).
`estimate_motion.py`'s 0.15 px is a correlator failure — a lock onto the static
illumination/detector background — not a property of the geometry.

The signal was visible in step 6's own position refinement at bin 2 all along:
`y = 2.15 ± 2.52 px (max 15.80)` against `x = 0.35 ± 0.70` — twenty times more
scatter in y than in x, the signature of a vertical error with no term in the
shift model.

`estimate_drift_proj.py` measures it where the angle does *not* cancel: from
`<pfile>_proj.h5:/exchange/proj_bin2`, step 5's stitched, shift-corrected,
Paganin projections, all 12000 of them. Those are already in the object frame
(`nobj = n = 4096`, so `proj_bin2` is 1024² and `_stitch` has taken
`cshifts_final` out), and a residual shift between two frames is motion the
model missed.

**Vertical: the sample slides down 28.5 bin-2 px = 114 raw px = 0.51 µm.** Flat
to about 30°, then a steady slide that begins near 60–70° and runs to +25 bin-2
px at 180°. See `motion_drift_proj.png`. Two independent landmarks agree with
the profile match to about a pixel:

| | over 0 → 180°, bin-2 px |
|---|---|
| profile match against θ=0 (quality 0.993) | **+25.1** |
| upper half-maximum edge of the dark band | +28.7 |
| lower half-maximum edge | +28 |
| incremental track, lags 0.15/0.30/0.60° | +27.5 |

The axis is vertical, so rotation moves features **horizontally only**: `dy` is
pure drift whatever the angular separation. `dx` is not — a centroid off the
axis at (a, b) genuinely projects to `a cos θ + b sin θ`, which the
reconstruction already models — so the horizontal drift is the residual after
that term is fitted out, and the fitted term itself (+34 cos, +33 sin bin-2 px)
is never written.

Applied on 2026-09-04: `--zero-x --out correct_motion.txt` added the vertical
column, mean removed, running −28.8 px at θ=0 to +71.7 px at θ=180. The
horizontal column was then reset to the commanded displacement exactly, so
**`correct_motion.txt` now carries zero x drift** (it had 0.81 px p-p left over
from the retake-only file; see "Horizontal" below for why that is discarded
rather than kept). Backups beside it: `correct_motion.txt.bak_retakeonly`
(retake-only, both columns) and `correct_motion.txt.bak_xnonzero` (this file
before x was zeroed). The
sign is fixed twice over — by `estimate_motion.py`'s convention (its correlation
returns retake − scan frame and it writes `-dy`, so a positive col 1 means the
sample sat lower) and by `_stitch`'s own padding, where a positive `r[j,k,0]`
leaves the bottom `r` rows invalid, i.e. moves the frame up.

**Horizontal: left at zero, deliberately.** Two honest estimators — the
unclipped first moment (Helgason–Ludwig) and 1-D profile matching — disagree by
~30 bin-2 px on the residual, which means what is left after the
`a + b cos θ + c sin θ` fit is not a rigid translation and neither number is
measuring one thing. The stakes are capped anyway: injecting a 10 px monotone
x ramp and refitting leaves only **1.87 px** of residual, i.e. a smooth x drift
is 81 % absorbed by the shift model the reconstruction already has. There is no
28-px-class error hiding in x the way there was in y, because the axis is
vertical and y has no such escape. The 0.81 px p-p that the retake-only file
carried has been reset to zero.

That was a decision taken under uncertainty; it has since been **measured**, and
zero is right — see [Horizontal
drift](#horizontal-drift--measured-on-purpose-and-it-is-zero) below, which puts
the end-to-end x motion under 1 raw px where y moves 117.

### Cross-check: the same drift measured from the raw holograms

`rawtrack.py` measures the drift a third way, with **steps 3–5 taken entirely
out of the loop** — no stitching, no shift model, no Paganin, no reconstruction
grid. It reads raw EDF frames, divides by the flat, forms the vertical profile
(a column mean) and NCC-matches every frame against frame 0. The rotation axis
is vertical, so the vertical profile is rotation-invariant: any frame can be
matched against frame 0 regardless of ω. The search is **centred on the
commanded displacement** and scans only ±170 px of residual — a span smaller
than the ±300 px displacement locks onto discrete repeat peaks and returns
nonsense.

51 frames, every 250th plus the two retakes. Against the correction now
installed, both curves referenced to frame 0:

| | raw − installed |
|---|---|
| all 51 points | mean +2.98, rms 20.4 px |
| excluding the 9 static-lock rows | **mean +5.3, rms 8.3 px** |

on a curve with **100 px** of total travel. Two measurements that share no code,
no intermediate product and no assumption agree to 8 % of the signal. See
`drift_two_methods.png`.

The retake pair, measured this way, gives **+118.2 px** (ω = 0, frame 0 vs
12002) — the direct, model-free number, and it agrees with the +100 px the
projection curve reaches at θ = 180 to within the same tolerance. The ω = 90
retake (12001) reads +87.3, ~30 px lower than 12002 taken seconds later; the
sample is wider than the 18.4 µm FOV, so the horizontal
truncation is ω-dependent and the vertical profile is not *exactly*
rotation-invariant. 12002 is the one to quote, because it repeats frame 0 at
the same ω.

**The static-component trap has a signature: anomalously high correlation
quality.** 9 of the 51 rows returned `measured ≈ 0` regardless of what
displacement had been commanded — including rows where the sample had been
moved 108 and 125 px. Those 9 rows carry qualities 0.54–0.94 against a median
of 0.47 for the rest. The correlator had locked onto the stationary
illumination/detector background, which is a *perfect* match and therefore
scores highest. **High quality here means the measurement failed.** This is the
same failure that produced `estimate_motion.py`'s 0.15 px.

### Why the obvious method fails, and how a self-test can lie

**Phase correlation does not work on these frames, and fails silently.** Two
independent reasons, both established by injecting a known shift into the **raw**
frame — the only honest test, because a real drift moves the sample and leaves
the static background where it is.

**1. A static component eats the correlation.** `proj_bin` is dominated by
something identical at every angle — Paganin halo, illumination that survived
the flat division, the mask edge; the angle-averaged image has **2.5× the std of
the angle-varying part**. Phase-correlate two raw frames and the peak locks onto
that and sits at zero whatever the sample does. Injecting 25 px moves the answer
by 0.12 px.

**2. Subtracting a template does not fix it — and its self-test lies.** The
template `t = mean over angles` is `S + mean(V)`, so `a − t` and `b − t` both
contain `−mean(V)`: a *different* common static image, and the peak pins at zero
just as hard. Injecting 25 px into the raw frame and then subtracting the
template moves the answer by **0.005 px**. Injecting it *after* the subtraction
gives **24.999** — because that shifts the common term along with the sample,
which reality does not do. An earlier version of this measurement passed exactly
that test and concluded there was no drift. There was 114 px of it.

```
injected into the raw frame      0.5     2.0     5.0    25.0 px
shift, then subtract template  +0.002  -0.005  -0.002  +0.005   <- honest, blind
subtract template, then shift  +0.494  +2.000  +4.998 +24.999   <- the flawed test
no template at all             -0.006     --   -0.063  +0.124
```

Windowing onto the sample does not rescue it either (tested at five windows,
including one where the varying part is stronger than the static one), and
without a template the whitened peak between frames 3° apart is only 0.03: the
shared structure is low-frequency and **whitening throws it away**.

**What works is not correlating at all in 2-D.** Collapse each frame to a 1-D
profile — column-mean over the central half for the vertical — and match the
profiles with a plain, *unwhitened* normalised cross-correlation. Low
frequencies then carry their natural weight and the match quality is 0.99 across
the full 180°. The two half-maximum edges of the dark band are correlation-free
and confirm it.

`check_proj_align.py` and `estimate_motion_proj.py` are both superseded. Their
sub-0.02 px residuals on this dataset are artefacts of trap 1 and trap 2
respectively and should not be quoted.

`<pfile>_srdata.h5` cannot be used for any of this: steps15 saves only
`min(20, ntheta)` angles there, so it never reaches 180°.

### Horizontal drift — measured on purpose, and it is zero

The vertical drift is 114 px and corrected. The horizontal column of
`correct_motion.txt` is identically zero, and the question is whether that is a
measurement or an omission. It is a measurement:
`python estimate_retake_drift.py config_steps15.conf`.

| pair | span | dx | dy (control) |
|---|---|---|---|
| 0 → 12002 | whole scan | **−0.5 ± 1.0 px** | −117.6 |
| 6000 → 12001 | second half | **+0.1 ± 0.4 px** | −58.4 |

Medians over nine crop/band combinations each (crops 1536…3072, bands 0.4…0.8),
with the static template subtracted. **Horizontal drift is under a pixel where
the vertical drift is 117**, so the zero column is right, and the axis measured
from the end-of-scan retakes is the axis for the whole scan.

Two things had to be ruled out before that could be said.

**The first moment cannot see it.** Helgason–Ludwig gives the centroid of a
projection as `a + b cosθ + c sinθ`, so the residual after that fit is the
drift, and `a` is the rotation axis — which would answer both questions from the
Paganin projections alone. It does not work here. The integrated mass varies
**20 %** with angle (the ±300 px displacement leaves a per-frame ramp-padded
margin, so the object is not inside one fixed always-valid window and the
baseline is neither flat nor left/right symmetric), and the same estimator run
*vertically*, where the drift after correction is known to be ~1 bin px, returns
**3.1 bin px rms**. That is the estimator's own artefact level and it is as large
as the horizontal signal: the 3.75 bin px horizontal residual it reports is
noise. It agrees with an incremental profile track at slope 1.07, r = 0.69, but
both run on the same profiles, so that agreement is common-mode.

**The retake correlation has to be read carefully.** Frames 0, 6000 and both
retakes are the only frames in the scan commanded to *zero* displacement —
everything else is somewhere in ±300 px. So in a retake/twin pair the
detector-fixed residual that survives flat-fielding is aligned at zero lag in
both frames and puts a correlation spike exactly where a small drift would show
up. It is not a small term: correlating horizontal profiles of frames at
unrelated angles in the *detector* frame gives cc ≈ 0.42, and the mean profile
carries ~50 % of the profile variance.

That 50 % is also what makes the answer quantitative rather than hopeless. A
static component of weight `w` drags a true shift `D` to about `(1−w)·D`, so a
raw reading of −0.45 px cannot be hiding tens of pixels — a −40 px drift would
have read as roughly −20. Subtracting a template built from 32 frames spread
over the scan (their random displacements blur the sample out of it) then gives
the numbers in the table. The vertical control passes at every step: −117.6 and
−58.4 px against −114 px from projection space.

**The neighbour cross-check, and why it fails horizontally.** Comparing a retake
against frames `j0 ± k` instead of against `j0` itself is the same trick
`rawtrack.py` uses: those frames carry the full random displacement, so their
detector-fixed component sits hundreds of pixels away and can be vetoed while
the sample stays near zero lag. Vertically it works — `−102.05 px` against the
motion file's `−100.37`. Horizontally it returns −39 px for one retake and +21
for the other, with a per-neighbour spread of 20–27 px: opposite signs from the
same drift, so it is measuring the sample-position-dependent illumination
residual, not the sample. The mirror-pair route fails the same way — the scan
pair (0, 12000) minus the retake pair would give the drift directly, but the
scan pair's own bands scatter over ±60 px.

**One loose end in the vertical.** The retakes put half the vertical drift in
each half of the scan (−58.4 of −117.6), while `correct_motion.txt` back-loads
it (+73.6 of +100.4). The endpoint magnitudes differ by 17 % too. The vertical
correction is applied and works, but its *shape* is worth revisiting before the
bin-0 level.

### Horizontal *and* vertical, a fifth way — through the volume, not through a pair

Every estimator above compares two frames, and that is precisely why the
horizontal is hard: rotation reshapes the horizontal profile, so a pair cannot
separate a shift from a shape change. The way around it is to compare each
projection against *all the others at once*, through the reconstruction.
[`estimate_xshift_ls.py`](estimate_xshift_ls.py) does that as a least-squares
problem rather than as a correlator,

    min over u, rx, rz   ½ ‖ W ( S_r ψ − R u ) ‖²

with ψ the bin-2 Paganin projections `steps15.py` already saved for every angle,
**u a variable of the same problem** rather than something reconstructed in an
inner loop, and r a **free shift per angle in x and z** — no polynomial, no
smoothness prior, one unknown per projection per axis. No peak search, no
whitening, no integer grid, which is what `estimate_reproj_align.py` next door
does and what this folder has repeatedly found to fail.

**Joint, not alternating, and that is the whole story.** The first version
alternated — reconstruct u from the current shifts, take an exact Gauss-Newton
step in r, repeat — and converged *linearly*, at ~0.93 per iteration, for a
structural reason: the inner u-solve **absorbs the misalignment**. Given a wrong
r it happily builds a doubled volume that explains the shifted data fairly well,
so the residual the r-step then sees is far smaller than the misalignment
warrants and the step under-corrects. An Aitken extrapolation was bolted on to
hide it. All three variables now move together under the **BH scheme of
[`rec_mpi.py`](../../src/holotomocupy/rec_mpi.py)** — one gradient, one
direction, one α from the true Hessian — so a wrong r cannot hide inside a
converged u, and there is nothing left to extrapolate.

The objective is *linear* in u, so there is no u–u and no u–r second derivative;
the only non-Gauss-Newton curvature comes from the shift, whose first and second
derivatives are the `2πik` and `(2πik)²` multipliers the spectral shift already
computes. The bilinear Hessian is therefore essentially free here. Each block is
preconditioned before the CG — u by `fbp` (≈ `(RᵀR)⁻¹Rᵀ`, at the cost of the one
backprojection it was going to do anyway), the shifts by the per-angle 2×2
Newton inverse `[[hxx,hxz],[hxz,hzz]]` — and `rho` is BH's own ρ, the relative
weight between the jointly solved blocks: 1 means each takes its own Newton
step, 0 freezes the shifts and leaves a plain CG reconstruction.

One iteration costs **one backprojection plus one forward projection**, because
`Ru` and `R·eta` are carried by linearity and `⟨eta_u, grad_u⟩` is evaluated in
sinogram space so `grad_u` is never formed. `--refresh` recomputes `Ru` from
scratch periodically against float32 drift.

**Pure minimisation.** There is no ridge, no model subspace, no penalty and no
line search: α and β both come out of BH, and the loop simply descends *f* until
`--iters` runs out. Three things that are still in it are *not* regularisation
and should not be read as such — `W` says which pixels are trusted, `mask_r`
says where the object is (both part of the measurement), and the two
preconditioners are changes of metric that cannot move the fixed point.
`--iters` is the one knob that does act like a regularizer, which is exactly why
the injection gain and not the misfit curve is what sets it.

**The blind directions differ by axis, and they are reported rather than
fitted.** Horizontally `cos θ` and `sin θ` are *exactly* blind: `b cos θ + c
sin θ` is a rigid xy translation of the object. The horizontal **constant is
not** blind — a constant detector shift puts a DC term in the first
Helgason–Ludwig moment, which is exactly what a centre sweep detects — it is
merely weakly determined, and on truncated data it trades against everything
else, so `step5_center_sweep.py` remains the tool that measures it. Vertically
only the constant is blind (translate u in z); a vertical `cos θ` is a real
wobble and perfectly visible, which is why **z is the easy axis here**. Along an
exactly blind direction the gradient is identically zero, so a pure descent
never drives one anywhere: the answer comes back up to a rigid translation, and
the components are printed every iteration next to `rotation_center_shift`.
`--strip-null` projects them out of every CG direction instead, if a fixed gauge
is wanted.

**What the self-test measured** (`--selftest --st-na 120 --st-n 160 --st-nz 64`,
injected `h=poly3:2, v=poly1:1.5`, gain against the identifiable part):

| iterations | x gain | z gain |
|---|---|---|
| 12 | 0.390 | 0.963 |
| 30 | 0.649 | 0.979 |
| 60 | 0.789 | 0.985 |
| 120 | 0.910 | 0.990 |
| **300** | **0.982** | **0.991** |

**z is finished by 30 iterations; x is still climbing at 120 and only arrives
near 300.** That is the geometry, not the solver — horizontally the near-blind
`cos`/`sin` directions leave the rest of the track poorly conditioned — and it
is why `iters=300` is the config default and why anything under ~100 should be
read as a partial answer in x. `--strip-null` changes none of this: the gain is
0.982 either way at 300, and all that moves is the leftover rigid translation
(residual 0.041 → 0.007 bin px rms).

**The halo, measured rather than assumed.** A Gaussian high-pass along x used to
be offered as a defence against the Paganin halo. `--selftest --st-halo 0.3` was
asked, and the answer was no: the gain fell from 0.69 to 0.05–0.08, because the
shift information lives in the same low frequencies the halo does. The option is
gone. The halo itself still costs about 30 % of the gain and that is a real,
uncorrected bias; `--inject` on real data cannot measure it, because injecting
into ψ drags the halo along with the sample, which is why the halo test is
synthetic.

It reads its own config, [`config_xshift.conf`](config_xshift.conf) — the
projection file, `nangles`, `iters`, `rho` and the `rotation_center_shift` step 5
already applied, and nothing that steps 1–5 need. `config_steps15.conf` cannot
serve: `parse_args_steps15` treats most of its fifty keys as required, so a short
file cannot be fed through it. Every key there is also a `--flag`, the flag wins
over the file, and a key that is not a flag is a hard error rather than a
silently ignored line. A variant is a second config:

```bash
sed 's/^iters=300/iters=600/' config_xshift.conf > config_xshift_long.conf
qsub -l select=4 -v CONFIG=config_xshift_long.conf polaris_run.sh
```

`rotation_center_shift` is in the config but is **not applied** — step 5 folds
it into the shifts before stitching ([`steps15.py`](steps15.py), `r[..., 1] +=
rotation_center_shift * scale + 0.5*(scale-1)`), so `proj_bin2` already carries
it. It is written down so the fitted constant, the residual centre error, can be
printed next to it.

Run it as the `xshift` stage (see "Running it on Polaris"). It writes
`xshift_ls.txt`, `xshift_ls.png`, and one `xshift_ls_iter###.png` per iteration
carrying both shift tracks plus the **axial and vertical mid slices** of the
volume those shifts build, so a running job can be watched from a login node —
and so a job killed by the walltime is still readable.

**Installing the answer** is a separate `--export-c3d` run, and it goes into
`{path}/{pfile}_/correct_correct3D.txt`, **not** into `correct_motion.txt`.
Step 3 sums the two —

```
shifts_final = random_shifts + rhapp_shifts + motion_shifts + correct3d_shifts
```

> **As shipped, two of those four terms are zero.** `motion_src=none`, so
> `motion_shifts` is identically zero and step 7 owns the whole drift; and
> `correct3D` is not read any more — step 7 writes
> `correct_correct3D_extra.txt`, which **step 6** applies, not step 3. The
> retake estimator that would fill `motion_shifts` exists but is not validated:
> it reads ≈ 0 wherever the true drift is below ~20 object px. See ["What the
> retake estimator actually does" in the HT_RD300
> README](../AtomiumS1_HT_RD300/README.md#what-the-retake-estimator-actually-does--why-motion_srcnone).
> The rest of this section is the research log from when both terms were live.

— so the drift measured from the retakes stays in the file it was measured into,
and this file carries only what the tomography asked for. Two columns, `x` then
`y`, `ntheta` rows, **unbinned** object-grid px (the solver multiplies its bin-`b`
answer by `2**b`). An existing `correct_correct3D.txt` is **added to**, not
replaced, and kept as a timestamped `.bak_*`: the projections the solve ran on
were stitched with it, so what comes back is the residual on top of it.

Three things are checked **before** the solve starts, so a run that cannot
install its answer says so in the first second rather than after several hours:
the file is not overwritten without `export_force`; a run with `--inject` will
not export (its track is the injected one, not a correction) unless sent
somewhere harmless with `--export-path`; and the export records which `_proj.h5`
it was solved on (name, size, mtime) and **refuses a second export from the same
projections** — that would install one correction twice. Step 3 and step 5 have
to run in between. To make a run *replace* an earlier export rather than add to
it, put the matching `.bak_*` back over the file first.

The **sign is derived, not fitted**. Step 5 stitches with
`curlySback(..., r=cshifts*scale, ...)`, whose kernel gathers the source pixel at
`(tx − npsi/2 + r)/mag + n/2`, so raising `r` moves content to *smaller* index;
`estimate_drift.py` measured the same end to end (`PROBE_R`: `r=(8,0)` moved the
binned centroid from `y=254.024` to `246.034`). Content sitting `+d` off is
therefore cancelled by adding `+d`. The solver returns `r` with
`S_r ψ(x) = ψ(x+r)` matching `R u`, i.e. the data sits `+r` from where the volume
says it should be — in *both* shift modes — so the number written is `+r·2**bin`
on **both** axes. Column 1 takes `+rz` with the same sign as column 0: what fixes
the sign is how the column enters `cshifts_final`, not what wrote the file.

Then re-run `steps15.py` with `start_step=3`. At `ndist=1` step 4 is a no-op that
still rewrites pdata, so the cheap route is
`apply_motion_delta.py --c3d-new <file> [--c3d-old <the .bak>]`, which patches
`/exchange/cshifts_final` in place after checking the h5 really was built from the
files named, followed by `start_step=5`. Either way, **re-run the estimator
afterwards**: the residual track must shrink toward zero, and if it comes back
roughly doubled the sign is inverted.

### Residual vertical offset, end to end — **+2.4 px**, and it is not a tilt

The mirror test that fixes the axis also reads the vertical, and it says the
vertical correction is not quite finished.

```
PYTHONPATH=../../src python fig_center_sweep.py config_steps15.conf \
    --axis y --range -10,10,2 --rcs -13 --fig vshift_sweep_m10_p10.png
```

The sweep has to be **differential**. The mirror is horizontal, so a vertical
shift applied to both frames moves them together and leaves the residual
untouched — a common vertical shift is exactly degenerate here, the same way a
common horizontal shift is degenerate with the axis itself. The offset is
therefore put on the 180 deg frame alone, which makes the swept value the total
relative offset. The check that this is doing what it claims is the slope: the
lag comes back at **−1.000** raw px per raw px on both stacks (the axis sweep
gives +2, doubled by the mirror), and the horizontal lag does not move at all
across the sweep (+1.81/+1.99 on srdata, +0.96/+1.04 on Paganin, dithering only
with the interpolation phase).

| stack | rms parabola vertex | lag zero crossing |
|---|---|---|
| `srdata` | **+2.24** | **+2.28** |
| Paganin 35 | **+2.83** | **+2.84** |

Unlike the horizontal case, the rms here is a well-behaved parabola and agrees
with the lag to 0.04 px, so the two metrics are not independent evidence so
much as confirmation that nothing is broken. The 0.6 px spread between the two
stacks is the real error bar: `srdata` is sharp and weights the fringes,
Paganin is low-passed and weights the bulk.

**It is not an axis tilt.** Slicing the same stitched pair into five vertical
strips at x = −1000 … +1000 raw px and fitting `dy(x)` — a tilt of α would give
`dy = 2α(x − c)`, because the mirror flips the sense of the tilt — leaves the
intercept solid and the ramp unestablished:

```
srdata      dy = +2.34 + 1.07e-3*x    ramp +4.4 px over 4096, strip scatter 0.45
paganin 35  dy = +2.38 - 0.05e-3*x    ramp -0.2 px over 4096, strip scatter 1.47
```

The intercepts agree to 0.04 px and match the global sweep, but the two stacks
disagree on the *sign* of the ramp and the Paganin scatter is larger than the
ramp it fits (its ±1000 strips both fall to ~+0.6 px, which is a field-edge
background effect, not a line). Tilt is below the noise: under about 0.03 deg,
under about 4 px across the frame.

So it is a **constant +2.4 px**. The pair is (0, 11999) — the two *ends* of the
scan — so the natural reading is 2.4 px of end-to-end vertical drift that
`correct_motion.txt` does not take out, on top of the 114 px it does. That is
consistent with the shape complaint above and is 2 % of the total drift.

**This cannot be refined into a per-projection correction from opposed pairs.**
The scan covers 180 deg, so the only opposed pairs that exist are the two
endpoints; there is no mid-scan pair to test whether the missing 2.4 px
accumulates linearly or arrives in a step. Fixing it means going back to
`estimate_drift_proj.py`, not adding a ramp on the strength of two frames.

### Axis tilt — **not detected**, bounded at about 0.04 deg

`estimate_tilt.py`, on the Paganin projections at 0.00 and 179.99 deg with the
axis and vertical corrections applied:

```
PYTHONPATH=../../src python estimate_tilt.py config_steps15.conf --scan -0.3,0.3,0.15
```

The signature is specific, and that is the point. An axis tilted by alpha
inside the detector plane makes the *mirrored* 180 deg view a **rotation** of
the 0 deg view by 2 alpha, so the residual field has to satisfy
`dy = +2a·x` **and** `dx = -2a·y`, two coefficients from one angle. A slope of
`dy` against `x` alone proves nothing — a detector skew, a stitch shear and a
magnification mismatch all produce one. So the fit is the full affine, split
into rotation, shear, dilation and stretch, and a tilt has to put everything in
the rotation.

| | value |
|---|---|
| rotation between the two views | **−0.076 ± 0.050 deg** |
| **axis tilt** | **−0.038 ± 0.025 deg**  (1.5 sigma, consistent with zero) |
| axis walk across the 4096 px frame | 2.7 raw px, 2-sigma bound 6.3 px |
| 3-sigma detectable tilt | 0.075 deg |

**Two things had to be fixed before that number meant anything.**

*Detrending.* Paganin projections carry a large smooth background, shared
between the views, that correlates at 0.99 on its own — so an undetrended NCC
scores empty blocks as well as it scores the sample, the quality gate stops
gating, and the fit is driven by whichever featureless blocks drift. Projecting
a bilinear plane out of every block and every shifted candidate fixes it. This
is not the spectral whitening the drift section warns against: a plane is three
degrees of freedom and it removes the background, not the sample.

*The gain.* Inject a known rotation before matching and the fit returns only
0.59 of it — an integer-grid NCC peak fitted with a 3-point parabola is pulled
toward the nearest integer, so sub-pixel displacements are compressed and every
slope with them. **The raw fit understates a tilt by 40 %.** The number above is
therefore a null test: sweep the injection, fit rotation against it, report the
crossing. That is exact whatever the gain is. The scan is linear to 0.0069 deg
over a 0.6 deg span, so the extrapolation is safe.

Note the crossing is the *injection* that nulls the fit; the physical rotation
is the negative of it. Both are printed for that reason.

**It is not a clean rotation, and the formal error bar is optimistic.** At zero
injection `b1 = +1.13e-3` against `-a2 = -2.45e-3` — same size, wrong sign
relation, so the field is not the rotation a tilt would make. And the fitted
rotation moves further than its error bar when the analysis window changes:
gain-corrected, it spans −0.09 … +0.19 deg over block sizes 96/128/160 and
analysed fractions 0.55/0.70/0.86. **Take ±0.05 deg on the rotation as the real
uncertainty**, not the ±0.025 the covariance reports. The conclusion is
unchanged — every configuration is consistent with zero — but a tilt of a few
raw px across the frame cannot be excluded from one pair.

**Do not tighten `--search` to clean this up.** It looks right: dropping it from
±5 to ±2 halves the apparent shear by rejecting bad locks. But it also drops the
gain from 0.59 to 0.27, because the blocks it rejects are the far-from-centre
ones with the largest displacement — exactly where the slope lives. It selects
against the signal. Use `--search 2` only to ask whether a shear survives the
bad locks.

**The shear does partly survive**: about 0.10 deg at ±5, about 0.045 deg at ±2,
so most of it was bad locks and some of it is real. A shear anti-commutes with
the mirror, so it appears doubled here; roughly 0.02 deg of skew. That is not a
tilt and is left unexplained. The way to chase it is the **retake pair
(12002, 12000)**, taken seconds apart rather than across the whole scan: a
detector or stitch skew is in both pairs equally, while anything that
accumulates over 12000 projections — the stage bending, say — is only in this
one. That needs a second extract, which has not been made.

## Shrinkage — measured, and absent

`python estimate_shrink.py config_steps15.conf --grid 5`, 25/25 blocks fitting
at every crop:

| pair | dt | crop | A_y (ppm) | A_x (ppm) |
|---|---|---|---|---|
| 12001 / 6000 | 0.50 | 2048 | +141 ± 273 | +379 ± 533 |
| | | 2560 | −120 ± 208 | +514 ± 493 |
| | | 3072 | −726 ± 657 | −1549 ± 823 |
| 12002 / 0 | 1.00 | 2048 | −478 ± 318 | +442 ± 568 |
| | | 2560 | +44 ± 261 | −565 ± 463 |
| | | 3072 | +105 ± 96 | +297 ± 366 |

Every term is inside 2σ of zero, at every crop, in both axes, and the signs are
not even consistent between crops — which is what a noise-dominated fit looks
like. At the frame edge (r = 2048 px) that is |y| ≤ 1.0 px and |x| ≤ 1.6 px.
The tool's own verdict at all three crops:

> Every scale term is within 2 sigma of zero: there is no measurable shrinkage
> in this scan. Leave the shrink model at A = B = 0 (rho[tp] = 0) — fitting it
> would be fitting noise.

There is no `shrink_list.mat` either, so `init_tp_from_shrink()` starts the model
at A = B = 0; with **`rho[tp] = 0`** at every level it stays there and the term
is absent. This is also the answer every 2026 folder in the tree reached, and
[`../Y350a_largedisp_006nm/README.md`](../Y350a_largedisp_006nm/README.md) shows the
15–24 % loss from leaving it non-zero on a ±300 px scan.

## `nobj` = 4096 — the grid is the detector, with no margin

At `ndist=1`, `norm_mag = 1` exactly, so each frame back-maps onto exactly 4096
object px and the ±300 px displacement only slides that window around inside the
grid. The grid therefore has to hold the **sample**, not the sample plus the
sweep, and `nobj = nzobj = n = 4096` at bin 0 (2048, 1024 going up the ladder).

The alternative, and what ctxl FT uses, is `4096 + 2·300 = 4696` → `4736 = 74·64`.
That covers the sweep completely but costs **1.55× the memory, the disk and the
per-iteration time**, all of it spent reconstructing a 320 px shell of air that
only the extreme angles ever see. 4096 is also a far better FFT length than
`4736 = 2⁷·37`.

What the choice costs, measured from `/exchange/cshifts_final`
(y ∈ [−300.02, +299.99], x ∈ [−300.18, +299.93]) against
`Rec._build_data_mask`'s own formula:

| | kept fraction |
|---|---|
| per axis, mean over angles | 0.962 |
| detector **area**, mean over angles | **0.926** |
| detector area, worst angle (\|r\| = 300 on both axes) | 0.858 |

Those are the outer rows and columns looking at the air outside the
reconstructed cylinder. The mask box is per (distance, angle), so it *follows*
the sample rather than taking the worst case over all angles — that per-angle
box is precisely what makes `nobj = n` affordable; the old centred-rectangle
mask would have thrown away half the detector at every angle.

So **expect the bin-2 log to report a kept fraction near 0.93, not 1.000.** What
would still be a bug is a number well below 0.86, or one that drifts strongly
with angle: at `ndist=1` that means a shift file in the wrong units.

Two consequences worth remembering: `err` in `conv.csv` scales with the kept
fraction, so it is **not** comparable with the `nobj=4736` runs; and `obj_init_*`
is nobj-sized, so changing this number means re-running step 5 and deleting the
old datasets first (they are `create_dataset`, not `require_dataset`).

## `paganin` = 35 — set by hand, not measured

**This is the one number in the folder that was not measured.** The values that
have been in play:

| value | where it comes from | Paganin blur length `sqrt(δ/β · λz/4π)` |
|---|---|---|
| 120 | Peter's octave driver for `../AtomiumL1_HT` and `../AtomiumS2` — **but at 33.35 keV** | 2.01 µm = 447 unbinned voxels |
| 32 | that value energy-scaled to 17.1 keV: away from an edge `δ ~ E⁻²` and `β ~ E⁻⁴`, so `δ/β ~ E²` and `120 · (17.1/33.35)² = 31.6` | 825 nm = 183 voxels |
| **35** | **what the configs use** (Viktor, 2026-09-11) — essentially the energy-scaled 31.6, rounded up, and the same value `../AtomiumS1_HT_RD300` carries | 862 nm = 192 voxels |
| 20 | what they used until 2026-09-11 — chosen by hand for a sharper, noisier start | 652 nm = 145 voxels |

Only the step-6 *starting point* depends on this; the iterative solve is free to
move away from it. But the starting point matters more here than in a
four-distance scan: at `ndist=1` one propagation distance cannot separate phase
from absorption, so the transport filter is doing the whole job. Look at it
before committing the ladder:

```bash
python show_slices.py config_steps15.conf --init
```

Re-running step 5 alone is cheap (`start_step=5`), so another value costs almost
nothing. If Peter's driver ever lands, it settles the question:

```bash
grep -i delta_beta <path>/<pfile>_/ht_<pfile>.m
```

`paganin` must be the same in `config_steps15.conf` and in **every**
`config_step6_*.conf` here:
step 5 writes `/exchange/obj_init_re{paganin}_{bin}` and step 6 reads that exact
name. A mismatch is a `KeyError` in step 6, not a silent fallback.

Step 5 has already been run at 35: `_obj.h5` (2026-09-07) carries
`obj_init_re35_2` and `obj_init_imag35_2`, and **nothing else** — the earlier
`re20_2` / `re32_2` did not survive the rewrite, so going back to another value
means re-running step 5 (`start_step=5`), not just editing a config. Only bin 2
is written (`start_level_rec=2`), which is all that is read: bins 1 and 0 take
their object from the previous rung's checkpoint.

### The Paganin projections are kept for every angle

`PROJ_SAVE_STEP = 1` at the top of [`steps15.py`](steps15.py) — step 5 writes
the Paganin-filtered projections themselves, not just the FBP volume, to
`<pfile>_proj.h5:/exchange/proj_bin{bin}`, one frame per angle. It used to keep
every 10th. Cost, at 12000 angles:

| bin | frame | dataset |
|---|---|---|
| 2 | 1024² | 50 GB |
| 1 | 2048² | 201 GB |
| 0 | 4096² | 805 GB |

Only bin 2 is written today (`start_level_rec=2`). Raise `PROJ_SAVE_STEP` to
thin the stack out again.

## The detector PSF — available, and off

`psf_sigma` is one Gaussian on detector **intensity**, applied after the
modulus inside the data misfit, in **binned detector pixels** — a detector
function, so it is the same at every distance and scales with the binning,
not with z. It ships as `0.0` everywhere: the PSF sweep that used to live in
this folder is gone, and only the psf = 0 arms remain. `src/holotomocupy/psf.py`
and `tests/psf/` keep the capability working; set `psf_sigma` in a config to
use it again.

## Step 7 — refining the per-angle drift

[`step7.py`](step7.py) re-projects a step-6 checkpoint and searches for the
per-angle shift that minimises the entropy of the FBP:

```bash
python step7.py config_step6_bin2.conf --dry-run      # geometry and units
python step7.py config_step6_bin2.conf                # ~20 min, one GPU
mpirun -np 4 ../../demo/bind.sh python step7.py config_step6_bin2.conf
```

Under `mpiexec`/`mpirun` the **z slices are split over the ranks** and the
256-bin histogram is allreduced, so every rank runs the same Nelder-Mead on
the same numbers and the answer does not depend on the rank count — checked
bit-for-bit on this scan, 1 rank against 4, identical `correct_correct3D_extra.txt`
and identical shifts. Outside a launcher nothing imports `mpi4py` and the old
single-GPU path is untouched. Measured 29.1 s → 8.4 s on four A100s (3.46×)
for a short ladder; what does not scale is the shift, 4% of an evaluation,
which every rank has to repeat because a vertical shift mixes z. `-n` cannot
exceed `--nslice` (64). `polaris_run.sh` runs it on 8 ranks.

The search uses **a quarter of the scan's angles**, on an even stride — the
drift is a low-order curve over 180°, so the full set oversamples it fourfold
and costs four times as much per evaluation. `--ntheta` overrides it. The
output file still carries a row for every angle in the scan, because the fit
is a polynomial and is evaluated back onto the full grid.

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
cd /eagle/APS_IRI/vnikitin/holotomocupy_gpu_reduced/experimental/AtomiumS1
source ../polaris_env.sh
qsub polaris_run.sh                                # steps15 + bin2 + bin1 + bin0
qstat -u $USER                                     # watch it
tail -f slurm-*.out                                # the job log, in this directory
```

[`polaris_run.sh`](polaris_run.sh) is **the whole chain in one job**: four
literal `mpiexec` lines -- `steps15.py`, then `step6.py` at bin 2, bin 1, bin 0
-- each ending `|| exit $?` so a failed stage stops the job instead of letting
the next level seed itself from a checkpoint that was never written. To run
only part of it, comment out the lines you do not want. The header asks for
`preemptable`, 2 nodes, 18 h; the job re-queues if it is evicted, so the
resume rules below matter.

There used to be a `check_data_read.py` pre-flight here -- it resolved the
paths step 3 would read and checked the shift files' shapes, units and bin
factors, and `polaris_run.sh` ran it before the healthcheck. It is gone, as
in the two ctxl folders: the script was never committed, so the line only
aborted the job. The checks that matter are made by `steps15.py` itself
through [`esrf_meta.py`](../../src/holotomocupy/esrf_meta.py) and appear in
the run log.

The two diagnostic stages at the bottom of the script -- the rotation-centre
sweep and the x-shift fit -- stay commented out. They are one-off measurements
and the debug queues are enough for them: comment out everything above,
then

```bash
qsub -q debug -l select=2 -l walltime=01:00:00 polaris_run.sh
```

**`step5_center_sweep.py` needs GPUs** -- it re-runs step 5's stitch + Paganin
for all 12000 angles per candidate -- so it goes through the job script, *not*
a login node the way the sibling folder's README implies. Its range is the tail
of the `mpiexec` line (`--start 3 --stop 23 --step 1`). Sweep on the grid
`rcs0 + 4k` when at bin 2, so every candidate is an exact integer bin-2 shift;
off-grid candidates gave `tv` a spurious period-4 oscillation that moved its
vertex by 5 px.

**The `xshift` stage** is the same kind of one-off: GPUs, the bin-2 projections
`steps15.py` already wrote, and nothing under `<path>/` touched. It reads
[`config_xshift.conf`](config_xshift.conf), where the defaults live -- 1000
angles (every 12th of the 12000), a degree-5 polynomial, 8 outer iterations. A
second parameter set is a second config rather than a second script, and extra
flags go on the end of the same line:

```bash
... python "${SCRIPT_DIR}/estimate_xshift_ls.py" "${SCRIPT_DIR}/config_xshift_long.conf"
... python "${SCRIPT_DIR}/estimate_xshift_ls.py" "${SCRIPT_DIR}/config_xshift.conf" --iters 20 --nangles 300
... python "${SCRIPT_DIR}/estimate_xshift_ls.py" "${SCRIPT_DIR}/config_xshift.conf" --inject h=poly3:2
```

Between stages, on a login node:

```bash
python score_sweep.py <path_out>/center_sweep_bin2         # score the sweep
python show_slices.py config_steps15.conf --init           # eyeball the Paganin volume
```

Queue caps, from `qstat -Qf` on 2026-09-03:

| queue | nodes | walltime |
|---|---|---|
| `debug` | max 2 | 1 h |
| `debug-scaling` | max 10 | 1 h |
| `prod` | 10-496 | 24 h |
| `small` | 10-24 | 3 h |
| `medium` | 25-99 | 6 h |
| `preemptable` | 1-10 | long, evictable |

**The shift files, and where they have to sit.** Step 3 reads
`correct_motion.txt` out of the **raw scan** directory, and on an EDF flavour
that is the *distance* directory (`..._1_`), which is what `lay.dname(ref_dist)`
returns -- not `<pfile>/projections/`, which is right only for the `nxvds`
siblings. `correct_correct3D.txt` is read from `<pfile>_/` instead. Both are
Peter Cloetens' files now, both fitted on the **2x2 binned** grid, hence
`correct_motion_bin=2` and `correct3d_bin=2` in `config_steps15.conf`:

```
<path>/<pfile>_1_/correct_motion.txt        drift, ntheta+1 rows
<path>/<pfile>_/correct_correct3D.txt       tomographic residual, ntheta+1 rows
```

Step 3 logs the exact path it found each one at, so a file that
landed one directory away shows as a `MISSING -> zeros` line rather than a
silent zero column. There is no `rhapp.mat`: it registers distance planes
against each other and there is only one plane, so step 3's *"rhapp.mat not
found, using zeros"* is expected here.

**What is and is not corrected.** Shifts: random displacement (±300 px, from
`<pfile>/projections/<pfile>_0001.txt`) plus `correct_motion.txt` (drift) plus
`correct_correct3D.txt`. At `ndist=1` the motion file *telescopes* — it replaces
the random displacement rather than adding to it, so it carries displacement +
drift in raw detector pixels. There is no `rhapp.mat`: it registers distance
planes against each other and there is only one plane, so step 3's *"rhapp.mat
not found, using zeros"* is expected. Shrinkage is **not** corrected,
`rho[tp]=0` at every level — measured from this scan's own retakes as consistent
with zero.

If a stage runs out of walltime, resume it: **steps 1–5** by raising
`start_step` in `config_steps15.conf` to the first stage that did not finish;
**step 6** by raising `start_iter` to the last checkpoint written (they are
every 32 iterations, and `checkpoint_step=32` divides every `start_iter` and
every `niter−1`, so the handoff checkpoints are guaranteed to exist).

The three step-6 levels share one `path_out`, so each seeds from the previous
level's checkpoint via `start_iter`, and `Reader.read_checkpoint` upsamples
obj/prb/pos onto the finer grid. Iteration numbering is cumulative:

| config | bin | n | nobj | start_iter | niter | nchunk | `lam_laplacian` | `rho` |
|---|---|---|---|---|---|---|---|---|
| `config_step6_bin2.conf` | 2 (4×4) | 1024 | 1024 | 0 | 1025 | 32 | 5e−5 | 1.0, 0.00625, 0.32, 0 |
| `config_step6_bin1.conf` | 1 (2×2) | 2048 | 2048 | 1024 | 1281 | 8 | 2.5e−5 | 1.0, 0.00625, 0.32, 0 |
| `config_step6_bin0.conf` | 0 (1×1) | 4096 | 4096 | 1280 | 1537 | 4 | 0 | 1.0, 0.00625, 0.32, 0 |

`rho = 1.0, 0.00625, 0.32, 0` — set by hand (Viktor, 2026-09-11), the same
vector at every rung and in every config in this folder (baselines, the twelve
sweep arms, and the energy scan). `rho[prb]` is **halved** and `rho[pos]`
**doubled** against the inherited `1.0, 0.0125, 0.16, 0`, which came down the
chain `../Y350a_HT → ../ctxl_HT_4K_RD300_007p5nm → ../ctxl_FT_4K_RD300_007p5nm`
and was tuned against the **amplitude** misfit. Step 6 now minimises an
*intensity* misfit, `F0 = 1/N Σ W (K|x|² − d)²`, whose gradient and curvature
scales are not the old functional's, so the inherited vector had no standing
here in any case. `rho[tp] = 0` because the shrinkage is measured as absent.

`estimate_rho=False` at every rung, in the baselines and in the sweep alike.
Turning it on at bin 2 (`rho_estimate_niter=16`, +96 iterations) is the cheap way
to check a `rho` inherited from a different sample, and its winners are **logged,
not checkpointed** — but across the sweep it would let the arms differ in their
step sizes as well as in `psf_sigma`, and then a difference in the bin-0 volumes
would no longer be attributable to the blur model.

`start_level_rec=2` in `config_steps15.conf`: step 5 writes the Paganin+FBP
initial volume at the **coarsest** level only, which is the only one bin 2 needs.

**Walltime.** Steps 1–5 are now **measured**: job 7590111, 10 nodes / 40 ranks,
**22 min 47 s** wall, exit 0.

| step | | time |
|---|---|---|
| 1 | EDF → HDF5 (12003 × 4096², 378 GB read) | 2 min 07 |
| 2 | preprocess | 8 min 17 |
| 3 | combine shifts | 24 s |
| 4 | binned/stitched projections, all 3 levels | 10 min 21 |
| 5 | Paganin + FBP, bin 2 only | 30 s |

The `sweep` stage is measured too: **24 s** for 21 candidates, **35 s** for 41
(job 7590148 / 7590168) — the shift-independent half of the stitch is cached, so
each extra candidate costs 0.8 s.

Those 22 min were on 10 nodes; `polaris_run.sh` now asks for **2 nodes /
8 ranks** in `preemptable`, so scale accordingly — steps 1–5 should land near
1 h 30, and `start_step=3` skips the first two of those steps entirely.

Step 6 has **not** been timed here. Against the measured `../Y350a_HT` ladder
(4 distances, ntheta=4000, n=4096, nobj=4608; 13 min + 22 min + 1 h 47 min on
2 nodes / 8 ranks), this scan is 3× the angles but ¼ the distances, ≈0.8×
overall → ≈2 h 20 min at the same node count. The whole chain should therefore
fit the 18 h walltime with room to spare; the risk in `preemptable` is
**eviction**, not the clock, so plan on resuming by raising `start_iter`.

**Disk** — measured after job 7590111, in the steps15 `path_out`:

| file | dataset | size |
|---|---|---|
| `<pfile>.h5` | `data0` (12000 × 4096² × 2 B) | 402 GB |
| | `pdata0` (× 4 B) | 805 GB |
| | `pdata0_0` (× 4 B) | 805 GB |
| | `pdata0_1` (2048²) | 201 GB |
| | `pdata0_2` (1024²) | 50 GB |
| | darks + flats | 2 GB |
| | **total** | **2.27 TB** |
| `<pfile>_obj.h5` | 1024³ × 4 B × 2 — **bin 2 only** | 8.6 GB |
| `<pfile>_proj.h5` | 12000 × 1024² × 4 B, every angle | 50 GB |
| `<pfile>_srdata.h5` | | 0.1 GB |

**≈2.1 TiB**, against 227 TB free on eagle as of 2026-09-03.

Two things worth knowing before sizing a future run: `pdata0` and `pdata0_0`
**both** exist and are 805 GB each, so the padded projection data dominates the
file — not the raw frames. And `<pfile>_obj.h5` is only 8.6 GB here *because*
`start_level_rec=2` writes the FBP volume at bin 2 alone; at `start_level_rec=0`
it would be 4096³ × 4 B × 2 = **550 GB**, and the total would be ≈2.7 TB.
Step 6's own output goes to a separate `..._rec6` and is budgeted separately.

## Files

| file | what |
|---|---|
| [`esrf_layout.py`](esrf_layout.py) | the only place that knows bliss from ewoks from nxvds; filenames, geometry. **Carries a fourth flavour `edfinfo`** added here — EDF plus a `.info` sidecar and no NXtomo, the state this scan was in mid-copy. Unused now that the NXtomo has landed (`ewoks` wins), kept as a fallback for aborted scans |
| [`config_steps15.conf`](config_steps15.conf) | steps 1–5 |
| [`config_xshift.conf`](config_xshift.conf) | the `xshift` stage — projection file, `nangles`, `iters`, `rho`, `rotation_center_shift`. Deliberately not `config_steps15.conf`: a different and much shorter list of numbers, and `parse_args_steps15` cannot take a short file |
| [`config_step6_bin{2,1,0}.conf`](config_step6_bin2.conf) | the BH ladder |
| [`correct_motion.txt`](correct_motion.txt) | **measured here** — random displacement + drift, 12003×2; installed in `<pfile>_1_/`, *not* `projections/` |
| [`polaris_run.sh`](polaris_run.sh) | PBS job, one stage per submission; one literal `mpiexec` line per stage, all but one commented out |
| [`show_geometry.py`](show_geometry.py) | derived geometry and the per-level config blocks |
| [`scan_overview.py`](scan_overview.py) | scan overview figure |
| [`estimate_center.py`](estimate_center.py) | rotation centre from opposed projections — `--retakes` / `--search` / `--motion` |
| [`estimate_motion.py`](estimate_motion.py) | **superseded, do not quote its numbers.** Retake-vs-twin correlation with no static template and no zero-lag veto, so on this scan it locks onto the flat-field residual and reports the drift as horizontal when it is vertical. Still the writer of the `correct_motion.txt` skeleton (displacement + drift columns) |
| [`estimate_retake_drift.py`](estimate_retake_drift.py) | **the horizontal drift**, and the answer is zero — retakes again, but with a 32-frame static template removed and the commanded-displacement lag vetoed, plus the neighbour cross-check and the first-moment null test that show why the easy routes fail. `--neighbours` for the cross-check |
| [`estimate_drift_proj.py`](estimate_drift_proj.py) | **drift from the projections themselves**, where the angle does not cancel — the measurement the retakes cannot give, and the one that found the 114 px. Unwhitened 1-D profile matching, not phase correlation; `--out` folds the result into `correct_motion.txt`. Writes `drift_proj.txt` and `motion_drift_proj.png` |
| [`estimate_xshift_ls.py`](estimate_xshift_ls.py) | **both axes, through the volume** — `min ‖W(S_r ψ − R u)‖²` over the volume and a free per-angle shift in x and z, all of them solved *jointly* by BH (`rec_mpi.py`'s scheme), no polynomial and no regularisation. `--selftest` / `--inject h=poly3:2,v=poly1:1.5` / `--st-halo` are the honesty checks; writes a checkpoint page with both tracks and axial and vertical slices every iteration. `--export-c3d` installs the answer as `correct_correct3D.txt` (a separate summand from `correct_motion.txt`, sign derived from the stitch kernel, added to whatever is already there). The `xshift` stage of `polaris_run.sh` |
| [`estimate_reproj_align.py`](estimate_reproj_align.py) | the same idea with a bounded 2-D NCC peak search instead of a solver, and it also does the vertical. Written first; `estimate_xshift_ls.py` supplies the shared reader, phantom, null-space bases and injection shapes by importing from it |
| [`apply_motion_delta.py`](apply_motion_delta.py) | folds a changed `correct_motion.txt` **or `correct_correct3D.txt`** (`--c3d-new` / `--c3d-old`) into `cshifts_final` in place, so step 5 can be re-run alone instead of re-running step 4's ~1 TB of binning for nothing (valid at ndist=1, and it checks the h5 was built from the files named before writing) |
| [`estimate_motion_proj.py`](estimate_motion_proj.py) | **superseded, do not quote its numbers.** Phase correlation with a template; its `--inject` self-test shifts the template along with the sample, so it passes while being blind (see the drift section) |
| [`check_proj_align.py`](check_proj_align.py) | **superseded, do not quote its numbers.** Correlates raw frames, so it locks onto the static component |
| [`rawtrack.py`](rawtrack.py) | **third, independent drift measurement** — vertical NCC on raw holograms, steps 3–5 bypassed; NCC search centred on the commanded displacement. Writes `rawtrack.txt`. Run on Polaris (reads 51 × 33 MB EDF) |
| [`rawtrack.txt`](rawtrack.txt) | its output: `frame theta applied_dy measured_dy quality drift` |
| [`fig_drift_two_methods.py`](fig_drift_two_methods.py) | plots `rawtrack.txt` against the installed `correct_motion.txt`; writes `drift_two_methods.png` |
| [`estimate_shrink.py`](estimate_shrink.py) | shrinkage from the post-scan retakes |
| [`fig_retake_diff.py`](fig_retake_diff.py) | each retake against its counterpart, and the difference |
| [`step5_center_sweep.py`](step5_center_sweep.py) | centre refinement from the FBP volume — GPU, run as the `sweep` stage |
| [`mirror_center_proj.py`](mirror_center_proj.py) | **the rotation centre on the corrected projections, and the number to quote** — opposed-pair mirror correlation on step 5's stitched Paganin frames. No Paganin low-pass in the loop, no focus metric, no baseline. `--crop` is the stability knob (scatter falls as it grows); the `|dy|` gate doubles as the check that `correct_motion.txt` removed the drift |
| [`sweep_center_registered.py`](sweep_center_registered.py) | the focus sweep with its two artefacts removed — registers each candidate back onto `d=0` before scoring (a 180° axis error translates the slice by `0.64 d`), and wants candidates on a constant-fractional-`d` grid (`rcs₀ − 4k`) or the metrics track Fourier-interpolation smoothing instead of the axis. Averages over rows; per-row vertices are the tilt check |
| [`estimate_center_com.py`](estimate_center_com.py) | **superseded, do not quote its number.** First moment of the sinogram; biased +25 px by this object's Paganin background. Kept for `pipeline_axis_column`, which measures the pipeline FBP's axis column with a delta through `Tomo.R` (**512.0000**, exactly) |
| [`arbitrate_center.py`](arbitrate_center.py) | reconstructs the two competing answers side by side and crops onto a ring — the one look that killed the first-moment number |
| [`diag_sino.py`](diag_sino.py) | truncation, per-angle mass and the harmonic content of the moment residual; run this before believing any moment-based centre |
| [`run_gpu.sh`](run_gpu.sh) | runs any of these on a tomo5 GPU — a non-interactive ssh skips the `module` lines in `.bashrc`, and the conda env has an older `holotomocupy` than this repo, so `PYTHONPATH` must point at `src/` |
| [`score_sweep.py`](score_sweep.py) | scores a sweep directory with three focus metrics and fits the peak |
| [`inspect_sweep.py`](inspect_sweep.py) | first/last/difference of a sweep — use when `score_sweep.py` finds no peak |
| [`show_slices.py`](show_slices.py) | the step-5 initial volume |
| [`show_iter.py`](show_iter.py) | one step-6 checkpoint: slices, probe, positions, convergence |
| [`steps15.py`](steps15.py) | steps 1–5 driver |
| [`step6.py`](step6.py) | BH reconstruction driver |
| [`sync_tiff.sh`](sync_tiff.sh) | pulls the light step-6 output down from eagle to `/data3/vnikitin/ESRF/AtomiumS1` |
