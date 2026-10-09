# ctxl — cortex tissue, 4-distance HT, 7.5 nm voxels

ESRF ID16A, proposal **ihls3888**, scan taken **2026-08-31 08:23**.
Cortex tissue, four propagation distances, 4000 projections over 180°,
±300 px random sample displacement, 4096² detector at 17.1 keV.

Raw data: `/eagle/APS_IRI/vnikitin/20260829/ctxl/ctxl_HT_4K_RD300_007p5nm_0001*`

This folder is modelled on [`../Y350a_largedisp_006nm`](../Y350a_largedisp_006nm)
(the pipeline, the estimator scripts, the config layout) and on
[`../Y350a_HT`](../Y350a_HT) (the 4-distance step-6 parameters). It has never
been reconstructed; every number below was either measured from the raw data on
2026-08-31 or carried over from a named sibling and marked as such.

> **This is the first 2026 scan in the repository, and its directory layout is
> not the 2025 one.** ESRF moved from *bliss* to *ewoks* between the beamtimes.
> All of that difference is confined to [`esrf_layout.py`](esrf_layout.py) — see
> the table in its docstring — so `steps15.py` and the estimators read both
> flavours and nothing in the raw tree has to be copied or renamed.

| | ctxl (here) | `../Y350a_HT` | `../Y350a_largedisp_006nm` |
|---|---|---|---|
| layout | **ewoks (2026)** | bliss (2025) | bliss (2025) |
| distances | 4 | 4 | 1 |
| energy | 17.1 keV | 33.35 keV | 33.35 keV |
| voxel size | **7.500 nm** | 20 nm | 6.000 nm |
| random displacement | ±300 px | none | ±300 px |
| `nobj` | **5056** | 4608 | 4736 |
| `rotation_center_shift` | **−15.77 px** (from ESRF's nabu) | −10.386 px | −37.50 px |
| `paganin` | **60** | 60 | 40 |
| `correct_motion.txt` | **yes, from ESRF** | yes | estimated locally |
| shrinkage correction | none — *measured* as absent | `rho[tp]` non-zero | none — measured as absent |

## Geometry

Read off the raw data with `python show_geometry.py config_steps15.conf`, and
cross-checked against the four `.info` sidecars, which are written by different
beamline software:

```
energy 17.1 keV   detector pixel 1.47601 um   focus->detector 1212.9965 mm
 k    z1 [mm]    z2 [mm]       mag  norm mag  voxel [nm]  prop [mm]  fringe px
 1     6.1635   1206.833   196.802   1.00000      7.5000     6.1322       88.9
 2     6.4280   1206.569   188.707   0.95886      7.8217     6.3939       87.0
 3     7.4856   1205.511   162.044   0.82338      9.1087     7.4394       80.6
 4     9.6817   1203.315   125.288   0.63662     11.7810     9.6044       70.8
```

`focus->detector` agrees to 7 digits across all four planes, and the derived
voxel sizes reproduce the `.info` `PixelSize` values exactly — which is the
independent check that `z1` was read with the right sign, since the NXtomo
stores it as a *negative* source distance.

Field of view 30.72 µm; angular step 0.045°; exposure 0.1 s; scan duration
0.44 h for all four distances; 20 flats and 20 darks per distance;
4003 EDF frames × 4 planes ≈ 500 GiB raw.

![data overview](scan_overview.png)

Regenerate with
`python scan_overview.py config_steps15.conf --sample "ctxl (cortex tissue)" --proposal "ihls3888 @ ESRF ID16A"`.

Two things in that figure are worth a second look before committing GPU hours.
The mean transmission at distance 1 climbs from 0.98 to 1.25 over the first
~500 frames and then falls back linearly to 1.00 — the flats themselves differ
by 2.6 % between the start and end of the scan, so this is illumination drift
that the 20-flat average does not track. Step 2's flat-field correction is
what it is; the residual shows up as the vertical striping visible in every
projection panel, and it is the probe's job in step 6 to absorb it. And the
sample is a low-contrast one: the only high-contrast feature in the frame is a
near-vertical edge, everything else is fringe-level texture. That is why the
rotation-centre measurement below is as weak as it is.

## Shifts

Step 3 combines four sources. Three of the four are present:

| source | file | present? |
|---|---|---|
| random displacement | `<pfile>/projections/<pfile>_000k.txt` | **yes**, 4003 rows per plane |
| inter-plane (RHAPP) | `<pfile>_/rhapp.mat` | **yes**, 2 × 4 × 4003 — *in 2×2-binned px, see below* |
| slow drift | `<pfile>_2_/correct_motion.txt` | **yes**, from ESRF |
| 3-D tomographic | `<pfile>_/correct_correct3D.txt` | no → step 3 uses zeros |

The commanded displacement is a full ±300 detector px at every plane, redrawn
every frame — consecutive frames jump the whole range, which is the point of
the scheme. In the object frame that is ±300 px at plane 1 and ±471 px at
plane 4 (300/0.63662), i.e. ±3.53 µm of sample motion. `nobj` is sized for it;
see below.

**RHAPP is in binned pixels, and step 3 now scales for it.** Peter's driver
`ht_<pfile>.m` sets `bin_factor=2`, so his pipeline measured the inter-plane
registration on a 2×2-binned detector grid and `rhapp.mat` is in *those*
pixels. Every other shift source here is in raw detector pixels. Step 3
therefore multiplies rhapp by `rhapp_bin` (config, pinned to 2; `0` reads
`bin_factor` out of the driver automatically) before summing the four sources.

This scan is the first place it matters. `holotomo_slave.m` defaults
`bin_factor` to 1, every other driver in the tree leaves it unset, and the
2025 scans' rhapp offsets are small enough to hide the error anyway
(`AtomiumL1_HT` `[0, 9.6, 3.7, 6.3]` px, `YY037A` `[0, −3.3, −5.8, −18.6]` px).
This scan's are `[0, 41.9, 81.4, 183.6]` px unscaled — so a factor of two is
92 px of misregistration at plane 4, and the 6-distance `20260516/ctxl` scan
reaches 316 px.

It was measured two independent ways before the code was changed:

* **From the data, with no shift file involved.** Resample the four planes of
  one projection onto a common object grid and cross-correlate *adjacent*
  pairs — adjacent because planes 1 and 4 are 1.57× apart in magnification and
  their Fresnel fringes no longer match well enough to correlate. Input is
  `/exchange/pdata{k}_1`, which is flat-corrected and amplitude-matched but
  **unshifted**. Over 24 angles, the residual left after removing the known
  random displacement is **1.79 / 1.83 / 1.98 ×** the rhapp increment for pairs
  1–2 / 2–3 / 3–4 (robust median; MAD 0.22 on the two well-conditioned pairs),
  and the correlation peak sits at the ×2 prediction rather than the ×1
  prediction in **69 of 72** angle-pairs. The three pairs have very different
  magnification ratios (0.959 / 0.859 / 0.773) and very different rhapp
  increments (22 / 18 / 50 px), so a resampling-centre artefact — which would
  scale with `1 − ratio` — cannot produce this.
* **From the reconstruction.** Step 6 at bin 2 with `rho[pos]` free walked the
  plane-2 positions from the input spacing out to **2.07×** it over 288
  iterations and then plateaued (2.019 → 2.063 → 2.071 over the last three
  checkpoints), with planes 3 and 4 following in proportion to how far each had
  to travel. With rhapp scaled the input offsets become
  `[0, 86.7, 163.2, 365.4]` px, and 86.7 is exactly where plane 2 stopped.

The second one is also the cost of *not* fixing it: plane 4 was moving 9.4 px
per 32 iterations and needed another ~130, i.e. ~900 more iterations than the
512 bin 2 has, so it would have handed bin 1 a geometry that was still ~120 px
wrong.

**The drift, and one disagreement worth knowing about.** Unlike
`../Y350a_largedisp_006nm`, ESRF supplied `correct_motion.txt` for this scan, so
nothing had to be fitted. It is the plane-2 random displacement plus a drift, in
exactly the units and column order step 3 assumes. Amplitude over the whole
scan: **2.11 px ptp in x, 1.62 px ptp in y**. It is analytic — consecutive rows
differ by |2nd difference| < 1e-4 px, so ESRF fitted a smooth function rather
than following the retakes frame by frame — but it is *not* a straight line;
taking a linear ramp out leaves a clearly curved residual in both columns.
Their fit source is `quali.mat`:

```
corr_imagesafterscan = [0.769 -2.182;  3.207 -1.429;  0 0]   (bin-2 px)
rot_positions        = [0 90 180],  index [0 2000 4000]
```

[`estimate_motion.py --validate`](estimate_motion.py) re-measures the same drift
independently, by correlating the three post-scan retakes against the scan
frames they repeat. It **confirms the vertical and contradicts the horizontal**:

| column | measured from the retakes | ESRF's file |
|---|---|---|
| y (vertical) | 1.572 px ptp | 1.620 px ptp |
| x (horizontal) | **0.080 ± 0.053 px ptp** | **2.109 px ptp** |

Agreement to 3 % in y is a real cross-validation of both. The x column of
`correct_motion.txt` therefore carries something besides sample drift — most
plausibly ESRF's own axis correction, which their pipeline folds into the same
file. **We use their file verbatim**, exactly as their pipeline does, and the
rotation centre below was measured *with it installed*, so the two are
consistent. Do not mix one with the other. A purely retake-derived alternative
is written to `./correct_motion.txt` by the `--validate` run if a reconstruction
ever shows horizontal streaking that points back here; step 3 would pick it up
only if it were copied into `<pfile>_2_/`.

![drift](motion_estimate.png)

## Shrinkage: off, and measured to be off

`rho[tp]=0` at every level. There is no `shrink_list.mat`, so
`load_shrink_from_mats` starts the linear model at A=B=0 regardless — but the
reason it is also *frozen* there is a measurement, not a default.

[`estimate_shrink.py`](estimate_shrink.py) reads it off the scan's own post-scan
retakes. A sample that shrank makes the retake a **scaled** copy of the frame it
repeats, so the residual displacement field between them stops being a constant
translation and acquires a slope; block-wise correlation plus an affine fit
reads that slope off directly.

| pair | spans | A_y | A_x |
|---|---|---|---|
| 4002 (ω=0) vs frame 0 | whole scan | **+107 ± 210 ppm** | +3 ± 141 ppm |
| 4001 (ω=90) vs frame 2000 | second half | +54 ± 294 ppm | −30 ± 389 ppm |

Nothing, at a 2σ limit of **\|A\| < 420 ppm** over the whole scan — 0.86 px at
the frame edge. The two pairs span different fractions of the scan, so a linear
shrink would have to show twice as much at ω=0 as at ω=90; they agree to 0.1σ.

![shrinkage](shrink_estimate.png)

In the figure the measured block displacements (blue) are ~0.5–1 px and point
in random directions, while the fitted affine part (red) is negligible: there is
no coherent radial expansion, only correlation noise.

**Re-checked 2026-09-01 on three more configurations**, because a null that
rests on one plane and one block size is worth very little. Same two retake
pairs, changing only what the fit is run on:

| configuration | A_y, ω=0 pair | A_x, ω=0 pair | A_y, ω=90 pair | A_x, ω=90 pair |
|---|---|---|---|---|
| plane 1, grid 5 *(above)* | +107 ± 210 | +3 ± 141 | +54 ± 294 | −30 ± 389 |
| plane 2, grid 5 | −77 ± 154 | −8 ± 174 | −84 ± 319 | +51 ± 400 |
| plane 1, grid 3 | +258 ± 224 | −131 ± 328 | −105 ± 454 | +90 ± 627 |
| plane 4, grid 5 | +20 ± 157 | −205 ± 175 | +94 ± 272 | +456 ± 271 |

All sixteen scale terms are within 2σ of zero and the signs are incoherent —
plane 4's x term reads +456 ppm on one pair and −205 ppm on the other, which the
script flags as *inconsistent with a linear shrink* precisely because that sign
flip is what noise does and a shrinkage cannot. Planes 1, 2 and 4 differ by
1.57× in magnification and see the sample through different propagation
distances, so a real dimensional change would have to appear in all three at the
same ppm; it appears in none.

**How much sensitivity is behind that null.** `--inject 500` rescales the retake
by a known 500 ppm before measuring. The fit responds at 7–11σ, moving from
+27/+107 ppm to **+815/+812 ppm** — so a shrinkage of this size could not have
been missed. The response is not 1:1 though: driven with synthetic frames the
same code returns ~750 ppm for an injected 500 and ~930 for an injected 1000.
That is the parabolic sub-pixel peak fit in `local_peak` inflating sub-pixel
displacements (500 ppm at a block 1200 px out is only 0.6 px). It is an
*over*-response, which makes a null result stronger rather than weaker — a
measured +107 ppm bounds the truth from above. The note above `peak_near()` in
[`estimate_shrink.py`](estimate_shrink.py) spells this out, and says what to fix
if the script is ever used to *measure* a shrinkage instead of bounding one.

For scale on what leaving `rho[tp]` free costs when there is nothing to fit: on
`../Y350a_largedisp` the optimizer invents ≈2200 ppm of edge displacement, 8×
that scan's upper limit, by absorbing displacement the position refinement
should own — [`../Y350a_largedisp_006nm/README.md`](../Y350a_largedisp_006nm/README.md)
measures the resulting 15–24 % loss. With ±300 px of random displacement, this
scan is exposed to exactly that failure mode.

## Rotation centre — measured, but weak

**`rotation_center_shift = −19.22 px`** (bin-0 detector px, convention
`c − n/2`), measured with correct_motion.txt installed:

```
python estimate_center.py config_steps15.conf --pairs 1 --motion
```

| crop | result | individual bands |
|---|---|---|
| 2048 | −19.22 ± 1.70 | −17.5, −21.6, −18.6 |
| 2560 | −20.18 ± 1.30 | −18.4, −21.6, −20.5 |
| 3072 | −18.16 ± 3.27 | −14.5, −17.6, −22.4 |

All nine bands together: mean −19.2, scatter 2.4 px. Without
`correct_motion.txt` it is −19.72 ± 1.26 — the drift barely moves it here,
unlike `../Y350a_largedisp_006nm` where the same swap was worth 2.3 px, because
this scan's horizontal drift is only 2.1 px ptp and an opposed pair sees half of
it. More bands does not help: at 5 bands the strips stop containing enough
sample and the answer falls apart (−5.4 ± 52.9).

![rotation centre](center_estimate.png)

⚠️ **Refine this before the step-6 ladder.** 4000 projections over 180° give
exactly *one* exactly-opposed pair, so there is no averaging to be had, and the
correlation peak is only about twice the noise floor (A−B RMS improves from
0.0819 to 0.0776, a 5 % gain). ±1.7 px is 14 nm at a 7.5 nm voxel — enough to
soften the reconstruction. Run steps 1–5 first, then
`python step5_center_sweep.py config_steps15.conf`, and put the winner in
`config_steps15.conf` **and** all three `config_step6_bin*.conf` before starting
bin 2.

## What the refinement did to the positions

`python show_pos_errors.py config_step6_bin0.conf -o pos_errors_total.png`
(add `--root /eagle/APS_IRI=$HOME/eagle` to run it off an sshfs mount).

![total position correction](pos_errors_total.png)

**Read the step-6 log with care: it does not print a total.** `rec_mpi.precalc`
does `self.pos_init = vars['pos'].copy()` (rec_mpi.py:836) at the entry of
*every* level, so the logged "pos abs error" restarts from zero at each ladder
boundary and is in that level's own pixels. It measures motion *within* a level.
`show_pos_errors.py` instead differences every checkpoint against the one fixed
reference the whole ladder started from — the step-3 shift sum in
`/exchange/cshifts_final` — undoing `Reader.read_pos`'s per-level mapping

    pos[...,0] = cs[...,0]/2**b
    pos[...,1] = cs[...,1]/2**b + rcs/2**b + 0.5*(1/2**b - 1)

so all three levels land on one bin-0 curve. Getting that mapping wrong shows up
at once as a ~20 px constant in x. (The level of each checkpoint is read from the
object grid it holds, 5056/2528/1264, not from `start_iter` in the configs —
those were retuned mid-run to 1024/1280 and no longer describe what ran.)

At iteration 1376, against the step-3 input:

| | Δy mean ± sd | Δx mean ± sd | max abs |
|---|---|---|---|
| plane 1 | −0.61 ± 1.07 | −1.33 ± 2.45 | 8.2 |
| plane 2 | −1.39 ± 1.14 | −2.89 ± 2.59 | 9.2 |
| plane 3 | −0.87 ± 1.20 | −3.24 ± 2.48 | 9.8 |
| plane 4 | +0.75 ± 1.30 | −4.27 ± 2.62 | 12.0 |

bin-0 **object** px; total RMS 4.29 px = **32 nm**. The unit matters and is easy
to get wrong: `pos` is a shift on the object grid, not on each plane's detector
grid — `cuda_kernels.s_kernel` computes `x = mag*(tx-(n-1)/2) - r + (npsi-1)/2`,
so `r` is in `npsi` units. All four planes therefore share **one** scale, 7.5 nm,
not the per-plane detector voxels [7.500, 7.822, 9.109, 11.781] nm that step 3
divided the commanded detector displacement by to get here in the first place.

**Almost all of it is common mode, and that is the good news.** Split the
correction into the part the four planes share and the part that differs:

    common mode   y −0.53 ± 1.01   x −2.93 ± 2.48   (x ptp 14.3 px)
    inter-plane   y rms 0.99       x rms 1.18

The common mode moves the whole detector image and says nothing about `rhapp` —
it is sample/stage motion that step 3's drift file did not carry. Only the
inter-plane part, ~1.1 px rms, is inter-plane geometry, and that is the number
the ×2 `rhapp_bin` fix bought: the shift files are now right to about a pixel
between planes, out of offsets that reach 182 px. Quoting the 4.29 px total as
if it were a shift-file error would be the same mistake that let the ×2 hide.

Two things in the common mode are worth acting on.

1. **A −2.93 px mean in x is degenerate with the rotation centre.** A constant
   horizontal shift of every projection *is* a change of axis position, so the
   refinement is telling us `rotation_center_shift` should be near **−22.15**,
   not −19.22. That is 1.7σ outside the ±1.70 px the one opposed pair could
   measure — consistent with it, but pulling in a definite direction. Fold it in
   with `step5_center_sweep.py` rather than by hand; see the section above.

2. **A smooth ±6 px excursion in x over the scan**, ~7× larger than the 2.1 px
   ptp in ESRF's `correct_motion.txt`. What that is has its own section below.

**Convergence.** Each level jumps in its first 32 iterations and then settles:
bin 2 walks 0 → 0.73 level px and is still creeping at 1024; bin 1 adds
0.46 level px over its 256 iterations and is *still rising* when it hands over;
bin 0 adds 0.11 px in 96 iterations and is flat. Only bin 0 had converged in
position when the job hit the 3 h wall at 1376 of 1700 — the argument for
trimming bin 2 and bin 1 and giving bin 0 the time, as in **Running it on
Polaris**.

## Is the sample moving, or the stage?

`python diagnose_positions.py config_step6_bin0.conf -o pos_diagnosis.png`

![what the corrections are](pos_diagnosis.png)

The section above says how big the corrections are. This says what they *are*.

**The lever is that this is four separate scans.** `entry_0000/start_time` in
`projections/<pfile>_000k.nx` gives **04:54, 05:53, 06:53, 07:57** on 2026-08-31
— four acquisitions an hour apart, spanning three hours, and each carries its
**own** random displacement sequence (the four `.txt` files differ; the sequence
is white, lag-1 autocorrelation ≈ 0). So the four planes are four independent
measurements of the same rotation, and each candidate cause lands somewhere
different:

| cause | locked to | signature |
|---|---|---|
| rotation-stage error motion | angle | one curve **all four scans share** |
| sample / thermal drift | the clock | per-scan **deviation** from that curve |
| displacement-stage error | the commanded shift | frame-to-frame, tracks that plane's own `.txt` |
| local deformation | nothing rigid | cannot appear in `pos` at all |

Smooth each plane's Δ over 151 frames and overlay the four (top row of the
figure): **pairwise correlation 0.92–0.99 in y and 0.95–0.99 in x**, with ptp in
x of 10.6 / 10.6 / 10.7 / 12.0 px. The big term repeats in scans taken an hour
apart, in the same place in the rotation each time. It is **angle-locked, not
time-locked**. The budget, in bin-0 object px (sd, total 2.49 px = 19 nm):

    shared and smooth   2.18 px = 16 nm   angle-locked -> the rotation stage
                                          y ptp 5.89 px = 44 nm
                                          x ptp 10.96 px = 82 nm
    per-scan and smooth 0.97 px =  7 nm   clock-locked -> real drift
    displacement coupling 0.60 px = 5 nm  see below
    leftover            0.39 px =  3 nm

So, to the four hypotheses:

* **Sample moving — yes, but it is the small term.** The genuinely per-scan part
  is 0.97 px sd, 7 nm, y 0.81 / x 1.10. That is consistent with what
  `estimate_motion.py` got from the post-scan retakes independently (y 1.57 px
  ptp measured, x 0.08 ± 0.05 px): y has real time drift, x essentially none.
* **Acquisition problem — yes, and it is hugely significant statistically
  though sub-pixel.** Regress the fast (unsmoothed) part on the commanded
  displacement, jointly over all four planes: the y error has a **+0.138 %**
  same-axis gain error and a **−0.234 % cross-axis** term (169σ), the x error
  **+0.111 %** and **−0.135 %** (91σ). At the ±300 px sweep that is +0.42 /
  −0.70 px in y and +0.33 / −0.41 px in x. The same-axis term is a **calibration
  error of the displacement stage** (~0.1 %); the cross term is **axis
  non-orthogonality / cross-talk** (~0.2 %) — commanding x moves y as well. Both
  are worth reporting to the beamline; neither hurts this reconstruction, since
  step 6 recovers them.
* **Local deformations — no evidence.** A deformation is not a rigid shift, so
  it cannot show up in `pos` in the first place, and nothing that depends on the
  specimen could reproduce itself to 0.99 correlation across four scans three
  hours apart. The independent shrinkage test (`estimate_shrink.py`, **Shrinkage**
  above) is also a null.
* **Rotary-stage tilt — closest, but the mechanism is error motion, not tilt.**
  A static tilt of the axis produces a height-dependent shear, which `pos`
  (one shift per projection) cannot express at all; what `pos` *can* express is
  the axis position wandering as a function of angle, i.e. **radial and axial
  error motion (runout)** of the rotation stage. That is exactly the shape of
  the shared curve: its power sits at **1–2 cycles per 180°** (x: 39 % at 1,
  32 % at 2; y: 46 % at 2, 17 % at 3), i.e. once and twice per stage revolution
  — the classic once-per-rev eccentricity plus a two-per-rev bearing term.

**The one control that matters.** Step 3 tiles the reference plane's
`correct_motion.txt` onto all four planes identically, so an error *there* would
also look common-mode. It cannot be the explanation: `motion_base` has ptp
y 1.686 / x 2.200 px against the observed 5.89 / 10.96 (3.5× and 5.0× too
small), and correlates with the recovered curve at only +0.19 (y) and −0.26 (x)
— wrong size and wrong shape (bottom row of the figure). This is the same
conclusion the retake analysis reached from the other direction.

**Caveat.** Over a 180° scan a once-per-revolution component is only half a
cycle and is nearly degenerate with a linear trend, so the split between
"1 cyc/180° runout" and "monotonic drift shared by all four scans" is not
separable from this data alone. What *is* separable, and is the load-bearing
result, is shared-across-scans versus per-scan — and 82 % of the variance is
shared.

## `correct_correct3D.txt` — what ESRF's third shift file is

Step 3 sums four shift sources, and this is the fourth:

```python
raw_3d = np.loadtxt(f'{pfile}_/correct_correct3D.txt')[:ntheta, ::-1]   # (h,v) -> (v,h)
shifts_final = random_shifts + rhapp_shifts + motion_shifts + tile(raw_3d * correct3d_bin)
```

It is tiled onto all four planes, in reference-plane detector pixels, after
scaling by `correct3d_bin`. On scans where Peter fitted it on the unbinned grid
that factor is 1; here his grid is 2×2 binned and it is 2 — see below. ESRF hands the same file to
nabu as `translation_movements_file`, whose own documentation fixes the column
order — *"each line describes the horizontal and vertical translations of the
sample… The order is 'horizontal, vertical'"* — which is why step 3 reverses it.
The companion `correct_correct3D_v.txt` is the same file with the horizontal
column zeroed.

**Peter's script is not in any copy of the toolbox we have** (it lives at ESRF,
under `/data/id16a/inhouse1/sware/pub/tomo-esrf`), but his output files identify
the model exactly. `estimate_correct3d.py --verify <file>` fits families of
increasing order and reports the first that reaches the file's own write
precision:

| file | rows | horizontal | vertical |
|---|---|---|---|
| `20250604/Y350a_HT_nobin_020nm_` | 4001 | 4 rotation harmonics, 2.7e-10 | cubic, 1.6e-09 |
| `20250604/Y350a_FT_large_rand_disp_nobin_020nm_` | 4001 | 4 rotation harmonics, 2.8e-10 | cubic, 9.6e-10 |
| `20251115/Y350a_HT_20nm_8dist_` | 4501 | 4 rotation harmonics, 2.6e-09 | cubic, 2.5e-09 |

Same model on all three, and each family fails on the other column (the
horizontal is not a polynomial at any order ≤ 10; the vertical needs ≥ 8
harmonics). So:

* **horizontal** = `a₀ + Σ_{h=1..4} aₕ cos hω + bₕ sin hω` — axis runout, which
  by construction repeats every revolution;
* **vertical** = a cubic in the projection index — sample creep, which by
  construction does not.

The split is the physics, and it is the part worth carrying over: horizontally
the stage comes back to where it was, vertically the sample does not.

The angle grid matters and is worth stating: the files have **ntheta+1** rows
spanning 0…180° *inclusive*, i.e. ωᵢ = iπ/ntheta with the last row repeating the
first projection. On that grid the horizontal column is 4 harmonics to 1e-10; on
a 0…180-exclusive grid of the same length it needs 6 and still only reaches
1e-8. Get it wrong and you approximate the model instead of reading it off.

### The correction is also inside `correct_motion.txt` — and that is consistent

**Peter has since supplied the file** (2026-09-07), and step 3 uses it; what
follows is the analysis from before it arrived, kept because it is what shows
the two files agree rather than double-count.

When this scan was first set up there was no `correct_correct3D.txt` in
`ctxl_HT_4K_RD300_007p5nm_0001_/`, and nabu's `translation_movements_file` was
empty in `naburec/nabu_final.conf` — his production config at the time. The
correction had not been skipped; it was folded into the motion file. Take
`<pfile>_2_/correct_motion.txt`, subtract the plane-2 random displacement (which
is what step 3 does), and fit the remainder — `estimate_correct3d.py
config_steps15.conf --motion` prints this:

| column | ptp | rotation harmonics, order 2 / 4 / 6 | polynomial, order 2 / 4 / 6 |
|---|---|---|---|
| horizontal | 2.109 px | 4.2e-02 / **8.3e-04** / 2.0e-05 | 3.9e-01 / 2.5e-02 / 5.5e-04 |
| vertical | 1.618 px | 6.1e-03 / 1.4e-04 / 3.4e-06 | **2.9e-07** / 2.9e-07 / 2.9e-07 |

The vertical is a **quadratic**, exact to the file's six decimals and flat
thereafter — a polynomial, as Peter writes it, one order lower than his cubic.
The horizontal is not literally a 4-harmonic fit (the residual keeps falling
with order), but it is unambiguously the harmonic family: at every order the
harmonics beat the polynomial by 10–30×, and 4 of them already account for
100.00 % of the variance. This settles the open question in
[Shifts](#shifts) — the horizontal column of `correct_motion.txt` disagrees with
the retakes (2.109 px written against 0.080 ± 0.053 px measured) because **it is
not sample drift, it is the axis correction**, carried in the motion file rather
than beside it. The vertical column, which does agree with the retakes to 3 %,
is the drift.

### What is left, and a file for it

Peter's estimate is not complete: after installing it, step 6 still had to walk
the positions by a common mode of **rms 18.6 nm horizontal / 7.6 nm vertical**
(ptp 107 / 67 nm). `estimate_correct3d.py` fits that residual and writes it in
his format, so a re-run can start from the aligned geometry instead of
travelling there:

```bash
python estimate_correct3d.py config_step6_bin0.conf \
    --root '/eagle/APS_IRI='$HOME'/eagle' --cache /tmp/poserr2.npz --write .
```

| column | model | variance explained | residual |
|---|---|---|---|
| horizontal | const + 4 rotation harmonics | 94.6 % | 0.576 px = 4.3 nm |
| vertical | const + 4 rotation harmonics | 82.0 % | 0.428 px = 3.2 nm |

The vertical is fitted with **harmonics, not Peter's cubic** (22.2 % — the
script picks per column and says which it took). That is the expected answer
rather than a contradiction: the sample drift is already gone into
`correct_motion.txt`, so what remains in both columns is angle-locked, and the
power sits at 3 cyc/rev horizontally (28.8 → 48.3 → **90.8** % at orders 1/2/3)
and 2 cyc/rev vertically (14.3 → **69.8** %). Same 3-lobed bearing runout the
[tilt tests](#not-an-axis-tilt) found.

Two things to keep straight before using it:

* **The constant is dropped on purpose.** A constant horizontal shift *is* the
  rotation centre and a constant vertical shift is only where the volume sits;
  the axis in the configs is ESRF's own (`-15.77`, from `naburec/`), fitted
  with `correct_correct3D.txt` in place, so leaving the DC out keeps the two
  consistent. `--keep-dc` overrides.
* **It goes in `<pfile>_/` on eagle, not here.** The copy in this folder is the
  estimate; step 3 reads the raw-tree path. Installing it changes `shifts_final`
  and therefore invalidates every downstream product from step 3 on.

![correct3D](correct3d.png)

## The boundary artifact, and what it is not

An axial slice of the finished volume has a comb of fine tails hanging off the
sample surface into the vacuum — 0.7–1.1 µm long, all leaning the same way,
strongest around azimuth 60–90° where they are 2× the value on the opposite
side. The interior and the far background are clean.

**It is not this pipeline.** ESRF's own reconstruction of the same scan,
`volfloat/ctxl_HT_4K_RD300_007p5nm_0001_rec_.vol` (PyHST FBP, 15 nm voxel,
3216³), has the *same* comb: same length in micrometres, same lean, same
azimuthal envelope peaking at 68°, and the fine azimuthal profiles of the two
correlate at 0.38 despite different voxels and independently estimated centres.
Two unrelated algorithms — multi-distance Paganin + FBP, and our iterative
multi-distance solver with free position refinement — put the same structure in
the same place.

![ESRF standard vs this pipeline](esrf_vs_ours.png)
![the same artifact, unrolled](edge_artifact_both.png)

What the two share is the raw frames and the shift files, so the artifact is in
one of those. Position refinement is free in our run and did not remove it,
which means no rigid per-projection shift can.

### Not an axis tilt

`tilt_check.png`, three measurements:

| test | tilt would give | measured |
|---|---|---|
| artifact vs height, z = 7.5…30 µm | ∝ \|z−z₀\|, vanishing at the pivot | 0.091 → 0.082, monotone, no null |
| vertical blur vs radius | blur 2δr, so V_z/V_xy falls with r | flat to 1.1 % out to r = 1050 px |
| 1 cycle/rev in the common mode | all of it | 6 % (y) / 25 % (x); the jump is at 3 cyc/rev |

The middle one is the direct test and it is sharp. A roll tilt δ makes each
axial slice a mixture of heights z ± δr. Calibrating against an artificial box
blur (2 voxels costs 49 % of V_z/V_xy) the 1.1 % measured over 1050 px is under
0.3 voxel, i.e.

    δ < 1.4e-4 rad = 0.008° = 0.5 arcmin.

Fed back through the taper of this pillar (dR/dz = 0.049, R = 1370 px) that
buys an edge displacement of δ·R·dR/dz = **0.01 px**. The tails are 100–150 px.

The once-per-revolution test covers the other tilt: an out-of-plane tilt is a
rigid 1 cyc/rev wobble, and 1 cyc/rev is a minority of the common mode here —
the signal is a 3-lobed bearing runout — and the rigid part is absorbed by
`pos` in any case.

![is it tilt](tilt_check.png)

## `nobj` = 5056

(This section is about the **projection** grid, 5056 at bin 0 and 5056/2^bin
below it. The object's x/y grid is half of that at every level — see
`tomo_upsample` in **The three step-6 levels** below — but everything derived
here, the margin included, is a projection-plane quantity and is unaffected.)

The grid has to hold the sample plus the whole displacement sweep:
4096 + 2 × 471 = 5038, rounded up to **5056 = 79 × 64**. It bins cleanly over
all three levels (2528, 1264), and leaves (5056 − 4096)/2 = 480 px of margin per
side against a worst-case 471 px displacement — 9 px of slack.

What it deliberately does **not** cover is the object-plane footprint of the
demagnified planes. Plane 4 has `eff_demag = 1/0.63662 = 1.571`, so its detector
back-maps onto 4096 × 1.571 = 6434 object px. Covering that would need
nobj = 7424: 1.47× linearly, 3.2× the volume. It is not worth it and not needed
— the outer ring of a demagnified plane looks at object coordinates the
reference plane never sees, so nothing there is constrained by the rest of the
data anyway. `mask_oob=1` drops those detector pixels from the fit. Expected
kept fractions, which `Rec._build_data_mask` logs at startup:

| plane | eff_demag | object footprint | kept |
|---|---|---|---|
| 1 | 1.000 | 4096 px | 1.000 |
| 2 | 1.043 | 4272 px | 1.000 |
| 3 | 1.215 | 4975 px | 0.87 – 1.00 (displacement-dependent) |
| 4 | 1.571 | 6434 px | 0.616 (flat — the box just translates) |

`../Y350a_HT` has run in exactly this situation since the start (same 0.637
magnification spread, nobj=4608 against a ~6500 px plane-4 footprint). Note
that `F0` keeps its `1/(ntheta*ndist*nz*n)` normalization, so `err` in
`conv.csv` scales with the kept fraction and is **not** comparable across runs
with different `nobj`.

## What `conv.csv` holds

One row per `error_step` iteration (plus an `iter=-1` row for the initial
state), written by `Rec.error_debug` to `{path_out}/conv.csv` and, so the levels
of a ladder do not overwrite each other, to `{path_out}/conv_bin{bin}.csv`:

| column | meaning |
|---|---|
| `iter` | cumulative iteration number; `-1` is the state before the first step |
| `err` | the functional `F0`, subject to the caveat above |
| `time` | seconds since the previous logged iteration (so `error_step` iterations' worth) |
| `dobj2` | ‖obj<sup>n+1</sup> − obj<sup>n</sup>‖², summed over the whole volume, in object units |
| `dobj_rel` | ‖obj<sup>n+1</sup> − obj<sup>n</sup>‖ / ‖obj<sup>n+1</sup>‖ |

`dobj2` and `dobj_rel` are the step the object actually took, which is what says
whether a run has stopped moving — `err` can flatten while the object still
drifts, and at a level change `err` jumps for reasons that have nothing to do
with convergence. They are `NaN` on the `iter=-1` row, no step having been taken
yet.

They cost nothing to compute. The BH step is exactly `alpha * etas['obj']`, so
no copy of the previous object is needed, and the two reductions ride inside the
object's own update pass (`Rec._apply_step_obj`), which streams both arrays
anyway. The pass is only asked for them on the iterations that get logged.
`dobj2` is scaled back up by `norm_const²` — inside BH the object carries a
`1/norm_const` — so it is in the same units as the volume that is written out;
`dobj_rel` is scale-free either way and is the one to compare across levels.

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
cd /eagle/APS_IRI/vnikitin/holotomocupy_gpu_reduced/experimental/ctxl_HT_4K_RD300_007p5nm
source ../polaris_env.sh
qsub polaris_run.sh                                   # steps15 + bin2 + bin1 + bin0
qstat -u $USER                                        # watch it
tail -f slurm-*.out                                   # or the job log
```

There used to be a `check_data_read.py` pre-flight here — it resolved the paths
step 3 would read, checked the shift files' shapes and units, the bin factors,
`ref_dist`, the drift, and `rotation_center_shift`, and both `polaris_run*.sh`
ran it before the healthcheck. It is gone: the script was never committed and
the `python ../check_data_read.py` line only aborted the job. The checks it made
that still matter are made by `steps15.py` itself, through
[`esrf_meta.py`](../../src/holotomocupy/esrf_meta.py) (next section), and they
appear in the run log.

### State as of 2026-09-07

`config_steps15.conf` has **`start_step=3`**. `<pfile>_rec/<pfile>.h5` (3.0 TB)
was written on 2026-08-31 and nothing that changed since touches steps 1–2, so
the job redoes shifts → binned data → Paganin+FBP only. Set it back to 1 to
rebuild from the EDFs.

What changed, and why those are exactly steps 3–5:

* **`correct_correct3D.txt`** — Peter's new fit. His first drop landed one
  level too deep, in `<pfile>_/<pfile>_/`, where step 3 would have read it as
  zeros; the directory has since been recopied and outer and nested are now
  **byte-identical** for every file step 3 reads. Enters at **step 3**.
* **`correct3d_bin=2`** — the file is in 2×2 binned detector px.
  `<pfile>_rec_.info` says `PixelSize = 0.015 µm` against this scan's 7.5 nm
  voxel, and `ht_<pfile>.m` sets `bin_factor = 2`. The pre-flight prints both.
* **`rotation_center_shift=-15.77`** — from ESRF's own nabu parameters rather
  than measured here (table below). Enters at **step 5**.

### Read from ESRF's metadata, not retyped

Four numbers that used to be hand-copied into the config now come out of what
ESRF itself recorded beside the scan, through
[`src/holotomocupy/esrf_meta.py`](../../src/holotomocupy/esrf_meta.py). `steps15.py`
calls it, so these statements appear in the run log:

| what | where it is recorded | value here |
|---|---|---|
| rhapp bin factor | `rhapp.mat`'s own `pixelsize` | 15.0000 nm / 7.5 nm = **2** |
| correct3D bin factor | `<pfile>_rec_.info` `PixelSize` | 15.0000 nm / 7.5 nm = **2** |
| reference plane | `reference_motion.mat` `reference_plane` | 2 (1-based) = `ref_dist=1` |
| rotation axis | `<pfile>_/naburec/*.conf` | **−15.7725** raw px |

The config still pins `rhapp_bin=2` and `correct3d_bin=2` rather than using the
`0` sentinel, so a run does not depend on those files being readable; step 3
warns if a pinned value and the recorded pixel size disagree. `find_drop_file`
looks one level deeper whenever the outer path is empty, so a repeat of the
nested drop is reported instead of silently zeroing a correction.

**`reference_motion.mat` is a check, not an input.** Its `ref_v` / `ref_h` are
the reference plane's own drift, which is exactly what step 3 reconstructs as
`correct_motion.txt / norm_mag[ref_dist] − random_shifts[ref_dist]`. They agree
to **4·10⁻⁵ object px** in both columns (ESRF stores them negated). That is a
direct validation of the magnification, the column order and the
random-displacement subtraction all at once. Its `pixelsize` is 15 nm, but that
does **not** make `correct_motion.txt` binned — the file carries the full 600 px
random-displacement amplitude, so it is in raw detector px and
`correct_motion_bin` stays 1.

The axis is **logged, never applied**. There is no knob to make nabu win, on
purpose: the axis is degenerate with the x column of `cshifts_final`, so it has
to be typed into `rotation_center_shift` in `config_steps15.conf` **and** into
all three `config_step6_bin*.conf` together — a switch that changed only step 5
would silently desynchronise it from step 6. What the run does instead is print
nabu's value and every candidate conf, and warn past 0.5 raw px. Here the two
agree to 0.003 px, so nothing fires.

### The axis, from Peter's nabu configs

`<pfile>_/naburec/` holds the configs he actually reconstructed with.
They state the axis outright, on the 3216-wide grid `<pfile>_rec_.nx` — the
2048-wide binned detector padded to `floor(4096/2·π/4)·2 = 3216` and centred in
it. nabu measures from `(N−1)/2 = 1607.5`; PyHST is 1-based, from
`(N+1)/2 = 1608.5`. Both times 2 for the binning:

| file | shift file it uses | `rotation_axis_position` | raw px |
|---|---|---|---|
| `nabu_final_cm.conf` (production) | `../correct_correct3D.txt` | 1599.613741 | **−15.7725** |
| `nabu_final_cm_odd/_even.conf` | `../correct_correct3D.txt` | 1599.613741 | −15.7725 |
| `nabu_final.conf`, `nabu_final_odd/_even.conf` | `correct.txt` | 1599.613741 | −15.7725 |
| `nabu.conf`, `nabu_correct3D.conf` | — / `../correct_correct3D.txt` | 1599.543995 | −15.9120 |
| `nabu_correct3D_v.conf` | `../correct_correct3D_v.txt` | 1608.000000 | +1.0000 |
| `<pfile>_rec_.par` (PyHST, 2026-08-31) | — | 1599.065161 | −18.8697 |

Seven of the nine configs agree on 1599.613741, including every `nabu_final*`,
with and without the horizontal correction — so the axis does not depend on
which shift file he fed it. `esrf_meta.nabu_axis` prefers `nabu_final_cm.conf`
because that is the one whose shift file is the `correct_correct3D.txt` we also
apply, and reports the others so a disagreement is visible rather than
silently resolved. `nabu_correct3D_v.conf` is the vertical-only variant and is
not our geometry.

The value here before was **−19.22**, measured locally — 0.35 px from the older
PyHST number. That agreement is what validates the padded-centre arithmetic;
the remaining 3.5 px is Peter's own refinement between 08-31 and 09-07, not a
convention error. The Aug 31 drop said 1600.565065 / PyHST 1602.0; that
directory has been replaced.

[`polaris_run.sh`](polaris_run.sh) is four literal `mpiexec` lines in sequence.
To run only part of it — steps 1–5 already done, or resuming after a
preemption — **comment out the lines you do not want**. Each ends in
`|| exit $?`, so a failed stage stops the job instead of letting the next level
seed itself from a checkpoint that was never written.

Splitting the job in two is no longer necessary for the rotation centre — it
comes from nabu now, not from a sweep. It is still the way to re-measure it if
Peter's shift files change again, since `rotation_center_shift` and the x column
of `cshifts_final` land in the same slot and only their sum is defined:

```bash
# comment out the three step-6 mpiexec lines, leaving steps15
qsub polaris_run.sh
python step5_center_sweep.py config_steps15.conf      # on a login node
# edit rotation_center_shift in ALL FOUR configs
# comment out the steps15 line, uncomment the three step-6 lines
qsub polaris_run.sh
```

The three step-6 levels share one `path_out`, so each seeds itself from the
previous level's checkpoint via `start_iter` and `Reader.read_checkpoint`
upsamples obj/prb/pos onto the finer grid. Iteration numbering is cumulative:

| config | bin | n | obj x/y | proj | start_iter | niter |
|---|---|---|---|---|---|---|
| `config_step6_bin2.conf` | 2 (4×4) | 1024 | 632 | 1264 | 0 | 1025 |
| `config_step6_bin1.conf` | 1 (2×2) | 2048 | 1264 | 2528 | 1024 | 1281 |
| `config_step6_bin0.conf` | 0 (1×1) | 4096 | **2528** | 5056 | 1280 | 1537 |

**Every level sets `tomo_upsample=2`,** with a parallel `tomo_upsample=1` ladder
in `config_step6_u1_bin*.conf` for comparison (see **The two arms** below).
`tomo_upsample=2` decouples the object grid from the
projection grid: the object's x/y is half the projection width, so at bin 0 `R`
maps `[5056, 2528, 2528] → [4000, 5056, 5056]`. The projection plane is
bit-for-bit the grid it always was — same FOV, same sampling, same centre — so
`pos`, `rotation_center_shift`, `eff_demag`, the probe, `Propagation` and the
whole cascade below `fwd_tomo` are untouched, and the fit still sees the full
detector at every level. What changes is that the object costs a quarter of the
memory: bin 0's would otherwise be 5056³. The level-to-level handoff stays a
plain ×2 in obj z, obj x/y, prb and pos, exactly as before.

The volume is anisotropic during the run — z voxel `v`, x/y voxel `2v`. That is
deliberate: z costs nothing extra in the Radon transform. `bin_z.py` averages
the bin-0 result down to an isotropic 2528³ afterwards, outside the
reconstruction:

```bash
mpiexec -n 8 python bin_z.py --iter 1536 --zbin 2   # -> checkpoint_1536_zbin2.h5
python extract_tiff.py --iter 1536 ...              # point it at the binned file
```

`lam_laplacian` **must stay 0 at every level** now: its stencil assumes
isotropic voxels and would penalise z gradients 4× too strongly against a
z:x = 1:2 object. `Rec` warns if both are on. Bins 2 and 1 previously ran it at
5e-5 and 1.25e-5; those levels now have no Laplacian term at all. Restoring it
would need per-axis weights in `LaplacianTerm`, which is not implemented. The
`tomo_upsample=1` arm has it at 0 too, even though its voxels are isotropic and
the stencil would be valid there — the two arms are only comparable if they
regularise the same way.

The detector frequencies above the object grid's Nyquist -- which only exist
once the projection plane is finer than the object -- **wrap**: the index is
taken mod 2n, so the object is modelled as a delta comb on its own grid and
those bins carry aliased replicas of the low frequencies.  That is what
`~/APS_PXM/tomo_usfft` does, and this kernel matches it.  The band-limited
alternative -- zero those bins, since a coarse object physically cannot produce
content above its own Nyquist -- is carried as a commented-out block in `gather`
in `cuda_kernels.py` and is restored by uncommenting.

The two differ **only** above the object's Nyquist; the `|f| < 1/2` content is
identical.  Measured at `n = 128 -> nd = 256`, 64 angles, disc phantom, against
the true fine-grid line integrals:

| | in-band err | out-of-band power | total model err | `fbp` at `nd=2n` |
|---|---|---|---|---|
| wrap | 0.0255 | 17.24x truth | 0.1925 | 4.3219 |
| zero | 0.0255 | 0.06x truth | 0.0278 | 0.4405 |

`fbp` at `nd = n` is 0.4405 for both, so zeroing is what makes `nd = 2n`
reproduce it.  BH never calls `fbp`; step 5's initial guess does.  On the
synthetic e2e run the solver converges under both, final/initial `err` 0.0855
(wrap) against 0.0672 (zero) -- so this choice is a model-accuracy question, not
the cause of a stall.

At `theta = 0` and `90` degrees -- and only there -- the wrap is exact rather
than aliasing, and those two sinogram rows come out as **combs** with every odd
detector sample exactly zero.  The shift between detector bins `tx` and `tx+n`
is `2n*cos(theta)` cells in the padded spectrum, and the mod-`2n` wrap maps it
onto itself only when that is `== 0 (mod 2n)`, i.e. `cos`/`sin` in `{0, +-1}`.
An integer shift is not sufficient: `theta = atan(3/4)` at `2n = 320` shifts by
a whole 256 cells and still aliases (`odd/even = 1.006`, no comb).

In practice this is **2 rows out of 4000** -- 0.05% of the sinogram -- and zero
rows if the angle grid does not land exactly on 0 and 90 degrees.  It is a
curiosity, not a reason to pick one model over the other; the real cost of the
wrap is the 17x out-of-band aliasing above, which affects every angle and is
invisible in a sinogram because it sits above the object's Nyquist.

At `nd = n` neither branch can trigger -- `fr` is in `[-1/2, 1/2)` -- so the
`tomo_upsample=1` arm and every pre-`tomo_upsample` run are bit-for-bit
unchanged whichever way the block is set (`max|R_wrap - R_zero| = 0`).

Unrelated trap found while measuring this: `R` returns **all zeros** for
`n % 4 == 1` (the `(1 - n % 4)` factor in `phi` evaluates to 0), at any `nd`.
Every size in this ladder is even, so it never bites here.

Two test scripts cover the option, both single-GPU and file-free:
`tests/tomo/test_tomo_nd.py` for the operator (`nd = n` is bit-identical to the
un-parameterised `R`, `R`/`RT` stay adjoint at `nd = 2n`, `R`'s *values* are
`nd`-independent, and the `|f| >= 1/2` bins carry the aliased replicas with the
θ = 0 and 90° rows combed — those last checks pin down which aliasing model is
compiled in), and `tests/tomo/test_upsample_e2e.py` for the
plumbing: a synthetic 2-distance step-6 run reconstructed from the same data at
`nobj=160, upsample=1` and at `nobj=80, upsample=2`, plus the bin1 → bin0
checkpoint read (obj z ×2, obj x/y ×1, prb ×2, pos ×2).

The half-set ladders carry the same `tomo_upsample=2` at all three levels, so
the three volumes — full, p0, p1 — share one grid and are directly comparable.

**Step 5 is untouched by this**, and does not have to be re-run. It writes its
Paganin+FBP init on the PROJECTION grid — `/exchange/obj_init_re60_2` at
1264³ — exactly as it always has, and `Reader.read_obj` averages it 2×2 in x/y
onto the object grid when the step-6 config asks for `tomo_upsample=2`. Binning
is an average, not a sum, because the forward model gives
`proj = C · mean_j(obj_j)` along the ray independently of the object grid; that
is the same value convention that lets obj carry between bin levels unrescaled
and that `bin_z.py` uses. Measured on a synthetic phantom, the two arms get
equivalent starting points: initial-guess best-fit scale 0.605 (u1) vs 0.647
(u2) against the truth, and `||R(init/norm) − psi||/||psi||` 0.378 vs 0.351.
z is not binned — the object keeps the full projection z.

Only bin 2 reads `obj_init` at all (`start_iter=0`); bins 1 and 0 resume from
checkpoints.

### Iterations, walltime and disk

1024 iterations at bin 2, then 256 each at bin 1 and bin 0. `polaris_run.sh`
does not repeat them in its comments, so the config is the single place they
live.

`checkpoint_step=32` divides every `start_iter` and every `niter−1`, so the
handoff checkpoints are guaranteed to exist. Preemption is survivable: a
resubmit loses at most 32 iterations.

**Walltime.** No timings exist for this scan. The closest measured ladder is
`../Y350a_HT` (4 distances, ntheta=4000, n=4096, nobj=4608) on 2 nodes / 8
ranks: 13 min + 22 min + 1 h 47 min ≈ 2 h 25 min of step 6. Scaling by the
1.20× area ratio gives ≈2 h 55 min here. Steps 1–5 read 537 GB of EDF and write
~2.1 TB back and were never timed on any scan — that is the real unknown, and
why the script asks for 18 h. Trim it once the first `.out` file exists.

**Disk**, in the steps15 `path_out`:

| file | size |
|---|---|
| `<pfile>.h5` (4000 × 4096² × 2 B × 4 dist) | 537 GB |
| bin-0 pdata (× 4 B × 4 dist) | 1074 GB |
| `<pfile>_obj.h5` (bin 2 only) | 4 GB |

≈2.5 TB, against 240 TB free on eagle as of 2026-08-31. Set
`start_level_rec=1` to stop at the 2×2 level if that changes.

`<pfile>_obj.h5` is small because `start_level_rec=2`: step 5 writes a
Paganin+FBP init only for bin 2, and only bin 2 ever reads one (bins 1 and 0
resume from checkpoints). That is 1264 × 632² × 4 B × 2 = 4 GB. Lowering
`start_level_rec` to 0 would add 5056 × 2528² (258 GB).

## Probe

**Not wired up, and it cannot be from what was transferred.** `prb_file` is left
unset, so step 6 starts from a flat probe and refines it (`rho[1] = 0.05`).

There *are* NFP companions, one per plane —
`projections/<pfile>_NFP_before_000k.nx`, each `70 x 4096 x 4096` uint16 with
`image_key` = 50 projections + 20 darks, and each carrying its own
`source/distance` matching the four `z1` values. But those datasets are HDF5
**virtual** datasets whose sources are

    ../../../../RAW_DATA/ctxl/<pfile>/scan000{4,5}/balor_0000.h5

and `RAW_DATA/` was never copied to eagle — `/eagle/APS_IRI/vnikitin/20260829/`
contains only `ctxl/`. Every frame therefore reads back as zeros. The same is
true of the *projection* `.nx` files (`entry_0001/instrument/detector/data`,
`4063 x 4096 x 4096`, virtual, all zeros): under ewoks the `.nx` files hold
**geometry only**, and the pixels live in the `<pfile>_k_/*.edf` trees, which is
what `esrf_layout.py` reads. Do not point anything at `.nx` for image data.

So there are two prerequisites for a measured probe, in order:

1. copy `RAW_DATA/ctxl/<pfile>/scan0004` and `scan0005` (per plane) from ESRF,
   or the NFP EDFs if they exist there;
2. adapt `step0.py`, which was copied verbatim from the 2025 folders and expects
   the bliss `<pfile>_NFPwS_1_/` EDF directory. Against the ewoks layout it
   would instead read `entry_NFP_before_000k/instrument/detector/data`, split on
   `image_key` (0 = NFP, 2 = dark), and take `z1` from
   `instrument/source/distance` — note the root group is
   `entry_NFP_before_000k`, **not** `entry`.

Until then no `config_step0.conf` is provided. The commented `prb_file` line in
`config_step6_bin2.conf` is where it would go once it exists; bins 1 and 0
inherit the probe from the previous level's checkpoint, so it only ever goes in
the bin-2 config.

## `paganin` = 60

`delta/beta = 60`, set 2026-09-07. Peter's octave driver for this scan says
`delta_beta=20` (`<pfile>_/ht_<pfile>.m`); 60 is a deliberate override — the
same value the 20 nm Y350a scan uses — chosen for the step-5 initial volume, and
it must be identical in `config_steps15.conf` and all three
`config_step6_bin*.conf` because step 6 rebuilds the Paganin reference with it.
The driver's other settings do corroborate the rest of the config:
`nvue=4000`, `refon=4000`, `numbers=[1 2 3 4]`, `random_disp=1`,
`reference_plane=2` (hence `ref_dist=1`, 0-based), `correct_shrink=0`.

## Files

| file | what |
|---|---|
| [`esrf_layout.py`](esrf_layout.py) | **the only place that knows bliss from ewoks**; filenames + geometry |
| [`config_steps15.conf`](config_steps15.conf) | steps 1–5 |
| [`config_step6_bin{2,1,0}.conf`](config_step6_bin2.conf) | the BH ladder |
| [`polaris_run.sh`](polaris_run.sh) | PBS job, one `mpiexec` line per stage; comment out what you do not want |
| [`show_geometry.py`](show_geometry.py) | prints the derived geometry and the per-level config blocks |
| [`scan_overview.py`](scan_overview.py) | the overview figure above |
| [`estimate_center.py`](estimate_center.py) | rotation centre from opposed projections |
| [`estimate_motion.py`](estimate_motion.py) | drift from the post-scan retakes |
| [`estimate_shrink.py`](estimate_shrink.py) | shrinkage from the post-scan retakes |
| [`estimate_correct3d.py`](estimate_correct3d.py) | ESRF's third shift file: identifies Peter's model, fits one for this scan |
| [`correct3d.png`](correct3d.png) | the residual common mode and the correct3D model of it |
| [`step5_center_sweep.py`](step5_center_sweep.py) | centre refinement from the FBP volume |
| [`show_iter.py`](show_iter.py) | one step-6 checkpoint: slices, probe, positions, convergence |
| [`show_pos_errors.py`](show_pos_errors.py) | total position correction across the whole ladder |
| [`diagnose_positions.py`](diagnose_positions.py) | splits that correction into stage runout, drift and displacement-stage error |
| [`tilt_check.png`](tilt_check.png) | three tests that rule an axis tilt out as the cause of the boundary artifact |
| [`esrf_vs_ours.png`](esrf_vs_ours.png) | ESRF's PyHST volume against this one, same sample, same physical scale |
| [`steps15.py`](steps15.py) | steps 1–5 driver |
| [`step6.py`](step6.py) | BH reconstruction driver |
| [`step0.py`](step0.py) | NFP probe retrieval — **bliss-only, unusable here**, see Probe |
