# AtomiumL1 — FT large random displacement, 14 nm, 1 distance

ESRF ID16A, proposal `ihmi1591`, acquired 2025-06-08, two hours before the HT
scan of the same sample. Raw data on **tomo5** at
`/data3/vnikitin/ESRF/atomium/20250607/AtomiumL1/AtomiumL1_FT_large_rand_disp_014nm_1_/`.

The single-distance counterpart to `../AtomiumL1_HT`: the same near plane
(z1 = 6.113 mm), with ±300 px random sample displacements between projections
in place of the four propagation distances. Structured like
`../Y350a_largedisp`.

## Verified geometry (`python show_geometry.py config_steps15.conf`, 2026-08-22)

| | |
|---|---|
| ndist / ntheta | 1 / 1800 |
| detector | 2048 × 2048, pixel 2.9520 µm |
| energy | 33.35 keV |
| focus→detector | 1.289 m (sx0 = 1.280 mm) |
| z1 | 6.113 mm — magnification 210.86 |
| voxel size | 14.00 nm |
| nref / ndark | 21 / 20 |
| n / nobj (bin 0) | 2048 / 2688 |
| `correct.txt` sweep | y ∈ [−299.74, +299.30] px, x ∈ [−298.90, +299.50] px |

All 1800 projections, 21 ref pairs and 20 darks are present.

`nobj=2688` rather than the auto 2048: with one distance the normalised
magnification is 1, so the auto value is exactly the detector footprint and
leaves no room for the ±300 px sweep. 2688 = 42·64 covers 2048 + 2·300 = 2648
in full and bins cleanly to 1344 / 672.

## Running

On **tomo5** (4 x A100-SXM4-40GB; `/data3` is not mounted on handyn):

```bash
SCRIPT=step0.py CONFIG=config_step0.conf ./local_run.sh    # optional NFP probe
./local_run.sh                                             # steps 1-5
SCRIPT=step6.py CONFIG=config_step6_bin2.conf ./local_run.sh
SCRIPT=step6.py CONFIG=config_step6_bin1.conf ./local_run.sh
SCRIPT=step6.py CONFIG=config_step6_bin0.conf ./local_run.sh
```

Ladder: bin 2 (n=512, nobj=512, iters 0–1024) → bin 1 (1024/1024, 1024–1280)
→ bin 0 (2048/2048, 1280–1536). Disk: ~350 GB.

The step-6 solver settings are a straight copy of `../AtomiumL1_HT`'s at the
matching bin level — same `niter`/`start_iter` ladder, `nchunk`, `lam_prbfit`,
`lam_laplacian`, `rho`, `mask`, `checkpoint_step`, `error_step`, and
`estimate_rho` at bin 2. The two scans are the same sample on the same detector
at the same angle count, so the configs differ only in `ndist`, the paths and
`rotation_center_shift`. Copy in that direction — HT first, then here — so the
two stay comparable.

## Finding the rotation centre

`step5_center_sweep.py` redoes step 5 alone, once per candidate
`rotation_center_shift`, and keeps only the middle slice of each — so a whole
sweep costs about as much as one step-5 run rather than one per candidate
(measured: 1.2 s per candidate at bin 2 on 4 A100s, plus ~2 s to prime the
cache). It needs steps 1–4 to have run; it reads its geometry back out of
`{path_out}/{pfile}.h5`, so the raw EDF tree does not have to be mounted. Flags
come after the config, which `local_run.sh` cannot pass, so run it directly:

```bash
mpirun -n 4 ./set_affinity_gpu.sh python step5_center_sweep.py \
    config_steps15.conf --start -20 --stop 20 --step 1
```

That writes `{path_out}/center_sweep_bin2/` — one TIFF per candidate, named
`center_<100-shift>_r<shift>.tiff` with both numbers in **unbinned** pixels, a
`center_sweep_bin2.tiff` stack of all of them in ascending shift order, and a
`.txt` table. Open the stack in ImageJ, scroll to the frame where the edges stop
splitting, and read the shift off the frame's `r` field. To refine, sweep a
narrow range around the winner — but note the units: at bin 2 four unbinned
pixels are one grid pixel, so `--step 1` is already a quarter-pixel step there
and anything finer needs `--bin 1` (4x the cost, still under a minute).

## Estimating the sample drift during the scan

`estimate_drift.py` measures how far the sample wandered over the 1800
projections, on top of the ±300 px random displacement step 3 already knows
about and removes. It redoes the step-5 Paganin pass in memory for **every**
angle (step 5 itself only stores every 10th to `{pfile}_proj.h5`), takes the
centre of mass of each phase projection, subtracts projection 0, and fits
polynomials to the two resulting curves. Like `step5_center_sweep.py` it reads
its geometry back out of `{path_out}/{pfile}.h5`, so only steps 1–4 have to have
run, and flags come after the config:

```bash
mpirun -n 4 ./set_affinity_gpu.sh python estimate_drift.py config_steps15.conf
```

~7 s wall on 4 A100s at bin 2, all 1800 angles, everything at its default. It
writes `{path_out}/drift_bin2/`: `drift_bin2.png` (the six panels below),
`drift_bin2.txt` (coefficients, per-degree rms, and a per-angle table) and
`drift_bin2_profiles.npz` (the raw marginals, centroids and applied shifts, if
you want to re-fit without recomputing). A copy of the figure is checked in next
to this README:

![measured drift and its polynomial fits](drift_bin2.png)

### What it is a centroid *of*, and why that is the whole problem

Not of the phase. The Paganin phase carries a large low-frequency background —
a bright halo hugging the object, dark corners — and the per-frame
normalisation makes the air level jump by ~0.05 and the air spread double
between projections **0.1° apart**. Every weight built on that level fails, and
each failure was measured before being discarded:

* **`max(air − φ, 0)`, air at a percentile.** Rectification gives every air
  pixel positive weight, so the air pedestal drags the centroid towards the
  centre of the analysis window and the window moves with the applied
  displacement. `corr(dy, cshift_y) = −0.76`, joint-fit slope −0.32: a third of
  the ±300 px the stage applied came straight back out as "drift". This is what
  produced the 236 / 256 px answer this section used to report; it was an
  artefact.
* **`max(lvl − φ, 0)` with `lvl` a fraction of the object depth (`--seg
  depth`).** Fixes the pedestal, but the level has to sit *somewhere* in the
  object's depth, and with the contrast wobbling frame to frame a large part of
  the object crosses it: the segmented area swings between 4 % and 51 % of the
  window and the centroid of a set that changes that much scatters by 45–60 px
  between adjacent angles — three times the drift being looked for.
* **`air − φ` unclipped.** The halo goes *above* the air level, so the weight
  changes sign, the denominator nearly cancels and the centroid runs off to
  thousands of px.
* **`|∇φ|` (`--seg grad`, the default).** A smooth background differentiates
  away; an additive per-frame offset differentiates away exactly; and there is
  no decision boundary for a wobbling contrast to push pixels across — the
  object's edges are where the gradient is, at every angle. Point-to-point
  noise **3.6 / 3.5 px**, a factor of 15 better than any level rule, and the
  leak essentially gone.

Two more things the estimator needs, both found the hard way:

* **The analysis window is the same at every angle** (`--window common`, the
  intersection of all 1800 measured windows, here 302 × 301 of 512 binned px).
  The object is *bigger* than the measured window and the window moves with the
  ±300 px displacement, so a per-angle window means taking the centroid over a
  different piece of the object each time. With a fixed window the residual air
  pedestal pulls towards a *constant* point, which attenuates the measured
  motion slightly instead of injecting the applied shift.
* **The fill outside the measured window is ramped to 1 (air), not `'edge'`.**
  `_stitch` in `steps15.py` pads vertically with `'edge'`, which replicates the
  top row outwards — and when the dark sample mount crosses the top of the
  window that becomes a solid black bar, by far the strongest edge in the
  frame, moving with the random displacement. Fatal for a centroid. This script
  deliberately deviates; step 5 keeps `'edge'` (BH only fits the measured
  region, so it does not care, but the bar *is* in the Paganin `obj_init`).

The window is additionally eroded by `--guard` (default = the Paganin kernel's
1/e width, 26 binned px at bin 2) so the kernel's skirt does not reach across
the seam, and `--box Y0,Y1,X0,X1` can restrict it further; neither is needed
with the gradient weight.

### The two acceptance tests

Both are printed at the end of every run and plotted:

1. **Leak.** Step 3 already removed the applied `cshifts_final`, so a correct
   estimator must show no trace of it: `corr` and the joint-fit slope of the
   measured centroid against the applied shift must both be ~0. Now
   **−0.07 / −0.002** on dy and **+0.30 / +0.010** on dx (bottom-right panel).
   The dx slope of 0.01 means about 3 px of the 4.5 px residual is still leak —
   small, but the one number that has not been driven to zero.
2. **Point-to-point noise.** Consecutive projections are 0.1° apart, so
   `std(diff)/√2` is the estimator's own noise with no assumption about the
   drift: **3.6 px (dy), 3.5 px (dx)** unbinned, i.e. under a binned pixel.

The same statistic at longer lags is the structure function, printed as a table
and the thing that says whether a polynomial is the right model at all:

| lag [°] | 0.1 | 0.2 | 0.5 | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| dy | 3.63 | 3.64 | 3.81 | 3.83 | 4.01 | 4.15 | 4.55 | 4.84 | 4.85 |
| dx | 3.53 | 3.45 | 3.53 | 3.46 | 3.58 | 4.08 | 4.44 | 5.13 | 6.02 |

It is **flat** from 0.1° out to 5°, so on that timescale there is nothing but
estimator noise and everything above it is the smooth drift the polynomial
fits — which is exactly the case a polynomial model is for. `../AtomiumL1_HT`
is the counter-example: there the same curve climbs from the first lag, and the
fit residual is real motion. Compare the two before trusting either fit.

### What it found for this scan

The script is shared with `../AtomiumL1_HT`, which runs it on the four-distance
scan of the same sample two hours earlier. Develop it here — this is the harder
case — and copy the file across.

Measured (bin 2, all 1800 angles, defaults, **unbinned** px):

| | dy | dx |
|---|---|---|
| raw centroid peak-to-peak | 36.8 | 34.8 |
| off-axis orbit amplitude | — | 7.6 |
| **drift ptp, deg 2** | **8.0** | **0.6** |
| deg 3 | 10.0 | 7.3 |
| deg 5 | 12.0 | 6.0 |
| rms residual, deg 2 / 3 / 5 | 4.56 / 4.49 / 4.44 | 4.92 / 4.75 / 4.48 |

**The sample drifted about 10 px vertically and essentially not at all
horizontally**, over 180° — a tenth of a percent of the field, and about 0.6 µm
in the sample. Read the deg-2 row and stop there: going from 2 to 5 degrees buys
2.6 % of rms while the vertical drift grows 8 → 12 px and the horizontal one
starts oscillating (right-hand panels), which is a quintic chasing 3.5 px of
noise, not structure. The vertical curve is a settle, not a ramp: most of the
10 px happens in the first 50° and it comes part way back by 180°.

On x the raw centroid is dominated not by drift but by an `A·cos θ + B·sin θ`
orbit of 7.6 px amplitude, which is exactly what a sample whose centre of mass
sits 7.6 px off the rotation axis must produce, with zero drift. It is fitted
and removed first — jointly fitting `poly5 + cos + sin` is hopeless (over a
180° arc `cos θ` is nearly a cubic in θ; `cond = 5.9e4`, and the fit returned a
"drift" of 267 484 px). `--no-orbit` skips the removal, `--orbit-y` adds it on y
too, where it should not exist and does not help.

The honest caveat is the bottom-left panel: with `nobj = 2048` between 1 % and
22 % of the sample's edge mass (mean 6.9 %) falls outside the measured window,
angle by angle, and a centroid over a support that changes is biased by roughly
(lost fraction × object half-width). At 6.9 % of a ~190 binned px half-width
that is the same order as the drift being reported, so treat 10 px as an upper
bound on a real motion, not a calibrated number. Getting below it needs the
object to stay inside the grid at every angle, i.e. `nobj > n` — see open
item 5.

### Feeding the drift back into the reconstruction

`--export-correct3d` writes the fit where step 3 of `steps15.py` looks for it,
`{path}/{pfile}_/correct_correct3D.txt`, and the next run of steps 1–5 folds it
into `shifts_final` along with the random, rhapp and motion shifts:

```bash
mpirun -n 4 ./set_affinity_gpu.sh python estimate_drift.py config_steps15.conf \
    --export-correct3d              # --export-deg 5 by default
```

It is a degree-5 polynomial least-squares fitted to the measured centre of
mass on the sampled angles and evaluated at **all** `ntheta` of them, so a run
with `--stride` still writes a complete file and every evaluation is an
interpolation. Three things about it differ from everything reported above, on
purpose:

* **It is the fit to the centre of mass itself**, the left-hand column of the
  figure — not to any orbit-free or re-referenced version of it.
* **No orbit is removed.** A plain degree-5 polynomial through the *raw*
  centroid, whatever `--no-orbit` / `--orbit-y` did to the report. That is the
  right thing to apply: `A·cos θ + B·sin θ` in x is exactly what a rigid
  translation of the object projects to, so keeping it only re-centres the
  reconstructed volume laterally — tomographically consistent either way, and
  one less thing to get wrong. It does mean the exported x amplitude (13.5 px
  ptp) is larger than the orbit-free drift in the table above (6.0 px); y is
  unchanged at 12.0 px, there being no orbit on y.
* **Nothing is added to or taken off the curve.** What `lstsq` returns is what
  is written — no re-zeroing at projection 0, no mean-centring. The one
  reference already in the numbers is the measurement's own: `dy_un` / `dx_un`
  are the centroid *relative to projection 0*, which is a property of the data,
  not a choice the export makes. On this scan that shows up as a pedestal — the
  dy cloud sits about 20 px below projection 0, so the exported y runs around
  −21 px while spanning only 12.0 px — and it is harmless: a constant common to
  all angles in y just translates the reconstructed volume. In x it is less
  inert (a common constant moves the rotation axis, competing with
  `rotation_center_shift`), but here the exported x constant is small.
* **Sign and units were measured, not assumed.** Adding a constant to `r` at
  read time moves the centroid the other way (`r = (8,0)` took the binned
  centroid from y = 254.02 to 246.03), i.e. `centroid = const − r`, so an
  excursion of `+d` is cancelled by writing `+d`; and because `r = cshifts /
  2**bin`, the number to write is `d` in **unbinned** px — exactly what the
  script already reports, no rescaling.

Column order is **x, y**: step 3 reads the file as
`np.loadtxt(...)[:ntheta, ::-1]` and then tiles the same row across all
distances. It is 1800 rows and refuses to overwrite an existing one without
`--export-force` — checked before the GPU work, not after. `--export-path` sends
it somewhere else for a dry run. A copy also lands in
`{path_out}/drift_bin2/drift_bin2_correct3D.txt`, and the exported curve is the
**black dashed line** on the two left-hand panels of `drift_bin2.png`, drawn
exactly as written.

The figure is two columns: the measured centre of mass with its fits on the
left, the reliability trace and the acceptance test on the right. The
orbit-free "drift" panels that used to occupy the right column are gone — they
showed the same polynomials with a constant and an `A·cos θ + B·sin θ` taken
out, which is not what is exported and only invited the comparison. The orbit
amplitude is still reported, in the left panels' titles and in the text table.

Written 2026-08-23: ptp x 13.48 px, y 11.98 px; largest value in the file x
7.20, y 21.57 px (the y pedestal above); rms residual 4.48 / 4.44 px.
`{pfile}_/` did not exist for this scan and was created to hold it.

## Open items

1. **`rotation_center_shift` is borrowed, not measured.** There is no PyHST
   reconstruction of this scan (the `AtomiumL1_FT_large_rand_disp_014nm_/` aux
   directory holds nothing but the `correct_correct3D.txt` written above -- no
   par file, no `rhapp.mat`), so `-14.375964` comes from the HT scan's par file. Same sample,
   same stage, two hours apart — plausible, but the axis can drift. Re-measure
   with `step5_center_sweep.py` (above) before the bin-0 run, and put the
   winner in `config_steps15.conf` **and** all three `config_step6_bin*.conf`.
2. **`paganin=120` is inherited** from AtomiumS2 / AtomiumL1_HT. Check on a
   bin-2 slice.
3. **`step0.py` has not been run yet.** Unlike the HT scan this dataset does
   have a usable NFP companion — `*_NFPwS_1_`, 50 frames with `spy`/`spz`
   spanning ±75 px — so `config_step0.conf` is enabled. Once it has run,
   uncomment `prb_file=nfp_results.h5` in the `config_step6_bin*.conf` files.
   Do **not** point it at `*_NFPwCA_1_`: that one has `cay`/`caz` and
   `step0.py` raises `KeyError` on it.
4. **This scan is not dose-matched to AtomiumL1_HT.** At 1800 angles it costs
   1803 near-distance projections against the HT scan's 5414 (`dose.py`), i.e.
   1/3.0028 of the dose. A fair HT-vs-largedisp comparison would need either
   the HT scan cut to ~600 angles or this one at ~5406 — which the data does
   not have. Worth knowing before reading anything into a side-by-side.
5. **Bin 0 is marginal on tomo5.** The configs currently use `nobj=2048`, i.e.
   no margin at all around the 2048 px detector footprint — 69 GB of complex64
   object against a 160 GB aggregate (4 x 40 GB), which fits but leaves little
   for the probe, the data chunks and the Hessian work arrays. The `nobj=2688`
   this README argues for above would be 155 GB and would certainly need
   Polaris. Decide the margin before running bin 0: it lives in
   `config_steps15.conf`, so changing it means re-running steps 1–5, and all
   three `config_step6_bin*.conf` have to follow.  `estimate_drift.py` (above)
   is new evidence for the margin: with `nobj=2048` a mean 6.9 % of the sample's
   edge mass falls outside the measured window, up to 21.9 % at the worst angle,
   and that is not just a nuisance for the drift estimate — it is data BH has no
   grid to put.
