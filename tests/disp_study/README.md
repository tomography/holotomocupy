# Random-displacement / probe-smoothness study, single distance

Two scripts around the `holotomo3d` sample (same phantom, same ID16A probe,
same geometry as `../holotomo3d/test.py`):

| script        | what it does |
|---------------|--------------|
| `gen_data.py` | forward model → synthetic data with **random sample displacements** of half-width `--amp` px, from a probe blurred by `--prb-smooth` px |
| `rec.py`      | BH reconstruction of object + probe (+ positions) from that data |
| `make_figure.py` | middle slices of every reconstruction side by side, plus NRMSE vs the swept parameter |
| `run_dose.sh` / `compare_dose.py` | the equal-dose test: a four-distance scan against a single-distance one with 2.7x the angles |
| `run_brain.sh` / `gen_data_brain.py` / `rec_brain.py` / `make_figure_brain.py` | the brain test: the same study on a real reconstructed volume at the full unbinned 2048 px detector |

There are **two study parameters**, and either or both can be swept:

* `--amp` — how far the sample is randomly displaced between angles;
* `--prb-smooth` — how much structure the illumination still has.

Both feed the same thing: at a single distance the object is recovered only
through the diversity that a structured probe plus a moving sample put into the
data, so these two knobs turn that diversity up and down from opposite ends.

`common.py` holds the shared phantom/probe/geometry code, `run_study.sh` loops
over the grid of amplitudes × probe smoothnesses and calls `make_figure.py` at
the end.  Both scripts
put `../../src` at the front of
`sys.path`, so they always use this checkout of `holotomocupy` regardless of
what the environment resolves `import holotomocupy` to.

## Quick start

```bash
cd tests/disp_study

# one dataset + one reconstruction, single distance, +-8 px displacements
mpirun -np 4 ./set_affinity_gpu.sh python gen_data.py --amp 8
mpirun -np 4 ./set_affinity_gpu.sh python rec.py --in /home/beams2/VNIKITIN/tmp/disp_study/amp8_ndist1

# full displacement sweep (datasets under $OUT_ROOT, figures next to these scripts)
NP=4 AMPS="0 1 2 4 8 16 32" ./run_study.sh

# probe-smoothness sweep at +-16 px displacements
NP=4 AMPS=16 PRB_SMOOTHS="0 0.225 0.5 1 2 4 8 16" \
    OUT_ROOT=/data3/vnikitin/prb_study ./run_study.sh

# just redraw the figures from checkpoints that already exist
python make_figure.py --root /data3/vnikitin/disp_study --amps "0 1 2 4 8 16 32"
python make_figure.py --root /data3/vnikitin/prb_study --amps 16 \
    --prb-smooths "0 0.225 0.5 1 2 4 8 16"
```

The probe TIFFs (`prb_abs_2048.tiff`, `prb_phase_2048.tiff`) are looked up in
`/home/beams2/VNIKITIN/data/prb_id16a`; override with `--prb-dir` or
`HOLOTOMO_PRB_DIR`.  Datasets go to `$DISP_STUDY_OUT`
(default `/home/beams2/VNIKITIN/tmp/disp_study`) unless `--out` says otherwise.

## The displacement parameter

`pos[theta, dist, (y, x)]` is the sample displacement in **detector pixels**,
drawn uniformly from `[-amp, +amp]`; `amp = 0` reproduces a conventional
fixed-sample scan.  Displacements are independent per angle (and per distance
when `--ndist > 1`).

Two limits worth knowing at the default `nobj = 1.5 n` (n = 256):

* `|pos| > (nobj-n)/2 = 64 px` — the cubic B-spline shift extends the object
  grid symmetrically, so the crop starts sampling mirrored edge data;
  `gen_data.py` warns.  Raise `--nobj-factor` if you want to go further.
* `|pos| > ~0.15 n = 38 px` — the phantom itself starts to leave the detector
  field of view, so part of the volume becomes unmeasured.

So a sweep over `0 … 32 px` stays physically clean.

## The probe-smoothness parameter

`--prb-smooth` is the standard deviation, in **detector pixels**, of an
isotropic Gaussian low-pass applied to the ID16A probe before it is normalised
to `mean |prb| = 1` (so the flat-field level is the same for every value).  In
Fourier space the filter is `exp(-2 pi^2 sigma^2 |v|^2)` with `v` in cycles per
pixel, and `sigma = 1/(pi*sqrt(2)) = 0.2251 px` reproduces bit-for-bit the
`exp(-|v|^2)` filter `../holotomo3d/test.py` uses — that is the default, i.e.
"the probe as measured".

`gen_data.py` prints, and stores in `/prb_contrast`, the one-number summary
`std|prb| / mean|prb|`; at `n = 512` it falls off like

| sigma [px] | 0 | 0.225 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| contrast | 0.253 | 0.248 | 0.237 | 0.216 | 0.181 | 0.145 | 0.114 | 0.079 | 0.052 |

so `0 … 16 px` covers the whole way from the measured speckle to an almost flat
beam.  The reconstruction never sees this: it still starts from `prb = 1` and
refines the probe itself, so the sweep measures how much the *available*
illumination diversity matters, not how good the initial guess was.

Case directories are named `amp<amp>_ndist<N>[_prbs<sigma>][_objs<sigma>]`; each
suffix is left off at its default smoothness, so the directories of a pure
displacement sweep keep the names they already have.

## The object-smoothness parameter

`--obj-smooth` is the same knob for the sample instead of the illumination: the
standard deviation, in **object voxels**, of the isotropic Gaussian low-pass the
phantom is passed through after its shells and beads are stamped in.  Same
convention, `exp(-2 pi^2 sigma^2 |v|^2)` with `v` in cycles per voxel, and the
same default `1/(pi*sqrt(2)) = 0.2251` — bit-for-bit the `exp(-|v|^2)` filter
that has been baked into the phantom since `../holotomo3d/test.py`, so the
default phantom is unchanged and every volume already cached on disk is still
reused.

Unlike the probe's sigma, this one is in voxels of the object grid, so it means
the same physical blur whatever `nobj` is; at this geometry an object voxel and
a detector pixel are the same size in the sample, so the two sigmas are directly
comparable numbers.  `0` leaves the shells hard-edged, a few voxels rounds off
the thin outer layers and the small beads — it is the knob for "how much of what
we are trying to resolve is actually there to resolve".

    OBJ_SMOOTH=4 ./run_dose.sh          # the whole comparison on a blurred phantom
    python gen_data.py --obj-smooth 4 ...

The phantom cache is keyed on it (`common.phantom_id`), so a non-default value
builds and caches its own volume rather than silently reusing the sharp one.  It
has no effect on the brain test, which supplies its volume with `--obj-vol` and
uses it as it is; `gen_data.py` warns if the two are combined.

Not to be confused with `rec.py --obj-blur` (`OBJ_BLUR` in the run scripts): that
one blurs the reconstruction's *starting guess* and never touches the data, while
`--obj-smooth` blurs the ground truth itself and so changes what there is to
reconstruct.  One is where the solver starts, the other is where it should end up.

## The phantom

A wireframe cube dilated by a set of balls, giving concentric shells of
alternating value.  The shells are arranged as **three identical resolution
bands** — one just under the surface, one at mid radius, one deep inside —
each ramping from 3 units of `n/256` px down to 0.5, with a 5-unit spacer
between them:

```
band:      0.5  0.75  1.0  1.5  2.0  3.0     (units of n/256 px)
at n=512:  1.0  1.5   2.0  3.0  4.0  6.0     voxels
```

The same set of feature sizes therefore appears at three depths, so a
reconstruction can be scored not only on how thin a layer it still resolves but
on **how deep** it still resolves it — the innermost band is what an ill-posed
single-distance problem loses first.  0.5 units is 1 voxel at `n = 512`, i.e.
right at the sampling limit; it is meant to be the layer that disappears.  `common.layer_table()` prints the full listing.

### Small features in the middle

Shells only test how thin a *layer* is resolved.  Isolated small features are
the harder case — there is no low-frequency neighbour to lean on — so the deep
interior also carries a **bead target**: spheres of shrinking radius on a
three-armed cross through the middle of the sample.

```
distance from centre:  5.0   10.0   14.5   18.0   20.5   22.5   (units of n/256 px)
bead radius:           2.0    1.4    1.0    0.7    0.5    0.35
diameter at n=512:     8.0    5.6    4.0    2.8    2.0    1.4    voxels
```

Bead value is 6, above every shell value, so the contrast never vanishes
whichever shell a bead happens to sit in.  Positions are specified in the
*final* volume — after the rotation and the roll — and `gen_object` maps them
back through both, so the beads always land in the mid slices the figures show.
The arm along `a2` appears in both slices, the `a1` arm in the horizontal one
and the `a0` arm in the vertical one.  `common.bead_table()` prints the listing;
the spec is `common.BEAD_DIST` / `BEAD_RAD` / `BEAD_VALUE`.

### Size on the grid, and the margin

The phantom is generated at the **detector** size `n` (`common.gen_object(n)`)
and then written into the middle of the `nzobj x nobj x nobj` object grid with
zeros around it (`common.write_centered`) — the same placement `fill_volume`
makes for `--obj-vol`, which rescales the source array to `span = n` px wide and
centres it.  So all the feature sizes above are fixed by `n` alone, and `nobj`
buys nothing but blank border: `MARGIN = (nobj - n)/2` is room for the sliding
crop, not a change of sample.  (Before 2026-08-23 the phantom was built on the
`nobj` grid instead, so raising the margin shrank the sample relative to the
field of view; datasets generated before that date are not comparable with ones
generated after it.)

The whole specification lives in `common.LAYER_THICK` / `LAYER_VALUES` /
`BEAD_*`, and `common.phantom_id()` hashes all of it into the phantom-cache
filename, so editing it can never make an old cache silently reappear.  It does mean reconstructions
made before an edit are no longer comparable with ones made after it.

## Object initial guess

By default the reconstruction starts from the ground-truth phantom smoothed
with an isotropic 3D Gaussian of `--obj-blur` px (8 by default, i.e. far wider
than any feature of the phantom).  That hands the solver only the coarse
low-frequency envelope — everything the study is about, the high-frequency
detail recovered from the positional diversity, still has to be reconstructed —
while keeping the first BH steps out of the flat region around `obj = 0`.  Use
`--obj-init zeros` for a cold start and `--obj-init true` as an upper-bound
reference.

`rec.py`'s own default is 8 px; both dose comparisons use 20 (`OBJ_BLUR` in
`run_dose.sh` and `run_dose_brain.sh`), so the phantom and the brain are scored
from the same kind of start.  Note that `sigma = 0` is *not* a cold start —
`gaussian_blur3d` is the identity there, so `--obj-blur 0` is exactly
`--obj-init true` and hands the solver the exact ground truth.  The phantom runs
in `/data3/vnikitin/dose_study_phantom` predate this default and were made that
way: their NRMSEs measure how far BH drifts away from a perfect start, not how
much detail it recovers, and are not comparable with the brain's.

Each rank filters its own z-slab plus a halo of `4*sigma` slices, so the result
is bit-for-bit what filtering the assembled volume would give; the cost is one
extra read of the phantom, a few seconds.

## Single distance needed a library fix

`ndist = 1` used to make every multi-rank BH run return NaN from the first
iteration (`ill-conditioned alpha denominator bottom=nan`).  The cause was in
`Chunking.gpu_batch`: it classifies an output as chunked ("proper") by
`shape[axis_out] == size`, and the scalar accumulator in `redot_batch` /
`linear_redot_batch` was shaped `(1,)`.  With a single distance the chunking
length of the probe- and position-shaped calls *is* 1, so the accumulator was
mistaken for a chunked output — and chunked outputs are backed by
uninitialised arena scratch, so the alpha numerator picked up whatever the GPU
pool happened to hold.  Every rank got a different value, only rank 0's counts,
and the step size came out ~200x too large.

Fixed by making those accumulators 0-d (`ndim > axis_out` is then false, so
they can never alias a chunking length) in `src/holotomocupy/chunking.py`.
Single- and four-rank runs now agree to seven digits.

## Files written per dataset

```
<out>/data.h5                    data, ref, true pos, probe, ground-truth object, geometry
<out>/rec_n<n>_ntheta<ntheta>/   one folder per detector size / angular sampling
    checkpoints/                 checkpoint_NNNN.h5  (obj, prb, pos)
    checkpoints_tiff/            mid-slice previews
    pos_errors/                  per-angle position drift plots
    conv.csv                     iteration, cost, time
    summary.txt                  masked NRMSE vs the ground-truth phantom
```

The ground-truth object is a deterministic function of things that do not vary
across a sweep — `(n, delta, beta)` for the phantom, `(source file, nobj,
scale, delta/beta)` for a real `--obj-vol` — so `gen_data.py` builds it once and
caches it beside the datasets, as
`phantom_n<n>_delta<delta>_beta<beta>_<hash>.h5` or
`objvol_nobj<nobj>_scale<scale>_<hash>.h5`; every later case copies it back slab
by slab instead of rebuilding it (`--phantom-cache none` to disable, or point it
at another file).  The hash covers everything the result depends on, including
the source file's size and mtime, so a replaced volume or an edited phantom
never reads back a stale cache.  The phantom cache holds the bare `n^3` sample,
not the padded grid, so one cached phantom serves every margin; the `--obj-vol`
cache is the whole rescaled `nobj^3` grid, which is what `fill_volume` streams.  Caching a real volume matters most: rescaling
the 3072³ source reads the whole 116 GB file, ~150 s of mostly disk, and
`run_dose_brain.sh` alone generates twice.

`data.h5` stores `data` as **sqrt of intensity** (what `Rec` consumes), in the
distance-major `[ndist, ntheta, nz, n]` order of `Rec.data`; `pos` is stored
theta-major `[ntheta, ndist, 2]`, the same layout as the `/pos` of a checkpoint.
The full layout is documented at the bottom of `common.py`.

## Useful options

`gen_data.py`

| flag | meaning |
|------|---------|
| `--amp` | displacement half-width [px] — study parameter 1 |
| `--prb-smooth` | Gaussian blur sigma [px] applied to the probe — study parameter 2 (0 = as measured, default 0.2251 = as in holotomo3d) |
| `--ndist` | number of distances (1 = the single-distance case) |
| `--photons` | mean photons/px for Poisson noise (0 = noiseless) |
| `--n`, `--ntheta`, `--nobj`, `--nobj-factor` | problem size (the detector pixel is `1.476 um * 2048/n`, i.e. the same 2048² chip binned) |
| `--obj-vol` | use a real volume as the object instead of the phantom: a path, or `path::dataset` for HDF5 |
| `--obj-scale` | multiplies `--obj-vol`; this is what sets the projected phase (the run prints the measured value) |
| `--seed` | RNG seed for the displacements |
| `--prb-dir` | directory with the ID16A probe TIFFs |

`rec.py`

| flag | meaning |
|------|---------|
| `--in` | dataset directory (or `data.h5`) |
| `--niter` | BH iterations |
| `--pos-err` | perturb the initial positions by ±this many px (0 = start from truth) |
| `--obj-init` | `blur` (default), `zeros` or `true` — see *Object initial guess* |
| `--obj-blur` | sigma [px] of the 3D Gaussian for `--obj-init blur` (default 8) |
| `--checkpoint-step` | iterations between checkpoints — one is a full `nzobj x nobj x nobj` complex volume, so at n = 2048 it is 111 GiB |
| `--prb-init` | `ones` (default, nothing assumed) or `true` |
| `--freeze-prb`, `--freeze-pos` | hold that variable at its initial value (`rho <- --frozen-rho`) |
| `--frozen-rho` | rho for a frozen variable (default 1e-3 → 1e-6 of the object step) |
| `--warmup N` | first N iterations with prb and pos frozen, then release |
| `--rho` | step scales `obj,prb,pos`. `--rho 1,0.05,1e-3` effectively freezes the positions; 0 is **not** allowed (0/0 in the alpha ratio) |
| `--estimate-rho` | pick `rho_prb` and `rho_pos` automatically — see *Estimating rho* |
| `--rho-estimate-niter` | iterations per trial of `--estimate-rho` (default 16) |
| `--lam-prbfit`, `--lam-laplacian` | regularisation weights |
| `--no-gt` | skip the ground-truth comparison |

## Estimating rho

`--rho` sets the step scale of each variable (the step goes as `rho^2`), and a
badly chosen `rho_prb` or `rho_pos` either stalls that variable or blows the run
up. `--estimate-rho` searches for them instead of guessing, using
`Rec.estimate_rho_coord` (ported from `holotomocupy_mpi_deform`):

* Snapshot the state right after `precalc`.
* Coordinate search, `prb` first and then `pos`, over the geometric grid
  `..., init/2, init, 2*init, ...` centred on the `--rho` value. Each candidate
  is scored by rewinding to the snapshot, running `--rho-estimate-niter`
  iterations silently, and evaluating `Rec.min`. If the centre wins it is kept;
  otherwise the search walks in the winning direction until the error stops
  improving (at most 8 rungs).
* A trial that diverges — CUDA error, or a non-finite functional — scores `inf`,
  so the search steps past it rather than crashing.
* The state is rewound one last time and the main loop runs with the rho it
  found. `rho_obj` is never searched; it is the reference the other two are
  measured against.

The cost is 3 to 19 trials per variable-ish (3 probes plus the walk), each
`--rho-estimate-niter` iterations, plus one extra copy of `vars` for the
snapshot — at n = 2048 that copy is the same 111 GiB a checkpoint is, so keep
`--rho-estimate-niter` small there or leave the flag off. The chosen values are
logged (`estimate_rho_coord: final rho = [...]`) and written to `summary.txt` as
`rho a,b,c (estimated)`.

With `--warmup N` the search runs at the start of the *second* phase, not the
warmup one, since the warmup rho is discarded anyway.

## Reading the result

`rec.py` prints, and writes to `summary.txt`:

```
amp=8 px  niter=257  NRMSE(obj)=...  NRMSE(delta)=...  NRMSE(beta)=...  mean(delta_rec-delta_gt)=...
```

NRMSE is computed inside the support mask against the phantom.  The mean
offset is reported separately because a single distance with an unknown probe
leaves a weak constant-offset ambiguity in `delta`; a large NRMSE together
with a large offset means the ambiguity, not a failure to converge.

`make_figure.py` writes `slices_ndist<N>_n<n>_ntheta<nt>.png` and
`nrmse_ndist<N>_n<n>_ntheta<nt>.png` into this folder (`--out` to change, `--tag`
to keep two sweeps in one root apart), reading the last checkpoint of every case
it finds and skipping the ones that have not been reconstructed yet.  One column
per case; the NRMSE plot puts whichever parameter is being swept on the x axis
(probe smoothness wins when both vary, with one line per amplitude).  When the
smoothness varies it also adds a `|probe|` row, so the illumination that
produced each column is visible next to it (`--probe-row on/off` to force).


## Ground-truth provenance

The phantom specification in `common.py` is editable, and every `data.h5` keeps
its own copy of whatever was current when it was written.  A sweep assembled
over several sessions can therefore contain two different ground truths, and
its NRMSE curve is then not one curve.  `make_figure.py` fingerprints
`obj_re[::37, ::37, ::37]` of each case, labels the distinct phantoms `[A]`,
`[B]`, ... in the panel titles, and draws each of them as its own unjoined
segment in the NRMSE plot.  A case whose `data.h5` is *newer* than its own
`summary.txt` was scored against a phantom that has since been overwritten and
can no longer be identified; it is labelled `[?]` / `phantom overwritten`.

If the figure comes out with one letter and no warnings on stderr, the sweep is
internally consistent.

## Dose-equivalent single-distance scan

The beam diverges from a focus, so at sample-to-focus distance `z1` the same
photons cover an area `~ z1^2` and the fluence in the sample goes as `1/z1^2`.
With `Z1_ALL = [5.110, 5.464, 6.879, 9.817] mm` the four distances cost, per
projection, relative to the nearest one:

| distance | 1 | 2 | 3 | 4 | sum |
| --- | --- | --- | --- | --- | --- |
| z1 [mm] | 5.110 | 5.464 | 6.879 | 9.817 | |
| magnification | 238 | 223 | 177 | 124 | |
| dose / projection | 1.000 | 0.875 | 0.552 | 0.271 | **2.697** |

So `ntheta` angles at all four distances cost 2.697 `ntheta` near-distance
projections, not 4 -- and a single-distance scan **at the nearest position**
matches that dose with

    N = 2.70 * ntheta          (common.dose_equivalent_ntheta)

| 4 distances, ntheta each | 300 | 450 | 900 | 1500 |
| --- | --- | --- | --- | --- |
| dose-equal single distance | 809 | 1214 | 2428 | 4046 |

Read the other way: the `ntheta = 450` single-distance runs in this study carry
the dose of a four-distance scan with **167 angles per distance**.

If the single distance is not the nearest one the factor grows, because the
reference projection itself is cheaper: 3.08 at z1 #2, 4.89 at #3, 9.96 at #4.

The equality is on photons through the reconstructed volume as well as on dose:
both scale as `1/z1^2`, so equal dose means equally many recorded ROI photons.
What differs is the exposure count -- 4 x 450 = 1800 frames deliver what 1214
single-distance frames do, the far frames being individually dose-cheap.

### Running the comparison

`run_dose.sh` runs both halves of that trade at equal dose and
`compare_dose.py` puts them side by side:

```bash
./run_dose.sh                              # 4 x 450  vs  1 x 1215, amp +-16, sigma 2
NTHETA=900 ./run_dose.sh                   # 4 x 900  vs  1 x 2430
AMP=4 PRB_SMOOTH=0 OUT_ROOT=/data3/... ./run_dose.sh
```

Defaults: `AMP=16`, `PRB_SMOOTH=2` (the optimum found above), `NTHETA=450`,
`NDIST=4`, `NITER=513`, `PHOTONS=0`, `NP=3`, into `/data3/vnikitin/dose_study`.
The dose-matched angle count is computed by `common.dose_equivalent_ntheta` and
rounded to a multiple of `NP` -- `2.6974 x 450 = 1213.8 -> 1215`, within 0.1 % of
exact and evenly splittable over the three ranks.  The two cases land in

    ndist4_ntheta450_amp16_prbs2/       4 distances, 450 angles each
    ndist1_ntheta1215_amp16_prbs2/      1 distance, 1215 angles

(`common.dose_case_name`, which unlike `case_name` puts `ntheta` in the
directory name -- the whole point of this pair is that it differs).  The
multi-distance run is given a proportionally smaller `--nchunk`, since holding
`ndist` distances at once costs `ndist` times the memory per chunk.

The two reconstructions are directly comparable: the object voxel grid is set
by the magnification of the *first* distance, which both share, so the NRMSEs
are against the same phantom on the same grid.  `compare_dose.py` writes
`dose_ndist4_n512_ntheta450.png` -- ground truth and the two reconstructions on
the top row, their error against the phantom below, with each column labelled by
its projection count and relative dose.  An unfinished run keeps its column and
leaves the panel blank, as in `make_figure.py`.

Next to it goes `dose_ndist4_n512_ntheta450_zoom.png`: one detail region of that
same horizontal slice, the multi-distance reconstruction above the
single-distance one, drawn `interpolation='nearest'` in the same colour range as
the main figure so the two textures can be compared pixel for pixel.  The region
is `--zoom x0,y0,w,h` (default `680,450,200,200`) in pixels of the *uncropped*
slice, so `--crop` does not move it; `--zoom off` skips the figure and
`--zoom-box on` outlines the region on the reconstruction panels of the main
figure.

`run_dose.sh` does not use that default: it passes a `ZOOM_W`-wide box centred
on the object grid (`277,277,150,150` at the standard `NOBJ = 704`, `ZOOM_W =
150`), since the phantom sits centred in `NOBJ` and the middle is where its
structure -- the wedge edges and the shrinking bead row -- actually is. Set `ZOOM_W` for a
different size or `ZOOM=x0,y0,w,h` (or `ZOOM=off`) to look somewhere else.

There is no convergence plot: `conv.csv` records the data-fit residual, which is
a sum over a different number of measurements in the two runs and is not
comparable between them.

(The rule of thumb in which the last distance is `pi/2` times the first gives
2.69 for geometric and 2.64 for linear spacing -- the same answer.  The real
`Z1_ALL` spans a factor of 1.92, not 1.57, but the sum is dominated by the near
distances either way.)

### The input frames

`show_data.py` draws the other end of the pipeline -- what the reconstruction is
given rather than what it produced.  One column per distance, four rows:
`|probe|`, `arg(probe)`, the intensity recorded at one angle (`--theta`, default
the first), and that frame divided by the flat field.  It reads `data.h5` only,
so it runs as soon as `gen_data.py` is done; no reconstruction needed.

    python show_data.py --root /data3/vnikitin/dose_study_phantom \
        --ndist 4 --ntheta 900 --amp 1 --prb-smooth 1 --tag phantom

-> `data_ndist4_n1024_ntheta900_amp1_prbs1_phantom.png`.  The fourth row is the
point of the figure: the ID16A probe's speckle is several times the sample's
contrast, so in the raw frame the phantom is barely a texture on top of the
illumination, and only the ratio shows the holograms -- and their fringes
spreading, and the projection shrinking with the magnification, from the near
distance to the far one.  `--flat off` drops that row.  `/data` and `/ref` hold
the *square root* of intensity; the script squares them back, so the panels are
intensities.

Any amplitude will do -- the frames differ between them only by a sub-pixel to
tens-of-pixels shift of the sample -- so `--amp 1` is the one worth keeping.
Point `--ndist 1 --ntheta 2704` at the other leg of a dose-matched pair to see
its single frame instead; its probe is the near-distance column of the four.

`--root /data3/vnikitin/brain_dose_study --tag brain` does the same for the
brain test, whose probe is the identical ID16A one -- the top two rows repeat.
What differs is the sample: the phantom is a compact object in a blank field, so
its hologram sits in an untouched background, while the brain block fills the
field and only its edge fringes are strong.  Those edges set the colour range
and flatten the tissue inside, so `--clip 99` is worth a look there; the figures
in this folder keep the default 99.8 so the two samples stay comparable.

## The brain test

`run_brain.sh` runs the same single-distance problem on something much less
forgiving than the phantom: a real reconstructed volume
(`/data3/vnikitin/mosaic_brain/init.h5`, 3072³ float32) at the **full unbinned
detector**, n = 2048.

| | |
|---|---|
| object | the stored volume × `--obj-scale 15`, rescaled to 1266 object px wide |
| detector | 2048 × 2048, pixel 1.476 µm (`common.detector_pixelsize(2048)`) |
| probe | the measured ID16A probe at its native 2048², `--prb-smooth 0` — no crop, no resampling |
| scan | 1800 angles, one distance, noiseless, displacements ±128 px |
| recon | 1025 BH iterations, positions started ±2 px off, initial object = the truth blurred by 32 px |

```bash
./run_brain.sh                        # the single-distance case
NDIST=4 ./run_brain.sh                # the four-distance reference
NTHETA=900 NITER=513 ./run_brain.sh   # a quick shakedown
```

`gen_data_brain.py`, `rec_brain.py` and `make_figure_brain.py` are one-screen
launchers that pin those defaults and hand everything else to `gen_data.py`,
`rec.py` and `make_figure.py`; any flag on the command line overrides the
default of the same name, so the brain test is not a fork of the study, just a
configuration of it.

### How wide the sample is

The whole 3072³ source array is rescaled to the detector width `n` — 1/3 at
n = 1024 — and the reconstruction grid adds a blank `MARGIN` (default 128 px) on
each side, so

    nobj = n + 2*MARGIN     (1280 px at n = 1024)

The sample itself is whatever part of the source array is not air: it is a
cylinder of radius 0.477 of the array, so ~977 px across at n = 1024, leaving
~150 px of blank grid around it on top of the margin.  `nobj - n = 2*MARGIN`
also caps the displacement: `amp` may not exceed `MARGIN`, or the sliding n-wide
crop reads mirrored edge data (`gen_data.py` warns).  `MARGIN` and `NOBJ` are
runner variables; `gen_data.py` takes the grid directly as `--nobj`.

### The projected phase

The Radon normalisation folds `nobj`, `ntheta` and `norm_const` together, so the
only reliable way to know the phase excursion is to measure it: `gen_data.py`
now prints

```
projected phase  [-xx, +xx] rad
projected absorp [...]      transmission exp(-imag) in [...]
```

right after the forward model.  A few rad to a few tens of rad is a
well-conditioned test; `--obj-scale` is the knob.  `obj_re` is the stored volume
(already −δ-like, i.e. mostly negative) and `obj_im = -obj_re/(delta/beta)`,
δ/β = 100 by default.

### What is different at n = 2048

* **Pixel size.**  `common.detector_pixelsize(n) = 1.476 µm × 2048/n` — the same
  chip binned.  It returns exactly the old constant at n = 512, so every n = 512
  result is unchanged; at any other `n` the geometry is now right instead of
  borrowing the n = 512 pixel.
* **The initial-object blur.**  σ = 32 on a ~750 × 2458² slab is a 257-tap kernel
  over 4.5e9 voxels along each axis — hours in `scipy.ndimage`.
  `common.gaussian_blur3d` does it as an in-plane pass over z-slices plus a z
  pass over bands of rows on the GPU, holding one slice or one band at a time.
  It falls back to scipy below `GPU_BLUR_BYTES` (2 GiB, override with
  `DISP_STUDY_GPU_BLUR_BYTES`), so the n = 512 study still takes the scipy path
  and is bit-for-bit what it was.
* **Checkpoint size.**  One checkpoint is `obj_re` + `obj_im` at 2458³ = 111 GiB,
  so the default step is 512 iterations, not 32 — at 32 the run would write
  3.5 TB.
* **The volume is streamed.**  `common.fill_volume` writes `/obj_re,/obj_im` one
  destination z-slice at a time (integer block-average, Gaussian pre-filter for
  the residual factor, then a linear zoom, all on the GPU), so nothing ever holds
  an nobj³ array.  The result is cached (see above), so the source file is read
  once per `(nobj, scale, delta/beta)` rather than once per case.

### Rough size

```
data.h5   /data    1800 x 2048²  float32     28 GiB   (x ndist)
          /obj_re + /obj_im  2458³ float32   111 GiB
checkpoint                                   111 GiB each
host RAM  ~160 GiB per rank at NP=4 (3 obj slabs + 3 proj slabs + data)
```

## Figures in this folder

Everything here comes from one root, `/data3/vnikitin/disp_study_final` on
`tomo5` -- one phantom (`117b87e9`, the one with the bead target), `ndist=1`,
`n=512`, 513 BH iterations, noiseless.  Every case was run at **two angular
samplings**, `ntheta = 450` and `ntheta = 900`, and each gets its own pair of
figures.  Figures from the earlier roots were removed rather than kept around:
they were scored against older phantoms and are not comparable with these.

| figure | what |
| --- | --- |
| `*_ndist1_n512_ntheta450_amp.png`, `*_ntheta900_amp.png` | displacement sweep, amp 0 / 4 / 16 / 64 px, probe at the default sigma |
| `*_ndist1_n512_ntheta450_prb.png`, `*_ntheta900_prb.png` | probe-smoothness sweep at +-16 px, sigma 0 / 1 / 2 / 4 / 16 / 64 px |

Masked NRMSE(obj):

| amp [px] (sigma = default) | 0 | 4 | 16 | 64 |
| --- | --- | --- | --- | --- |
| ntheta = 450 | 0.527 | 0.370 | 0.348 | 0.339 |
| ntheta = 900 | 0.509 | 0.223 | 0.204 | 0.186 |

| sigma [px] (amp = +-16) | 0 | 1 | 2 | 4 | 16 | 64 |
| --- | --- | --- | --- | --- | --- | --- |
| ntheta = 450 | 0.367 | 0.297 | 0.297 | 0.308 | 0.368 | 0.437 |
| ntheta = 900 | 0.210 | 0.197 | 0.203 | 0.219 | 0.297 | 0.404 |
| probe contrast | 0.253 | 0.216 | 0.181 | 0.145 | 0.079 | 0.032 |

The default sigma (0.225 px; NRMSE 0.348 at 450, 0.204 at 900; contrast 0.248)
is left out of the probe figure: it is visually indistinguishable from sigma = 0
and only crowds the axis.  It is the probe used throughout the amplitude figure.

Most of the gain against displacement is in the first step off zero.  Doubling
the angles buys much more than any of these knobs -- a third off the error
everywhere except at amp = 0, where displacement diversity is absent and 900
angles barely help (0.527 -> 0.509): the missing diversity is not angular, and
more of the same projections cannot supply it.

The probe-smoothness optimum survives the doubling but flattens: at 450 angles
the minimum at sigma = 1-2 px is clear (0.297 against 0.367 at sigma = 0), at
900 angles sigma = 0 / 1 / 2 sit within 0.013 of each other (0.210 / 0.197 /
0.203).  With enough angles the object no longer needs the probe to carry
diversity, so the default 0.225 px is only really on the wrong side of the
optimum in the angle-starved regime.  Over-smoothing still costs the same in
both: sigma = 64 px doubles the error either way.

`slices_*` is the image comparison (object slice + `|probe|` + `arg(probe)`),
`nrmse_*` the metric.  The probe sweep puts sigma on a symlog x axis and repeats
the probe contrast `std|prb| / mean|prb|` along the top; `nrmse_beta` gets its
own right-hand axis, being some fifty times the other two.  Both were drawn with

    --slices h --vmin -4 --vmax 1 --crop 0.10 --probe-cmap gray

i.e. the horizontal (z = mid) slice only, a fixed colour range, 10% trimmed off
each side, and both probe rows in grey.

A case that has no `summary.txt` yet -- one still on the GPU -- keeps its column
in the figure but leaves the object panel empty, so a sweep can be plotted while
it is still running without a half-converged checkpoint posing as a result.
