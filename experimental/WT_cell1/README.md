# WT_H cell01 — 2-D near-field ptychography, 25 nm

ESRF ID16A, proposal `ihls3905`, beamtime `20260929`, scan
`WT_H_cell01_NFP2D_025nm_0001`.

This folder is **step 0 and nothing else**. The scan is a standalone NFP2D
measurement — 50 frames on a 50-point spiral at a single rotation angle
(`somega = 35°`) — so there is no tomogram, no `steps15.py` and no `step6.py`.
What comes out is the probe and one 2-D projection of the cell.

```
experimental/WT_cell1/
  config_step0.conf           the one config
  step0.py                    NFP reconstruction from the NXtomo
  nx_frames.py                virtual-dataset reader (verbatim copy of ../AtomiumS1_FT_RD300/)
  polaris_run.sh              qsub wrapper, one mpiexec line
  set_affinity_gpu_polaris.sh one rank per A100
```

---

## Geometry

Read out of `projections/WT_H_cell01_NFP2D_025nm_0001_scan0001.nx` by
`reader.read_nxtomo_meta`; every number below was verified against the file.

| quantity | value | where it comes from |
|---|---|---|
| energy | 17.1 keV | `instrument/beam/incident_energy` |
| detector pixel | 2.9520 µm | `instrument/detector/x_pixel_size` |
| z1 (focus→sample) | 10.27259 mm | `-instrument/source/distance` |
| z_total (focus→detector) | 1213.000 mm | `detector/distance − source/distance` |
| magnification | 118.08 | z_total / z1 |
| **voxel size** | **25.000 nm** | pixel / magnification |
| frames | 50 data, 20 dark, 20 flat | `detector/image_key` = 0 / 2 / 1 |
| detector | 2048 × 2048 uint16 (balor) | |
| positions | y ∈ [−49.3, +51.1] px, x ∈ [−53.7, +49.6] px | spiral, 16 px between points |
| `nobj` | 2176 | ceil32(n + 2·(⌈max|pos|⌉+8)) |

The voxel size falls out at exactly 25.000 nm and `nobj` at exactly 2176 —
both match what ESRF's own run recorded (below), which is the cheapest
available check that the geometry is being read the same way.

### Positions

`nxtomomill` recorded the motor mapping in the `.json` beside the `.nx`:

```
x_translation = sy[mm] + spyp[µm]      →  spy, the horizontal motor
y_translation = sz[mm] + spzp[µm]      →  spz, the vertical motor
```

which is exactly what `read_nxtomo_meta` returns as `x_trans` / `y_trans` and
what `step0.py` converts to `pos = [−spz, spy] / voxelsize`. Nothing in this
folder guesses at the motor convention.

### The flats are not used — and the per-frame flux fix is parked

In NFP the probe **is** the unknown, so there is nothing to flat-field by.
Only the 20 darks enter the reconstruction: `step0.py` subtracts their mean,
clips at 0, and then normalises the **whole stack by one global mean**, which
is the scale `rho` and the PSF were tuned at. The 20 flats are counted in the
log and otherwise ignored by the solver.

> **Status, 2026-10-06.** The rest of this section documents a real defect —
> a 1.9 % per-frame flux drift that the global mean leaves in — and a fix for
> it that was written, verified and run once (job 7720369, output kept at
> `nfp_prb32_flux_posfrozen/`). The fix is **not** in `step0.py` right now: it
> is parked in `step0_perframe.py.bak`, because the `rho[prb]` sweep took
> priority and is being run on the unmodified loader so every arm shares the
> baseline's data. Restore it with
> `cp step0_perframe.py.bak step0.py`. The measurements below stand; only the
> shipped state changed.

Measured on the real frames (2026-10-05): dark mean 98.9 ADU, flat mean 8099
(8000 above dark), dark-subtracted data mean 8068 — so the sample attenuates
almost nothing on average, as expected for a cell. After normalising, the
data has mean 1.000, std 0.519, max 8.1: strong near-field fringes, no clipped
zeros.

#### The 1.9 % flux drift, and why one global mean was wrong

That last paragraph is the measurement that *looked* fine and hid the problem.
The stack mean was already exactly 1, so nothing in `err` or in the reference
curve ever complained — but the frames are not all at the same brightness:

```
frame mean, dark-subtracted:  8157.6 -> 8006.5, monotonic   1.89 % p-p
machine current:              199.71 -> 199.66 mA           0.03 %
```

It is **not the ring**. `control/data` is flat at 199.7 mA (top-up), and
dividing by it removes 0.03 of the 1.89 points, so that column is useless
here. The flats continue the same ramp — flat mean 8000.4 against a last data
frame of 8006.5 — so it is one continuous optics or thermal drift across the
whole acquisition, not a sample effect.

It matters because it is **larger than the signal**: 1.79 % p-p after flat
division against a 1.55 % object modulation. A per-frame gain the model cannot
represent (one probe, one object) has to go somewhere, and it goes into the
fit.

The fix is the one `../AtomiumS1_FT_RD300/steps15.py:419-421` already applies on the
tomography path — scale every frame to a common target instead of scaling the
whole stack by one scalar. Here the target is literally 1.0, which is
identical on every rank by construction, so unlike steps15 no `Bcast` is
needed and the result is independent of rank count. The old
`local_sum`/`allreduce`/`global_mean` block is gone.

Verified on the real frames: frame-mean spread 1.885 % → 1.8e-5 %, stack mean
preserved to a ratio of 1.000000, normalised data still mean 1.0000 / std
0.5188 / max 8.17 — so `rho`, `psf_sigma` and the `nfp_BH_basic` reference
curve all stay comparable.

**Why the whole-frame mean is a safe reference.** It also absorbs any real
frame-to-frame change in the object's mean transmission — but barely: the
reconstructed object's own per-frame window mean varies by 0.00015 p-p against
0.0179 p-p of instrument drift, a 120:1 ratio, so the normalisation costs
~0.9 % real signal. An air ROI would be cleaner in principle but there is no
air here: every 256-px corner has std ~0.003, the same order as the full
field.

**Outlier removal was deliberately not ported.** `steps15.py` median-filters
zingers (radius 9, threshold 0.9) before normalising. This scan has none — 0
pixels above 10× the frame mean, max/mean only 8.1 in both data and flats —
and a radius-9 median on a 1.5 %-contrast object is a real risk of smearing
the signal we are trying to recover.

**The darks need nothing.** Their per-frame means are stable (0.47 ADU p-p on
98.9 ADU), so a plain mean of 20 is right. The subtract-then-normalise order
*is* load-bearing, though: the dark is 1.2 % of the signal, the same order as
the drift, so normalising first would fold a constant pedestal into the
per-frame gain. `step0.py` logs the dark spread so a future dataset with a
drifting dark says so.

**What this does and does not fix.** It is a real defect and worth fixing on
its own, but it is *not* established to be the cause of the probe features in
the reconstruction. Forward-modelling the measured gain error onto the object
grid gives std 0.00025 against the reconstructed β's 0.01188 — ~50× too small
— because the frames overlap 95 % (103 px of diversity on a 2048 px probe), so
the per-frame error averages to nearly a constant over the object instead of a
radial pattern. The probe contamination sits almost entirely in β: β/δ std
ratio 2.81 where physics says ~1e-3 at 17.1 keV, corr(β, |prb|) −0.78 and
corr(β, measured flat) −0.88 at periods above 5 µm, while δ stays at −0.05.
That points at the probe/object split being unconstrained at low frequency,
not at the flux. Job 7720369 ran the fix and confirmed the prediction: the
log reported the drift being removed (`p-p=1.89%`, `corr(frame index,
mean) = -0.988`) and the probe features were still there. Weighting the probe
harder (`rho[prb]=32`) helped visibly where the flux fix did not, which is
what the sweep below is chasing.

**The twelve siblings still have the old code.** `AtomiumS1/step0.py:201-204`
and the `raw_slice` form at `Y350c/step0.py:154-155` (and the others) all
normalise by one global scalar. Whether that hurts them depends on whether
those scans drift, which nobody has measured. Worth checking per dataset; not
changed here.

---

## The `rho[prb]` sweep

`config_prb{1,2,4,8,16,32,64,128}.conf` differ from `config_step0.conf` in two
lines: `rho=1,<p>,1e-2` and `path_out=…/nfp_prb<p>/scan0001`. Everything else
is held — `niter=1025`, `model=amplitude`, `estimate_rho=False`,
`psf_sigma=0`, `shift_type=fft`, the global-mean loader. Submit one arm with

```
qsub -N WTprb16 -v CFG=config_prb16.conf polaris_run_sweep.sh
```

or let `sweep_driver.sh` walk the ladder: the debug queue takes **one queued
job per user**, so the arms cannot be fanned out. Score them with
`python compare_arms.py <arm>/nfp_results.h5 …`, which reports the probe-leak
metrics (β/δ std ratio, `corr(obj, |prb|)` above 5 µm, and the level and
spread of the four empty corners).

`nfp_prb32_flux_posfrozen/` is **not** part of the ladder: it additionally has
the per-frame flux fix and `rho[pos]=1e-5`.

### The second ladder: same arms, per-frame flux

`config_prb<p>_flux.conf` + `polaris_run_sweep_flux.sh` + `sweep_driver_flux.sh`
repeat the ladder into `nfp_prb<p>_flux/scan0001`, running `step0_flux.py`
instead of `step0.py`. That is the **only** difference between the two
families, so `nfp_prb<p>` vs `nfp_prb<p>_flux` is a one-knob comparison at
every rung. The two drivers must not run at once — they would fight over the
single debug-queue slot.

`plot_conv.py` and `sweep_table.py` both read the two families together: flux
solid, global mean dashed and faint. A global-mean curve that stops short is a
cancelled job and is dropped from the figure; a short flux curve is an arm
still running and is kept.

**Why the second ladder exists.** `rho[prb]` turned out not to be the knob —
across 1…32 on global-mean data, β std moved 0.01265 → 0.01392 (worst at the
top) while `corr(β,|prb|)` stayed near −0.7. The flux fix, at **`rho[prb]=2`**,
did the whole job: β std 0.01253 → 0.00229, corner β window-to-window 0.03788
→ 0.00098, β/δ 4.50 → 0.61, and δ std *up* 0.00279 → 0.00375 — signal moving
out of absorption, where it never belonged, and into phase. `err` agrees: the
flux arm sits 16 % below every global-mean arm from iteration 16 on, against
the 2 % band the whole `rho[prb]` ladder spans.

The plan that proposed the flux fix predicted it was **~50× too small to
matter**, having forward-modelled the gain error's *direct imprint* on the
object (std 0.00025 against β's 0.01188). That reasoning was wrong, and the
way it was wrong is worth keeping: what actually happens is the solver's
*response*. Data that no single probe × object can fit has to go somewhere,
and β — the near-degenerate channel — absorbs it. Do not size a data defect by
its direct imprint when a near-null-space variable can soak it up.

### `rho` is squared, and `pos` is float32

Both matter when reading a `rho` value, and neither is obvious from the config:

- `rec_nfp_mpi.py:84` stores `rho_sq[v] = args.rho[v]**2`, and
  `compute_gradient` multiplies each gradient by it. So `rho[prb]=32` scales
  the probe gradient by **1024**, and `rho[pos]=1e-5` scales the position
  gradient by **1e-10**.
- `alpha` is a *single* scalar shared by all three variables
  (`compute_alpha`), so raising one `rho` shrinks the step the others get.
- `vars['pos']` is **float32** (`rec_nfp_mpi.py:151`) holding values up to
  ±62 px, where the float32 spacing is 3.8e-6 px.

Together these are why `rho[pos]=1e-5` printed position errors of *exactly*
`0.0000`: the increment `alpha·1e-10·|grad|` is below the float32 floor unless
`alpha·|grad_pos| > 4e4`, so the update is silently dropped by rounding rather
than merely small. The sweep uses `rho[pos]=1e-2` (→ 1e-4), which needs
`alpha·|grad_pos| > 0.04` to register; if the `pos err` lines are still exact
zeros, the next step up is 1. Nothing logs the raw position step, so those
lines are the only evidence.

---

## The virtual dataset, and why `nx_frames.py` is not a verbatim copy

`instrument/detector/data` in the `.nx` is an HDF5 **virtual dataset**: it
holds no pixels, only three links into RAW_DATA written relative to the full
beamline tree,

```
./../../../../RAW_DATA/WT_H/WT_H_cell01_NFP2D_025nm_0001/scan000k/balor_0000.h5
```

Our copy of the tree is flatter in two places:

```
20260929/PROCESSED_DATA/projections/…nx          ESRF: PROCESSED_DATA/WT_H/<scan>/projections/
20260929/RAW_DATA/WT_H_cell01_NFP2D_025nm_0001/  ESRF: RAW_DATA/WT_H/<scan>/
```

so those four `..` land in the wrong place, and **HDF5 reports an unresolvable
source as the fill value — zeros, with no error and no warning**. A run would
succeed and hand back a probe retrieved from zeros.

`nx_frames.NxFrames` takes the mapping apart and rebases each source: the
recorded path first, so an untouched beamline tree keeps working, then the same
tail off every ancestor of the `.nx`, nearest first. The one change from
`../AtomiumS1_FT_RD300/nx_frames.py` is `_tail_variants`, which also tries the tail with
the `<sample>` level dropped — that is exactly the level missing above. The
raw root and the last three components (`<scan>/scan000k/balor_0000.h5`) are
never dropped, so a shortened match still names one file unambiguously.

Check it without starting a job:

```bash
python nx_frames.py ~/eagle/vnikitin/20260929/PROCESSED_DATA/projections/WT_H_cell01_NFP2D_025nm_0001_scan0001.nx
```

As of 2026-10-05 all three sources resolve (`scan0002` 168 MB darks,
`scan0003` 419 MB frames, `scan0004` 168 MB flats) and it exits 0. `step0.py`
runs the same check and refuses to start while any source is missing.

---

## Running it

```bash
qsub polaris_run.sh
```

**Do not scale this out.** It is 50 frames: `get_local_chunk` over more than
~50 ranks leaves ranks with zero. The script asks for 2 nodes = 8 ranks = 6–7
frames per rank, which is the intended size. One node also works.

For a first look, the debug queue is enough — the reference run below cost
~35 s per 16 iterations on ESRF's single GPU, so ~9 min for all 256:

```bash
qsub -q debug -l select=1 -l walltime=00:30:00 polaris_run.sh
```

Locally, on a box with the env:

```bash
mpirun -n 4 python step0.py config_step0.conf
```

### Output

`path_out = PROCESSED_DATA/nfp_BH_basic_rerun/scan0001/` — the same shape as
ESRF's `nfp_BH_basic/scan0001/` next to it, so the two are directly
comparable and neither overwrites the other:

- `conv_nfp.csv` — the error curve, on the same iteration grid as the
  reference's.
- `nfp_results.h5` — `prb_amp`, `prb_phase`, `proj_delta`, `proj_beta`,
  `pos_err`, `pos_init`, plus the geometry as file attributes.
- `checkpoints_tiff/proj{iter}.tiff`, `prb{iter}.tiff`, `prb{iter}.npy` —
  every `checkpoint_step` (64) iterations.

---

## What ESRF already produced, and how to compare

Three directories sit in `PROCESSED_DATA/` beside `projections/`. Two of them
are **our own code**, not
pynx, run at ESRF on 2026-10-05 through a CLI front end (`--bin / --niter /
-O= / --rho_pos / --load_probe`) that this repo does not have:

| dir | what ran | result |
|---|---|---|
| `nfp/` | pynx `pynx-ptycho-id16a-nf --algorithm=ML**2000,probe=1 --adu_scale=0.65` | CXI with object 2184×2192, probe 2048×2048; positions **not** refined (the correction rows of `result_5` are ~1e-14) |
| `nfp_BH_basic/` | holotomocupy, `--bin 0 --niter 256 --rho_pos=1e-5` | one full-resolution stage, n=2048, nobj=2176; ~35 s per 16 iterations |
| `nfp_BH_full/` | holotomocupy, a 4-level cascade: bin3 n=256 ×1024 iters (`rho_pos=1e-5`), then bin2 / bin1 / bin0 ×64 iters each with `rho_pos=0.1` and `--load_probe --load_object --load_position` | same full-resolution endpoint after only **64** iterations at n=2048; its four stages started 2:16 apart in total |

`config_step0.conf` starts from **`nfp_BH_basic`** — same `n`, `nobj`,
`ntheta`, `nchunk`, `rho`, `error_step` — and then deliberately differs in
three places: `niter=513` instead of 256, `estimate_rho=True`, and
`model=amplitude`. Its `conv_nfp.csv` is the reference curve:

```
iter      err
  -1      6.467e-02
   0      7.635e-04
  16      4.172e-05
  64      4.044e-05
 144      3.944e-05
```

Read it for the **shape**, not the values: `amplitude` measures the misfit in
a different space, so its `err` is roughly 4× smaller and the two columns are
not comparable number for number. What does carry over is that `err` has
dropped three decades by iteration 16 and is flat to three digits long before
the end. A first `err` that is not within an order of magnitude of 6.5e-2 is
still the fastest sign that the data scaling is off — and the fastest sign
that the frames came back as zeros.

`err` is **not** sensitive to the per-frame flux drift above, though: the
stack mean was exactly 1 both before and after that fix, so the curve looked
healthy throughout. That is why the drift went unnoticed, and why
`step0_perframe.py.bak` logs the frame means and their spread rather than
relying on `err`.

`err` **is** comparable across the `rho[prb]` arms, which I got wrong at
first. `RecNFP.min` ([rec_nfp_mpi.py:879](../../src/holotomocupy/rec_nfp_mpi.py#L879))
sums `F0` alone — no `rho`, no regularisation — so `rho` changes the
trajectory but not the functional being measured. The arms do start from
wildly different first steps, though: a bigger `rho[prb]` takes a much larger
iteration 0 (7.0e-3 at `rho[prb]=1` down to 4.8e-5 at 8), which is why the
curves only become readable after a log axis and a tail zoom.

`plot_conv.py` draws both from each arm's `conv_nfp.csv`:

```
python plot_conv.py              # -> conv_rho_prb.png, one line per arm
```

Left panel semilog over the whole run, right panel linear from iteration 128
where the arms separate. Arms still running are plotted as far as they got.
What `err` does **not** measure is the probe leak — the arms sit within 2 % of
each other on `err` while `compare_arms.py` spreads them much further, so the
leak metrics, not the final `err`, are what the arms are scored on.

(`nfp_BH_full/scan0001/conv_nfp.csv` is overwritten by each stage, so only the
tail of one survives; do not read it as a curve. The per-stage parameters above
come from the CXI's `process_1` groups, one per `entry_000k`, which are
complete.)

The binned cascade of `nfp_BH_full` is **not** reproducible here: it needs
`rec_bin_pow` and the load-probe/object/position plumbing, which live in the
CLI front end, not in this repo's `RecNFP`/`parse_args_step0_nx`. Porting it
would be worth it — it reached the full-resolution answer in a quarter of the
full-resolution iterations, and it is the variant that actually refines
positions — but it is a feature, not a config.

---

## Knobs

**`rho=1,2,1e-5`** — object, probe, positions. The third entry freezes the
positions, which is what the reference full-resolution run did: the motor
positions are good, and refining them from scratch at n=2048 is what the
cascade exists for. Set it to `0.1` to let them move, and read `pos_err` in
the output h5 to see how far they went.

**`estimate_rho=True`** — coordinate-searches `rho[prb]`, then `rho[pos]`, on
a geometric grid around the values above before the loop starts; `rho[proj]`
is the reference scale and is left alone. Costs 3–19 silent trials of
`rho_estimate_niter` (16) iterations per coordinate. Only a `rho` pinned at
exactly 0 is skipped, so `rho[pos]=1e-5` *is* searched and can be walked up
by up to 8 rungs (×256) — which unfreezes the positions. Watch `pos_err` in
the output h5. Set `rho_trial_error_step=16` to see inside the trials.

**`psf_sigma=0.0`** — one Gaussian on the detector *intensity*, sigma in
unbinned detector px (NFP never bins). Leave it at 0 until something measures
it: `err` is biased toward sigma=0 and **cannot** choose it. The empty-air rms
can; on AtomiumS1 FT that method picked 1.7 px.

**`model=amplitude`** — or `intensity`. Amplitude is the repo default since
2026-10-03 and the correctly weighted least squares under photon-counting
noise; on the synthetic pair it needed ~2.4× fewer iterations purely from
conditioning. `err` is **not** comparable between the two models.

This knob only works here because `step0.py` in this folder *forwards*
`args.model` into `RecNFP`. The twelve sibling `experimental/*/step0.py`
parse the key and then drop it, so `model=` is silently ignored there and
`RecNFP` falls back to its own `intensity` default whatever the config says.
Worth fixing across the board; not done here.

**`niter=513`** — twice the reference's 256. `err` is flat to three digits
well before either, so this is headroom, not a requirement; drop it back to
256 for a quick turnaround.
