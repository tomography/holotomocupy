# Y350a — large random displacement, 6 nm voxels

ESRF ID16A, proposal ls3231, beamtime 2025-06-04 (scan taken 2025-06-07).
Stained brain tissue, one propagation distance, 4000 projections over 180°,
±300 px random sample displacement.

This is the **6 nm sibling of [`../Y350a_largedisp`](../Y350a_largedisp)**
(same sample, same beamtime, same detector, 20 nm voxels). Everything that is
not resolution-dependent is deliberately identical to that folder so the two
reconstructions can be compared. It has never been reconstructed before — there
was no prior output anywhere on eagle, so every number here was either derived
from the raw data or carried over from the 20 nm run and marked as such.

| | 6 nm (here) | 20 nm (`../Y350a_largedisp`) |
|---|---|---|
| voxel size | 6.000 nm | 20 nm |
| z1 (focus→sample) | 5.24 mm | 17.23 mm |
| magnification | 246.0× | 73.8× |
| field of view | 24.6 µm | 82 µm |
| rotation_center_shift | **−37.50 px** (measured here) | −10.966 px |
| NFP companion scan | **yes** (`_NFPwS_1_`) | no |
| `correct_motion.txt` | **yes** (estimated here) | yes (from ESRF) |
| shrinkage correction | none — *measured* as absent | separate `_noshrink` ladder |

Everything else — 4096² detector, ntheta=4000, `mask_oob=1`, the bin 2→1→0
ladder — matches. `nobj` is 4736 here against 4608 there (see below), and
`paganin` is 40 against 60.

## What is and is not corrected

**Shrinkage: off, and measured to be off.** `rho[tp]=0` at every level. There
is no `shrink_list.mat`, so the model would initialise to A=B=0 regardless, but
the reason it is also frozen there is now a measurement rather than a default.

[`estimate_shrink.py`](estimate_shrink.py) gets it from the scan's own post-scan
retakes. A sample that shrank makes the retake a *scaled* copy of the frame it
repeats, so the residual displacement field stops being a constant translation
and acquires a slope; block-wise correlation plus an affine fit reads that slope
off directly:

| pair | spans | A_y | A_x |
|---|---|---|---|
| 4002 (ω=0) vs frame 0 | whole scan | **−75 ± 99 ppm** | −246 ± 473 ppm |
| 4001 (ω=90) vs frame 2000 | second half | −975 ± 3443 ppm | −1078 ± 1934 ppm |

Nothing, at a 2σ limit of **\|A_y\| < 275 ppm** over the whole scan — 0.56 px at
the frame edge. Two things make that a real limit rather than a failure to
measure. The pairs span different fractions of the scan, so a linear shrink must
show twice as much at ω=0 as at ω=90, and both pairs agree (0.3σ). And
`--inject 500` puts a known 500 ppm into the retake: the fit returns 501 ppm.

For scale, on the 20 nm sibling the optimizer *invents* A_y ≈ +4.5 px and
B_y ≈ −21.6 px of 4096 — around 2200 ppm of edge displacement, 8× the limit
above. That is the physical cause of the empirical result: noshrink is 15-24 % better on every largedisp ladder tested, because there is no shrinkage
to fit and the term absorbs displacement the position refinement should own.

Unlike the 20 nm folder there is no second ladder to compare against, so the
configs carry no `_noshrink` suffix.

**Shifts: `correct.txt` only.** Step 3 combines four sources; three are absent
here and default to zeros:

| source | file | present? |
|---|---|---|
| encoder / random displacement | `<pfile>_1_/correct.txt` | superseded — see the note below |
| inter-plane (RHAPP) | `<pfile>_/rhapp.mat` | no (and meaningless at 1 distance) |
| slow drift | `<pfile>_1_/correct_motion.txt` | **yes** — estimated here, see below |
| 3-D tomographic | `<pfile>_/correct_correct3D.txt` | no |

At ndist=1 the four sources telescope, so `correct_motion.txt` does not *add*
to `correct.txt` — it **replaces** it. The file installed here is therefore
`correct.txt` plus the drift, in the same unbinned object px.

**`correct_motion.txt`: estimated here.** This scan has no `quali.mat`, so ESRF
never built one. [`estimate_motion.py`](estimate_motion.py) builds it from the
same raw material a `quali.mat` comes from: the scan writes 4003 frames for
ntheta=4000, and frames 4001 (ω=90) and 4002 (ω=0) are retakes of frames 2000
and 0 at the same piezo position. Registering each against its original gives
the drift accumulated over that stretch; three points, one quadratic,
mean-removed. Result: 3.36 px peak-to-peak horizontally, 5.95 px vertically.

```
python estimate_motion.py config_steps15.conf --path <mount>   # -> motion_estimate.png
cp correct_motion.txt <path>/<pfile>_1_/                       # step 3 reads it there
```

Validated on the 20 nm sibling first, which has both a `quali.mat` and ESRF's
real file: the vertical column correlates **+0.9996** with Peter's at scale
1.80 ≈ 2 — he writes bin-2 px, step 3 wants unbinned, hence `--bin-factor 2`.

Two honest caveats. The horizontal column is a sound measurement (it matches
`quali` at ω=0, +0.11 px against +0.130) but does **not** reproduce Peter's
`col0` (correlation −0.10); his comes from a different, unidentified quintic
source. And the 6 nm ω=90 retake is weak — peak 0.0062 against 0.0176 for ω=0,
carrying ±1.27 px — and being the middle point it sets the bow of the quadratic.
`--zero-x` writes a vertical-only file if you want that column backed out.

## Rotation axis

`rotation_center_shift = -37.50 ± 0.79` bin-0 detector pixels, measured
2026-08-26 with [`estimate_center.py`](estimate_center.py):

```
python estimate_center.py config_steps15.conf --shifts correct_motion.txt --pairs 1
```

**This value belongs to `correct_motion.txt`.** Against plain `correct.txt` the
answer is −39.81 ± 0.94. The +1.96 px between them is not a better measurement
but a reparametrisation: a constant added to the horizontal shift column is
exactly degenerate with the axis position, because the mirror flips one frame
and a common offset therefore *adds* instead of cancelling. Running
`estimate_motion.py --zero-x` and re-measuring returns −39.46, the `correct.txt`
value, to the digit. Never swap one of the two without the other.

`--pairs 1`: frame 4000 sits at exactly ω=180, so (0, 4000) is the only
genuinely opposed pair and the only one with a real correlation peak at this
voxel size (0.008–0.010, against ~0.006 for pairs 1–3, which are noise). Letting
them in gives −37.92 ± 2.06 — the same answer with four times the error bar.

Opposed projections are mirror images of each other about the axis, and Fresnel
propagation commutes with a mirror, so the relation holds on the **raw
holograms** — no reconstruction needed. The script mirrors frame 0 against frame
4000 (and the next pairs), cross-correlates, and converts the alignment shift to
`c − n/2`. See [`center_estimate.png`](center_estimate.png) and the script's
header, which spells out the two traps: the scan covers 180° total, so the
opposed partner of frame *j* is *ntheta−j*, **not** *j+2000*; and the residual
detector-fixed illumination is stronger than the sample and has to be subtracted
first or every correlation peaks at exactly (0, 0).

Only the two pairs closest to 180° survive at this voxel size (4 of 9 band
estimates kept) — pair *j* is off 180° by 0.045 + 0.09·*j* degrees, and the
resulting rotational smear costs more when the structure is 3.3× finer in
pixels. Two independent checks that the value is right:

* the same script on the 20 nm sibling returns −11.07 ± 0.94 px against that
  folder's independently derived −10.966;
* −11.07 px of 20 nm detector-plane offset is −36.9 px of 6 nm ones, i.e. the
  axis sits in the same physical place in both scans to within the error.

If reconstructions still look doubled or smeared, sweep it with
`step5_center_sweep.py`.

## Running it

```bash
# 0. look before leaping
python show_geometry.py config_steps15.conf      # derived geometry + per-level blocks
python scan_overview.py config_steps15.conf      # -> scan_overview.png
python estimate_motion.py config_steps15.conf     # -> motion_estimate.png + correct_motion.txt
python estimate_shrink.py config_steps15.conf    # -> shrink_estimate.png
python estimate_center.py config_steps15.conf --shifts correct_motion.txt --pairs 1

# 1. the whole pipeline, one job
#    Check the eagle quota first: start_level_rec=0 writes a 783 GB object
#    volume and 268 GB of bin-0 projections.
qsub polaris_run.sh
```

`polaris_run.sh` is a single PBS job with four sequential `mpiexec` calls,
plus two more that are commented out (`step0.py`, the optional NFP probe, and
`step5_center_sweep.py`, the rotation-centre sweep):

| # | stage | config | grid | iters | measured\* |
|---|---|---|---|---|---|
| 1 | `steps15.py` | `config_steps15.conf` | — | — | not timed |
| 2 | `step6.py` | `config_step6_bin2.conf` | 4×4, n=1024 | 0 → 512 | ~13 min |
| 3 | `step6.py` | `config_step6_bin1.conf` | 2×2, n=2048 | 512 → 768 | ~22 min |
| 4 | `step6.py` | `config_step6_bin0.conf` | 1×1, n=4096 | 768 → 1024 | ~1 h 47 min |

\* on the 20 nm sibling, which has the identical `n=4096`, a comparable `nobj`,
`ntheta=4000`, on 2 nodes / 8 ranks. Step 6 totals ~2 h 25 min; the walltime is
set to 12 h to leave room for the untimed steps 1–5, which read 128 GB of EDF
and write a few hundred GB back and are eagle-I/O bound rather than GPU bound.

To run only part of it — steps 1–5 already done, or resuming after a
preemption — comment out the leading `mpiexec` lines. Each line ends in
`|| exit $?`, so a failed stage stops the job instead of letting the next level
seed itself from a checkpoint that was never written.

All three levels share one `path_out`, so each picks up the previous level's
checkpoint (`start_iter`) and upsamples obj/prb/pos onto its own grid.
`checkpoint_step=32` divides every `start_iter` and every `niter−1`, so the
handoffs always exist; if a level is cut short by walltime, lower the next
level's `start_iter` to the last checkpoint that landed.

12 h does not fit the debug queue, so the job goes to `-q preemptable` (1–10
nodes, long walltime, evictable). Preemption is survivable: `checkpoint_step=32`
means a resubmit loses at most 32 iterations. Confirm the current queue limits
with `qstat -Qf` before changing this.

### Optional: measured probe

Unlike the 20 nm scan, this one has a near-field ptychography companion,
`Y350a_FT_large_rand_disp_nobin_006nm_NFPwS_1_` (50 frames of 4096², spy/spz
spanning ±79 px). Uncommenting the `step0.py` line in `polaris_run.sh`, which
reads [`config_step0.conf`](config_step0.conf), retrieves a probe from it;
then uncomment `prb_file` in `config_step6_bin2.conf` (bin 2
only — bins 1 and 0 inherit the probe from the previous checkpoint). Not
required: with `prb_file` unset the probe starts from the flat field and is
refined. Do **not** point step0 at the `_NFPwCA_1_` directory — that is the
coded-aperture scan and `step0.py` dies on its missing `cay`/`caz` motor keys.

## Files

| file | |
|---|---|
| `config_steps15.conf` | steps 1–5: geometry, binning ladder, rotation centre |
| `config_step6_bin{2,1,0}.conf` | the three BH levels; `rho[tp]=0` throughout |
| `config_step0.conf` | optional NFP probe retrieval |
| `polaris_run.sh` | PBS driver — steps15 + all three step-6 levels in one job |
| `estimate_center.py` | rotation axis from opposed 0/180° projections → `center_estimate.png` |
| `estimate_motion.py` | stage drift from the post-scan retakes → `correct_motion.txt`, `motion_estimate.png` |
| `estimate_shrink.py` | sample shrinkage from the same retakes → `shrink_estimate.png` |
| `scan_overview.py` | one-page acquisition summary → `scan_overview.png` |
| `show_geometry.py` | print derived geometry without launching MPI |
| `step5_center_sweep.py` | rotation-centre sweep producing TIFFs |
| `steps15.py`, `step6.py`, `step0.py` | pipeline (copies of `../Y350a_largedisp`) |

`estimate_center.py`, `estimate_motion.py`, `estimate_shrink.py`,
`scan_overview.py` and `show_geometry.py` are standalone —
numpy/h5py/fabio/matplotlib only, no cupy and no MPI — so they run on a login
node before steps15 has ever been started. They read the raw EDF tree directly.
Both figure scripts take `--path` to point at a local mount of eagle, e.g.

```bash
python scan_overview.py config_steps15.conf --path ~/eagle/vnikitin/20250604
```

## Open items

* `rho` is carried over from the 20 nm run and is a starting point, not a
  measurement. A single distance constrains the probe far less than a
  4-distance HT scan does, so consider `estimate_rho=True` at bin 2 (+19 %,
  ~96 extra iterations) before committing 513 iterations. The winners are logged
  but not checkpointed — copy them into the bin-1 and bin-0 configs by hand.
* `paganin=40`, set 2026-08-26. It had drifted apart across the configs —
  `config_steps15.conf` said 50 while the three step-6 files said 60, which
  would have failed at bin 2, since step 5 writes
  `/exchange/obj_init_re{paganin}_{bin}` and step 6 reads that exact name. All
  four now agree. The 20 nm run used 60 (ad-hoc for stained tissue, and the
  value Peter's own script used), so this is a deliberate departure.
* `nobj=4736`, raised from 4608 on 2026-08-26 so the grid covers the whole
  displacement sweep: the footprint is 4096 + 302 + 302 = 4699 px, which 4608
  clipped. 4736 = 74·64 and still bins cleanly. `mask_oob=1` is consequently
  close to a no-op now and is kept only as a guard for position refinement.
* Trim the 12 h walltime once steps 1-5 have been timed once.
