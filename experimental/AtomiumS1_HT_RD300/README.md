# Atomium S1 — 4-distance HT, 4.5 nm voxels

`Atomium_S1_HT_4K_RD300_004p5nm_0004`, ESRF ID16A visit **blc17322**, on eagle at

```
/eagle/APS_IRI/vnikitin/20260829/AtomiumS1/
```

4000 projections over 180°, four propagation distances, 4096² frames at a 4.5 nm
voxel, ±300 detector px of commanded random displacement. This is the
**holotomography** scan of the sample whose **single-distance** scan lives in
[`../AtomiumS1`](../AtomiumS1) — same sample, same energy, same voxel, so the two
are meant to be compared and several numbers here are deliberately taken from
that sibling rather than re-tuned.

These scripts are a copy of [`../ctxl_HT_4K_RD300_007p5nm`](../ctxl_HT_4K_RD300_007p5nm)
retargeted at this scan. **That README is the reference for how the pipeline
works and why each shift term exists**; this one records only what is different
here and what has actually been established on *this* scan.

> **Status, 2026-09-07: steps 1–2 can run, step 3 cannot.**
> The four EDF trees are complete on eagle and so is `correct_motion.txt`, but
> `nxtomomill` has not run, so `<pfile>/projections/<pfile>_000k.txt` — the
> commanded random displacement — does not exist yet. Nothing substitutes for
> it. See [What is still missing](#what-is-still-missing).

## What arrived, and what it says

The scan directory is EDF-only: frames and an `.info` sidecar per plane, no
NXtomo. That is the **`edfinfo`** flavour of [`esrf_layout.py`](esrf_layout.py),
and it is why this directory's `esrf_layout.py` was taken from
[`../AtomiumS1`](../AtomiumS1) rather than from `ctxl_HT` — ctxl's copy has only
`bliss`, `ewoks` and `nxvds`. Everything geometric below is read out of the
`.info` files by that flavour; nothing is retyped into the configs except
`rotation_center_shift`, `nobj` and `paganin`.

```
<pfile>_1_ .. _4_/   4066 EDF each  + angles_file.txt + <name>.info + quali.mat
<pfile>_3_/          + correct_motion.txt          (4004 rows)
<pfile>_/            Peter's ESRF reconstruction drop, below
<pfile>/projections/ ABSENT   <-- the blocker
```

Step 3 confirms `proj[0]`, `proj[3999]`, 20 flats and 20 darks at
every distance. `angles_file.txt` has 4003 rows: 4000 projections from 0 to
−180 in 0.045° steps, then the three post-scan retakes at 180 / 90 / 0.

### Geometry, from the `.info` sidecars

| plane | SourceDistance (mm) | Distance (mm) | PixelSize (µm) | norm_mag |
|---|---|---|---|---|
| 1 | −3.69813 | 1209.300 | 0.0045000 | 1.000000 |
| 2 | −3.85678 | 1209.140 | 0.0046931 | 0.958865 |
| 3 | −4.49138 | 1208.510 | 0.0054653 | 0.823384 |
| 4 | −5.80903 | 1207.190 | 0.0070686 | 0.636617 |

Energy 17.1 keV, detector pixel (`Optic_used`) 1.47601 µm, `Dim` 4096.

Those four normalised magnifications reproduce
[`../ctxl_HT_4K_RD300_007p5nm`](../ctxl_HT_4K_RD300_007p5nm)'s to five decimals.
The two scans were run with the same optic at the same four plane positions, so
anything ctxl_HT established about the *geometry* — the rhapp bin factor above
all — carries over.

```bash
python show_geometry.py config_steps15.conf     # prints the table and the per-level blocks
```

## Peter's drop

`<pfile>_/` holds two ESRF reconstructions of this scan — an older PyHST one and
the nabu one that superseded it on 2026-09-08 — and the shift files they were
built from:

| file | what it gave us |
|---|---|
| `rhapp.mat` | 2×4×4003, −94.60 … +147.48 **binned** px, stamps `pixelsize` 8.99998 nm → `rhapp_bin=2` |
| `rhappnofit.mat` | the unfitted residual; `res_shift_sigma_h` [0.919 0.891 0.879], `sigma_v` [0.790 0.886 0.973] binned px |
| `reference_motion.mat` | `reference_plane = 3` → **`ref_dist=2`**; `ref_v` ptp 8.2691, `ref_h` ptp 2.3065 binned px |
| `ht_<pfile>.m` | his driver: `nvue` 4000, `bin_factor` 2, `reference_plane` 3, `delta_beta` 150, `pixelsize_detector` 1.47601e-06 |
| `<pfile>_rec_.info` | `Dim` 2048, `PixelSize` 0.00899998 µm, `nb_of_planes` 4, `delta_beta` 150 |
| `<pfile>_rec_.par` | PyHST, `ROTATION_AXIS_POSITION` 1046.598214 on `NUM_IMAGE_1` 2048 → **stale**, see below |
| `naburec/*.conf` | nabu, `rotation_axis_position = 1008.76` → **the axis we use**, below |
| `correct_correct3D.txt` | his own, 4001 rows, dropped 2026-09-08 with `naburec/` |
| `<pfile>_rec_.nx` | his 67 GB volume, for comparison |

Also 29 sparse phase-map EDFs, `angles_file.txt`, and the octave/sbatch
scaffolding. `shift_show/` and `out_err/` are empty.

Every one of the `.mat` files is Octave **text** format — read them with
`holotomocupy.reader.load_octave_text_mat(path, varname)`, and note that
`varname` is required.

### The rotation axis: `−29.48`, from Peter's nabu run

`<pfile>_/naburec/` landed on **2026-09-08 07:57**, and it is the source used.
Six of its seven configs — `nabu.conf`, `nabu_final.conf`, `nabu_final_cm.conf`
and the `_even`/`_odd` pair, plus `nabu_correct3D.conf` — agree on

```
rotation_axis_position = 1008.76
```

(the seventh, `nabu_correct3D_v.conf`, is the vertical pass and says
1009.082309). nabu counts from a **0-based** centre, `(N−1)/2`, and his grid is
2×2 binned relative to ours, so

```
rcs = (1008.76 − (2048−1)/2) × 2 = −29.48  →  rotation_center_shift = −29.48
```

The grid is unpadded — `Dim_1 = 2048 = 4096/2` exactly, unlike ctxl_HT's
π/2-padded 3216 — so there is no crop to account for and the arithmetic is not
in doubt. `nabu_final_cm.conf` is the config whose reconstruction actually ran,
and it reads `../correct_correct3D.txt`, the same file step 3 reads: axis and
correction are the **matched pair**, which is the whole reason to take both from
the same source.

`esrf_meta.nabu_axis()` prefers `naburec/` over the `.par` and both `steps15.py`
re-derives the number and warns at more than 0.5 px
disagreement, so what is in the configs is checked rather than trusted.

> **It was `+44.20`, from the PyHST `.par`, and that was wrong by 73.7 px.**
> ```
> rcs = (1046.598214 − (2048+1)/2) × 2 = +44.196…      (.par is 1-based)
> ```
> The `.par` is dated 2026-09-07 13:18, written by the octave driver *before*
> the `horizontal_search = true` alignment pass that produced `naburec/`; the
> nabu number is that pass's answer. Every other scan of the beamtime has the
> two sources within a few pixels of each other —
>
> | scan | PyHST `.par` | nabu | diff |
> |---|---|---|---|
> | AtomiumS1 FT | +10.53 | +10.74 | 0.21 px |
> | ctxl_HT 4K | −18.87 | −15.77 | 3.10 px |
> | AtomiumS1 HT | +44.20 | −29.48 | **73.68 px** |
>
> — so this `.par` is the identifiable outlier, not a convention error on our
> side. **Do not restore it.**

> **The FT sibling's axis is +10.74, and it is not used here.**
> [`../AtomiumS1`](../AtomiumS1)'s `naburec/` states it. Borrowing a sibling's
> axis is the rule when a scan has nothing of its own — [`../ctxl_FT`](../ctxl_FT_4K_RD300_007p5nm)
> does exactly that — but this scan now has its own nabu number, and the 40.2 px
> between the two is a real difference in where the stage sat: ctxl_HT and
> ctxl_FT are 25.6 px apart the same way, a day apart.

Re-quoting the axis costs only `start_step=5` — it does not enter steps 3–4.
**It must be the same in all four configs.** `rotation_center_shift` lands in
the same slot as `cshifts_final[..., 1]` at every distance, so only the sum is
defined; a mismatch between `config_steps15.conf` and a `config_step6_bin*.conf`
silently shifts the object between levels.

## Shifts

`shifts_final = random_shifts + rhapp_shifts + motion_shifts + correct3d_shifts`,
and step 3 reads each file as `np.loadtxt(path)[:ntheta, ::-1]` — **file column 0
is x, column 1 is y**.

| term | source | state |
|---|---|---|
| random | `<pfile>/projections/<pfile>_000k.txt` | **MISSING** |
| rhapp | `<pfile>_/rhapp.mat`, `rhapp_bin=2` | present |
| motion | `<pfile>_3_/correct_motion.txt` | present, 4004 rows |
| correct3d | `<pfile>_/correct_correct3D.txt`, `correct3d_bin=2` | present, **Peter's own** since 2026-09-08, 4001 rows |

### The drift is five times ctxl_HT's

`reference_motion.mat` carries the same drift as `correct_motion.txt`, negated,
so its size was known before the file arrived: **8.27 binned px = 16.5 raw px
vertical**, 2.31 binned = 4.6 raw px horizontal, against ctxl_HT's 1.62 raw px.
Far too large to leave out.

Step 3 re-derives it as
`correct_motion − random[ref_dist]` and compare against `ref_v`/`ref_h` — that
is the check that the file was read in the right units and column order — but
the subtraction needs the random displacement, so the comparison cannot be made
yet.

The retakes it is fitted from are in each plane's `quali.mat`; for the two that
have been looked at (bin-2 px, `rot_positions` [0 90 180], index [0 2000 4000]):

```
plane 1  corr_imagesafterscan = [ 0.588  -22.242 ;  2.602  -11.473 ; 0  0 ]
plane 2                         [-2.012  -18.061 ;  3.680  -10.135 ; 0  0 ]
```

`python estimate_motion.py config_steps15.conf --validate` re-measures the drift
from those and writes an independent `./correct_motion.txt`, worth running as a
cross-check the way it was on ctxl_HT. Note the horizontal column reaches −22
binned px, far more than sample drift — that is the same contamination ctxl_HT
showed, where the horizontal column of `correct_motion.txt` turned out to carry
ESRF's rotation correction rather than drift. Read
[`../ctxl_HT_4K_RD300_007p5nm/README.md`](../ctxl_HT_4K_RD300_007p5nm/README.md),
"`correct_correct3D.txt` — what ESRF's third shift file is", before acting on it.

### `correct_correct3D.txt` — Peter's own, since 2026-09-08

For a day this slot held a **transplant**: ctxl_HT's file, copied row for row and
multiplied by 7.5/4.5 = 1.666667. On **2026-09-08 07:57** Peter dropped the real
one for this scan, together with the `naburec/` that reconstructed with it, and
it replaced the transplant in place. Nothing in the configs changed —
`correct3d_bin=2` still holds, because `<pfile>_rec_.info` still says
`PixelSize 0.00899998` µm against our 4.5 nm voxel.

| | rows | ptp binned | ptp raw | horizontal mean |
|---|---|---|---|---|
| Peter's, this scan | 4001 | 25.269 × 21.479 | 50.54 × 42.96 | +1.7135 binned = **+3.427 raw** |
| the transplant it replaced | 4001 | — | 68.3 × 41.1 | +2.73 raw |

**The transplant was wrong in the column it was most confident about.** Fitting
Peter's real file against the ctxl_HT source it was made from:

| column | best scale | assumed | corr | residual |
|---|---|---|---|---|
| horizontal | **1.4148** | 1.66667 | 0.9740 | 1.85 px rms, of a 25 px range |
| vertical | **1.7445** | 1.66667 | 0.9981 | 0.26 px rms |

The earlier note here argued the opposite — that the horizontal column was one
physical curve in two unit systems (a ctxl_HT ↔ AtomiumS1 FT fit gave scale
1.66667 to 0.0040 px rms) and so was the safe one to scale, while the vertical
carried a few per cent of per-scan content. This scan says the horizontal was
18 % off and the vertical within 5 %. Correct3D does repeat between scans on
this stage, but not to the precision that fit suggested — **do not transplant it
again when a real file exists.**

> **It moves the effective axis, and that is already accounted for.** The
> horizontal mean is +3.427 raw px, and a constant in that column is degenerate
> with `rotation_center_shift`. The `−29.48` above comes from the same nabu run
> that *read this file*, so the pair is consistent and nothing needs adding.

`correct_correct3D_v.txt` arrived beside it. It is not read and does not need to
be: its column 0 is identically zero and its column 1 is identical to the main
file's — it is the vertical pass's own copy.

[`estimate_correct3d.py`](estimate_correct3d.py) can still fit one from the
refined positions once step 6 has run far enough, now as a cross-check of
Peter's file rather than a replacement for a transplant.

## Shrinkage: not measured, and off

`rho[tp] = 0` in all three step-6 configs, and there is no `shrink_list.mat`, so
`init_tp_from_shrink()` starts the model at A = B = 0 and it stays there. All
four `quali.mat` files are here, so
`python estimate_shrink.py config_steps15.conf` can be run as soon as steps 1–3
have; put the answer in that slot only if it comes back non-zero. ctxl_HT, the
other 2026 HT scan, measured |A| < 420 ppm (2σ) — under a pixel at the frame
edge.

Leaving it free is the risk, not the caution: on `../Y350a_largedisp*` a
non-zero `rho[tp]` let the optimizer invent ~2200 ppm of edge displacement, 8×
that scan's upper limit, by absorbing displacement the position refinement
should own — [`../Y350a_largedisp_006nm/README.md`](../Y350a_largedisp_006nm/README.md)
shows the 15–24 % loss. With ±300 px of random displacement this scan is exposed
to exactly that failure mode.

## `nobj` = 4096 — matched to the FT scan

`nobj = nzobj = n` at every level (4096 / 2048 / 1024), the same as
[`../AtomiumS1`](../AtomiumS1). No margin around the detector width at all;
`mask_oob=1` drops whatever falls outside the grid.

This is not the same trade the FT scan makes. Over there `ndist = 1` and
`norm_mag = 1`, so the only thing the grid has to absorb is the ±300 px sweep.
Here four planes back-map onto very different object widths:

| plane | footprint = 4096/norm_mag | kept, `nobj` 5056 | kept, `nobj` 4096 |
|---|---|---|---|
| 1 | 4096.0 | 1.00 | 1.00 → 0.86 |
| 2 | 4271.7 | 1.00 | 0.92 → 0.78 |
| 3 | 4974.6 | 1.00 → 0.87 | 0.68 → 0.56 |
| 4 | 6434.0 | 0.62 → 0.51 | 0.40 → 0.32 |

(2-D kept fraction, static → at the ends of the sweep.) So this masks roughly a
third of the data rather than roughly a tenth. It buys a 1.9× smaller bin-0
volume and a reconstruction directly comparable with the FT one, voxel for
voxel. `Rec._build_data_mask` logs the real fractions at startup — read them on
the first bin-2 run.

**The fallback is 5056**, the sweep-sized value ctxl_HT runs:
`4096 + 2 × 300/0.636617 = 5038.5`, rounded up to `79 × 64`. If the outer band
of the field degrades — it is the part only seen at some angles once the margin
is gone — put 5056 back here and 1264 / 2528 / 5056 in the three step-6 configs.

Note the pre-flight **will** report the reach outside `nobj/2` at 4096. That is
expected here, not a fault; `mask_oob` is what handles it.

## `paganin` = 35

Set by instruction, and it lands on the energy-scaled value: Peter's octave
driver for the other Atomium scans uses 120 at 33.35 keV, and delta/beta scales
as E², giving 120×(17.1/33.35)² = 32 at this energy.

It is not Peter's number *for this scan* — `ht_<pfile>.m` sets
`delta_beta = 150` and calls it "ad-hoc" in the comment beside it — and it is
not [`../AtomiumS1`](../AtomiumS1)'s 20 either, so the FT scan of this same
sample is started from a slightly different filter. That matters less here than
it would at ndist=1: with four distances the transport filter is doing much less
of the work, and step 6 refines away from the init regardless. For reference,
ctxl_HT uses 60 on stained cortex at the same energy.

It **must** match in all four configs: step 5 writes
`/exchange/obj_init_re{paganin}_{bin}` and step 6 reads that exact name.
Re-running step 5 alone is cheap (`start_step=5`), so trying 20 costs almost
nothing:

```bash
python show_slices.py config_steps15.conf 2
```

## The BH ladder

| config | bin | nz = n | nobj | start_iter | niter | new iters |
|---|---|---|---|---|---|---|
| `config_step6_bin2.conf` | 2 | 1024 | 1024 | 0 | 1025 | 1025 |
| `config_step6_bin1.conf` | 1 | 2048 | 2048 | 1024 | 1281 | 257 |
| `config_step6_bin0.conf` | 0 | 4096 | 4096 | 1280 | 1537 | 257 |

The same ladder ctxl_HT runs. All three share one `path_out`
(`..._0004_rec6`) and each level seeds itself from
`checkpoints/checkpoint_{start_iter:04d}.h5`, so they must be run in order and a
level that dies leaves the next one nothing to start from.

`rho = 1.0, 0.05, 0.04→0.02→0.01, 0` is inherited from ctxl_HT, not tuned here.
The FT sibling runs `1.0, 0.0125, 0.16, 0.0`; its `rho[pos]` is not transferable
(one distance, different position problem), but its 4× smaller `rho[prb]` may be
worth trying if the probe misbehaves.

**`estimate_rho=False`**, unlike ctxl_HT. The coordinate search costs +96 silent
iterations (+9 % at bin 2), and the FT scan of this same sample reconstructs
fine on a fixed `rho`, so it is not worth spending here.

## Nothing is missing any more

The last gap closed on **2026-09-08**: `Atomium_S1_HT_4K_RD300_004p5nm_0006/`
was synced, and with it the commanded random displacement of planes 3 and 4.

**Planes 3 and 4 were re-taken under scan number 0006.** They are not under this
scan's `projections/` at all, and the only thing on disk that says so is the
master `<pfile>.nx`:

```
entry_0001 -> ..._0004_0001.nx                              (here)
entry_0002 -> ..._0004_0002.nx                              (here)
entry_0003 -> ../../..._0006/projections/..._0006_0003.nx
entry_0004 -> ../../..._0006/projections/..._0006_0004.nx
```

`Layout` follows that table (`nx_master_links()`), so `nxfiles[k]` — and
`shift_source(k)` with it — resolve per plane rather than by pattern. Two
things depend on it:

- **The displacement cannot be borrowed.** Each plane's is an independent
  ±300 px draw: planes 1/2/3 differ from one another by an rms of 244 px, and
  ctxl_HT's four do the same. Reading plane 4's out of this directory would
  have silently used another plane's sweep.
- **`..._0004_0003.nx` is a trap.** It is an aborted-run leftover — 1083
  projections, 0…48.69° — sitting where plane 3's file "should" be. The master
  points past it; a pattern match would not.

Two guards were added alongside. `ewoks` now requires the NXtomo set to be
**complete and full-length** rather than winning on `nxfiles[0]` alone
(`nx_missing`, `nx_short`, via `nx_nproj()`), falling back to `edfinfo`
otherwise; and `steps15.py` prints every re-link
and every demotion, so neither is silent.

> **The NXtomos never held pixels for this sample.** All 44 virtual sources of
> every AtomiumS1 NXtomo point at `RAW_DATA/Atomium_S1/.../balor_*.h5`, which
> was never synced — `RAW_DATA/` here holds only `ctxl`. That is harmless:
> `ewoks` reads the NXtomo for geometry only and takes frames from the EDFs in
> `<pfile>_k_/`, which are complete for all four planes (4066 files each,
> `.info` `TOMO_N=4000`, acquired Sep 07 12:32–12:53). Only the `nxvds` flavour
> reads pixels out of a NXtomo, and this scan is not it.

### The pre-flight passes

```bash
```

**17 ok, 7 notes, 0 warnings, 0 bad**, exit 0 — against 13 ok / 3 notes /
4 warnings / 3 bad before the drop. The three BADs were all downstream of the
zeroed displacement sweep, and the four WARNs were the `edfinfo` flavour's "not
an independent check (no NXtomo present)", one per plane.

The check that could not be made until the displacement arrived now makes it:

```
[  ok  ] drift vertical   vs reference_motion.mat: max|diff| 0.000054 (ESRF negated)
[  ok  ] drift horizontal vs reference_motion.mat: max|diff| 0.000046 (ESRF negated)
```

`correct_motion.txt − random[ref_dist]` reproduces ESRF's own `ref_v`/`ref_h` to
5·10⁻⁵ px. It runs through `ref_dist = 2`, i.e. plane 3 — the re-linked 0006
file — so it is an independent confirmation that the 0006 displacements are the
right ones for these frames. The axis passes too:
`rotation_center_shift -29.4800 agrees with nabu to 0.0000 px`, with the stale
PyHST `.par` demoted to a note (`73.68 px from nabu`).

The seven notes are all expected: four "4003 rows for ntheta 4000" (the
post-scan retakes), "correct_motion.txt has 4004 rows", "correct_correct3D.txt
has 4001 rows" (the 180° repeat), and the PyHST cross-check. It still prints the
grid reach exceeding `nobj/2` at every distance (by 367 / 653 / 1127 / 1790 px)
— also deliberate, see [`nobj`](#nobj--4096--matched-to-the-ft-scan).

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
cd ~/holotomocupy_gpu_reduced/experimental/AtomiumS1_HT   # /home/vvnikitin/... , not eagle
source ../polaris_env.sh
qsub polaris_run.sh                                   # steps15 + bin2 + bin1 + bin0
qstat -u $USER
```

`polaris_run.sh` runs the pre-flight itself, before the GPU healthcheck, and
aborts the job if it fails — so a missing shift file costs seconds, not a
node-hour. It is one literal `mpiexec` line per stage; comment out what you do
not want.

To run steps 1–2 now and leave the rest for when the displacement arrives: set
`start_step=1` in `config_steps15.conf` and comment out the three step-6 lines.

## Files

| file | what |
|---|---|
| [`esrf_layout.py`](esrf_layout.py) | filenames + geometry; **the `edfinfo` flavour is what reads this scan** |
| [`config_steps15.conf`](config_steps15.conf) | steps 1–5 |
| [`config_step6_bin{2,1,0}.conf`](config_step6_bin2.conf) | the BH ladder |
| [`polaris_run.sh`](polaris_run.sh) | PBS job, one `mpiexec` line per stage |
| [`show_geometry.py`](show_geometry.py) | derived geometry and the per-level config blocks |
| [`scan_overview.py`](scan_overview.py) | overview figure |
| [`estimate_center.py`](estimate_center.py) | rotation centre from opposed projections |
| [`estimate_motion.py`](estimate_motion.py) | drift from the post-scan retakes |
| [`estimate_shrink.py`](estimate_shrink.py) | shrinkage from the post-scan retakes |
| [`estimate_correct3d.py`](estimate_correct3d.py) | fits a `correct_correct3D.txt` from the refined positions |
| [`step5_center_sweep.py`](step5_center_sweep.py) | centre refinement from the FBP volume |
| [`show_iter.py`](show_iter.py) | one step-6 checkpoint: slices, probe, positions, convergence |
| [`show_pos_errors.py`](show_pos_errors.py) | total position correction across the ladder |
| [`diagnose_positions.py`](diagnose_positions.py) | splits it into runout, drift and displacement-stage error |
| [`steps15.py`](steps15.py) | steps 1–5 driver |
| [`step6.py`](step6.py) | BH reconstruction driver |
