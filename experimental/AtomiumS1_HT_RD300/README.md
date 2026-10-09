# AtomiumS1_HT_RD300 — 4-distance HT, ±300 px random displacement, 4.5 nm voxels

`Atomium_S1_HT_4K_RD300_004p5nm_0004`, ESRF ID16A visit **blc17322**, on eagle at

```
/eagle/APS_IRI/vnikitin/20260829/AtomiumS1/
```

4000 projections over 180°, four propagation distances, 4096² frames at a 4.5 nm
voxel, ±300 detector px of commanded random displacement. This is the
**holotomography** scan of the sample whose **single-distance** scan lives in
[`../AtomiumS1_FT_RD300`](../AtomiumS1_FT_RD300) — same sample, same energy, same voxel, so the two
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
[`../AtomiumS1_FT_RD300`](../AtomiumS1_FT_RD300) rather than from `ctxl_HT` — ctxl's copy has only
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
> [`../AtomiumS1_FT_RD300`](../AtomiumS1_FT_RD300)'s `naburec/` states it. Borrowing a sibling's
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

```
shifts_final = random_shifts + motion_shifts + rhapp_shifts      (step 3)
             + the rotation axis, into the horizontal column     (step 4b)
```

> **The drift term is OFF as shipped: `motion_src=none`.** The retake
> estimator below is not validated — see "What the retake estimator actually
> does" — so step 3 carries `random + rhapp` only and step 7 owns the whole
> drift, which is what the pipeline did before the estimator existed. Set
> `motion_src=quali` in `config_steps15.conf` to switch it on, and only on a
> scan where `estimate_quali_motion.py --validate` has been checked against
> that scan's own `quali.mat`. The rhapp cache is named per setting
> (`rhapp_measured.npy` / `rhapp_measured_quali.npy`) because rhapp is
> measured with the drift already undone, so the two cannot share a file.

Three terms in step 3, and **all three are measured here** — no ESRF shift file
is read by the pipeline any more. `correct_motion.txt`, `correct_correct3D.txt`
and `rhapp.mat` are still on disk and still worth comparing against, but they
are inputs to `--validate`, not to the reconstruction.

| term | where it comes from | size on this scan |
|---|---|---|
| random | `<pfile>/projections/<pfile>_000k.txt`, read `[:ntheta, ::-1]` — **file column 0 is x, column 1 is y** | ±300 px |
| motion | `estimate_quali_motion.py`, from the post-scan retakes | ~16.5 raw px vertical |
| rhapp | `estimate_rhapp.py`, from the frames | 270–390 object px |
| axis | `estimate_axis_paganin.py` in step 4b, on bin-0 phase | −29.48 |

### Sample drift, from the post-scan retakes

Every plane's scan writes `ntheta+3` frames. The last three re-take omega
**180, 90, 0** immediately after the scan ends, so each one correlated against
its in-scan twin says how far the sample moved in between:

| point | in-scan frame | retake | value |
|---|---|---|---|
| omega 0 | `0` | `ntheta+2` | measured |
| omega 90 | `ntheta/2` | `ntheta+1` | measured |
| omega 180 | `ntheta` | — | `(0, 0)` **by construction** — it *is* the end of the scan |

Three points, so a quadratic fits them exactly and the order is a constant in
the code rather than a knob (`deg = min(npts-1, 2)`; a plane that loses a point
to the quality guards drops to degree 1). The curve is **mean removed**, which
is what reproduced Peter's `ref_v` exactly and what keeps the term from fighting
the step-4b axis — a mean-removed curve contributes no net translation. The fit
residual is meaningless here and is not reported as a quality number: an
exactly-determined quadratic passes through all three points by construction.
The honest error bar is the scatter across crops, propagated through the fit.

**On HT this is four separate scans**, taken at different times and drifting
independently, so it does not cancel in the plane-to-plane difference and
`estimate_rhapp` is handed the drift to undo along with the commanded move. A
motion term merely *added* to `shifts_final` would be re-measured by the rhapp
search and counted twice.

Three things make the correlation honest, and all three matter — the previous
`estimate_motion.py` returned 0.15 px against a true ~118 px drift on the FT
scan because it skipped the first:

1. **Flat-field correction.** Raw frames correlate on the static illumination,
   which does not move, and the peak locks at zero lag.
2. **A static template**, the mean of frames spread over the scan with the two
   being correlated nudged out of it, subtracted from both.
3. **The illumination peak is predicted and masked.** Undoing the commanded
   move puts the detector-fixed illumination at a *known* lag,
   `r_retake − r_scan`, so a disc there is masked out instead of hoping a narrow
   search window misses it. On an RD000 scan that lag *is* zero lag, the mask is
   skipped, and the point is flagged unverifiable.

The search window is 256 px, not ctxl's 40: FT_RD300's vertical drift reaches
~113 object px and a 40 px window would never see it.

#### Validation

`python estimate_quali_motion.py config_steps15.conf --validate` compares
against ESRF's own numbers without reconstructing anything, and this is the
check that settles the sign:

- `reference_motion.mat` `ref_v`/`ref_h` for the reference plane, compared
  sign-agnostically. **This scan: 8.27 binned = 16.54 raw px vertical**,
  2.31 binned = 4.6 raw horizontal — five times ctxl_HT's 1.62 raw px, far too
  large to leave out.
- `quali.mat` `corr_imagesafterscan`, the two measured points directly, on
  Peter's 2×2-binned grid so **×2**:

```
plane 1  corr_imagesafterscan = [ 0.588  -22.242 ;  2.602  -11.473 ; 0  0 ]
plane 2                         [-2.012  -18.061 ;  3.680  -10.135 ; 0  0 ]
```

> The horizontal column reaches −22 binned px, far more than sample drift. That
> is the same contamination ctxl_HT showed, where the horizontal column of
> `correct_motion.txt` carried ESRF's rotation correction rather than drift —
> and it is a reason to read our horizontal against `quali.mat` with suspicion,
> not a reason to distrust our own measurement. See
> [`../ctxl_HT_4K_RD300_007p5nm/README.md`](../ctxl_HT_4K_RD300_007p5nm/README.md),
> "`correct_correct3D.txt` — what ESRF's third shift file is".

Both curves are drawn in `shifts.png` next to the config, motion in red against
rhapp in blue, with the commanded displacement on its own axis.

#### What the retake estimator actually does — why `motion_src=none`

Measured on this scan, against all eight truth points (`quali.mat` ×2 on four
planes, two omegas each). Per-plane vertical, truth first:

| plane | truth, object px | measured |
|---|---|---|
| 1 | 44.48 | 51.98 |
| 2 | 37.67 | 35.90 |
| 3 | 16.52 | **1.39** |
| 4 | 15.07 | **2.00** |

Planes 1 and 2 come out roughly; planes 3 and 4 read ≈ 0 against a true 15–16
px, and the horizontal column reads ≈ 0 on six of the eight points. The failure
is a broad component at zero lag that outweighs the sample peak — at full phase
correlation its height is 380 against a sample peak of 54.

Four sweeps have tried to remove it and none worked:

| sweep | knob | result |
|---|---|---|
| `sweep_highpass.py` | unsharp σ 0…64 | **worse**: rms 11.85 at σ 0, 23.53 at σ 32 |
| `sweep_whiten.py` | `R/|R|^α`, zero-lag mask | rms 6.28, but only the mask helps, and the radius it needs (8 px) erases plane 4's true 5.96 px |
| `sweep_destripe.py` | per-row / per-column mean, template 0 vs 32 | **no effect**: all four destripe modes tie at rms 6.28, so it is not stripes |
| `sweep_tiles.py` | vote over 256/512/1024 px tiles | **every plane, every ω, every tile size votes 0.0**; 75–100 % of tiles within 3 px of zero, 0–4 % near truth |

A radius-8 zero mask barely moves the answer, so the thing beating the sample
is not a spike at zero but a broad blob centred there — which is also why the
high-pass, which should have killed a smooth blob, instead killed signal.

**The tile vote is the one that matters, and it says zero everywhere.** A
smooth blob should carry almost no contrast across a 256 px tile, so the
sample's fringes ought to win inside one. They do not: 75–100 % of tiles vote
within 3 px of zero, on every plane and both omegas, and only 0–4 % land near
Peter's number. That is consistent with two different worlds, and the tile
table cannot separate them:

- **(a)** the retake really does sit where its in-scan twin sits, so
  `corr_imagesafterscan` is not the lag between that pair and our truth is
  misassigned; or
- **(b)** something detector-fixed owns the correlation at *every* scale, in
  which case no filter was ever going to help.

#### It is world (b): the correlator cannot measure a shift it is *given*

[`sweep_selftest.py`](sweep_selftest.py) settles it. Two in-scan frames a few
indices apart are seconds apart, so the sample cannot have drifted — but their
commanded displacement is known exactly from the shift table, and on RD300 it
is tens of px. `Layout.read_proj` is a raw EDF/nxvds read with **no**
registration, so the frames really do differ by that amount.

Thirteen pairs, commanded |Δ| from 26 to 55 detector px:

```
 pl     j0     j1 |  cmd dy  cmd dx |  full dy  full dx |  med dy  med dx | %near cmd  %near 0
  1   3811   3812 |    6.67  -28.19 |      0.0      0.0 |     0.0     0.0 |        0%      94%
  1     37     38 |   30.47   38.18 |      0.0      0.0 |     0.0     0.0 |        0%      94%
  1   1591   1595 |   47.61   -5.13 |      0.0      0.0 |     0.0     0.0 |        0%      89%
  2    999   1003 |   -5.02  -25.48 |      0.0      0.0 |     0.0     0.0 |        0%     100%
  2   1813   1815 |   13.19   24.18 |      0.0      0.0 |     0.0     0.0 |        0%     100%
  2    851    853 |  -46.07   -6.24 |     51.0      0.0 |     0.0     0.0 |        0%      72%
  2   1295   1299 |  -37.03   19.14 |      0.0      0.0 |     0.0     0.0 |        0%      81%
  3    666    668 |  -21.92   -9.75 |      0.0      0.0 |     0.0     0.0 |        0%     100%
  3   2035   2036 |  -19.61   -7.97 |      0.0      0.0 |     0.0     0.0 |        0%     100%
  3     74     75 |   14.77   45.04 |      0.0      0.0 |     0.0     0.0 |        0%      81%
  3    814    817 |   22.65  -44.02 |      0.0      0.0 |     0.0     0.0 |        0%      83%
  4   1924   1926 |  -22.55    7.32 |      0.0      0.0 |     0.0     0.0 |        0%      89%
  4    481    483 |  -42.54    9.08 |     43.0     -1.0 |     0.0     0.0 |        0%      75%
  4   1036   1040 |  -34.30   23.94 |     32.0     -2.0 |     0.0     0.0 |        0%      81%
```

**The tile vote reads 0.0 on all thirteen — 0 % near the commanded value,
72–100 % near zero.** The whole-frame reads 0.0 on ten of thirteen. So the
correlation on these flat-fielded frames is pinned to zero lag across the whole
±50 px range, and the drift we were trying to measure (5–45 px) lies entirely
inside it. **No filter was ever going to fix that**, which is exactly what the
four sweeps above found the hard way.

The three non-zero whole-frame rows are a warning, not a rescue: all three have
a large *negative* commanded `dy` and come back with the **sign flipped**
(+51.0 against −46.07, +43.0 against −42.54, +32.0 against −34.30), while their
`dx` is never recovered at all (0, −1, −2 against −6.24, +9.08, +23.94). So the
earlier apparent agreement on the two largest retake points — −43.14 against a
truth of −44.48 — is not evidence that anything worked.

**Why `rhapp` is unaffected.** Its signal sits at 270–390 object px, far
outside the ±50 px zone the static owns, which is consistent with its own
independent validation against `rhapp.mat` (corr 0.875–0.998). The same
reasoning says anything measured on *phase* after Paganin — the step-4b axis —
or on a finished volume — step 7 — is likewise unaffected, because neither
correlates raw flat-fielded frames near zero lag.

`motion_src=none`, step 7 owns the drift, and **this line of work is closed**:
the limit is the data, not the filtering.

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
[`../AtomiumS1_FT_RD300`](../AtomiumS1_FT_RD300). No margin around the detector width at all;
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
not [`../AtomiumS1_FT_RD300`](../AtomiumS1_FT_RD300)'s 20 either, so the FT scan of this same
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
per-angle shift that minimises the entropy of the FBP:

```bash
python step7.py config_step6_bin2.conf --dry-run      # geometry and units
python step7.py config_step6_bin2.conf                # ~7 min on one A100, one GPU
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
cd ~/holotomocupy_gpu_reduced/experimental/AtomiumS1_HT_RD300   # /home/vvnikitin/... , not eagle
source ../polaris_env.sh
qsub polaris_run.sh            # 2 nodes, 18 h, preemptable: the full two-pass ladder
qstat -u $USER
```

`polaris_run.sh` runs the pre-flight itself, before the GPU healthcheck, and
aborts the job if it fails — so a missing shift file costs seconds, not a
node-hour. It is one literal `mpiexec` line per stage; comment out what you do
not want.

### The two-pass ladder

```
PASS 1   steps15
         step6  config_step6_bin2_nopos.conf   iter    0 -> 1024
         step6  config_step6_bin1_nopos.conf   iter 1024 -> 1280
STEP 7   step7  config_step6_bin1_nopos.conf   --iter 1280 --bin 1
         -> correct_correct3D_extra.txt, next to the configs
PASS 2   step6  config_step6_bin2.conf         iter    0 -> 1024   <- applies it
         step6  config_step6_bin1.conf         iter 1024 -> 1280
         step6  config_step6_bin0.conf         iter 1280 -> 1536
```

`_nopos` is pass 1's own set of configs, differing from pass 2's in exactly
three keys: the third `rho` component — the position direction — is **0, so
positions are frozen**; `correct3d_extra=0`; and `path_out` is
`*_rec6_paper_nopos` rather than `*_rec6_paper`. Pass 1 therefore produces a
volume whose drift has *not* been refined away, which is what step 7 needs in
order to measure it; the two trees do not collide.

Step 7's per-angle drift is only read at **bin 2** — the one rung with
`start_iter=0`, where `Reader.read_pos` adds it to `cshifts_final`. bin 1 and
bin 0 inherit it through the checkpoint; re-adding would double count.

**Re-submitting a finished run silently loses work.** On a second submission
`correct_correct3D_extra.txt` already exists, so pass 1 applies it; step 7 then
re-measures from an already-corrected volume and **overwrites** the file with
the residual. Step 6 *adds* the file rather than accumulating, so the original
correction is lost, not doubled — worse than the first run, and nothing errors.
`polaris_run.sh` refuses to start pass 1 when the file is present. After a
mid-pass-2 preemption, moving the file aside is the **wrong** answer: comment
out pass 1 and step 7 and resubmit pass 2 alone.

**Resume is manual.** `find_latest_checkpoint` globs
`checkpoint_*{start_iter:04}.h5` and returns `None` when `start_iter == 0`, so
it finds the checkpoint you *name*, not the newest on disk. After a preemption:
`ls {path_out}/checkpoints/`, set `start_iter` to the highest `checkpoint_NNNN`,
comment out the finished `mpiexec` lines, resubmit. The same keying is what
makes the deliberate bin2→bin1→bin0 handoff work, so it is not a bug to fix.

### `preemptable` is the only queue these fit, and all seven can start at once

A 2-node, 18 h job has nowhere else to go: `prod` has
`resources_min.nodect = 10`, and its `small` route caps walltime at 3 h.
`debug` is 2 nodes but 1 h. So a long wait behind a busy `preemptable` is
contention, not a misconfiguration — there is no faster queue to move to.

`preemptable`'s `max_run` is per **project** (`[p:PBS_GENERIC=10]`), with no
per-user cap on us, so nodes freeing up can release all seven together rather
than one at a time. Plan disk for the concurrent case.

### Disk: budget ≈ 57 TB for all seven, not 43

Two separate costs, and the checkpoint figure is only the second:

| | per scan | seven |
|---|---|---|
| steps 1–2 HDF5 | **2.1 TB** (FT) / **2.8 TB** (HT) | ≈ 14 TB (six convert) |
| step-6 checkpoints, if a run reaches bin 0 | ≈ 6.1 TB | ≈ 43 TB |

Measured on disk, not estimated. The h5 is large because step 1 writes
`/exchange/data{k}` as uint16 and step 2 **appends `/exchange/pdata{k}` as
float32 to the same file**, so it roughly triples. Only HT_RD300 skips this
(`start_step=3`); its h5 already holds `cshifts_final`, `shrink` and
`pref_0/1/2`.

Against ~94 TB free that leaves a ~37 TB margin. Checkpoints are never pruned,
and a bin-0 one is 550 GB, so the margin goes quickly once a run gets that far.

## Files

| file | what |
|---|---|
| [`esrf_layout.py`](esrf_layout.py) | filenames + geometry; this scan resolves as **`ewoks`** — geometry from the NXtomo, frames from the EDFs, because the virtual sources do not resolve |
| [`config_steps15.conf`](config_steps15.conf) | steps 1–5 |
| [`config_step6_bin{2,1,0}.conf`](config_step6_bin2.conf) | the BH ladder, pass 2 |
| [`config_step6_bin{2,1}_nopos.conf`](config_step6_bin2_nopos.conf) | the same, pass 1: positions frozen, `correct3d_extra=0`, own `path_out` |
| [`polaris_run.sh`](polaris_run.sh) | PBS job, one literal `mpiexec` line per stage |
| [`preflight.py`](preflight.py) | runs first in the job; fails it on a half-copied scan. Reads **no frames** — see `check_frames_nonzero.py` |
| [`steps15.py`](steps15.py) | steps 1–5 driver |
| [`step6.py`](step6.py) | BH reconstruction driver |
| [`step7.py`](step7.py) | per-angle drift from a finished volume → `correct_correct3D_extra.txt` |
| [`estimate_rhapp.py`](estimate_rhapp.py) | the rhapp offset, per plane (validated: corr 0.875–0.998 vs `rhapp.mat`) |
| [`estimate_center.py`](estimate_center.py) | rotation centre from opposed projections |
| [`estimate_axis_paganin.py`](estimate_axis_paganin.py) | the step-4b axis, measured on bin-0 Paganin phase |
| [`estimate_quali_motion.py`](estimate_quali_motion.py) | retake drift estimator — **does not work, `motion_src=none`**; see the section above |
| [`check_frames_nonzero.py`](check_frames_nonzero.py) | one read per plane across all seven scans: are the frames actually non-zero? |
| `sweep_{highpass,whiten,destripe,tiles,selftest}.py` | the five one-off diagnostics that closed the retake estimator; kept as the record, not run by anything |
