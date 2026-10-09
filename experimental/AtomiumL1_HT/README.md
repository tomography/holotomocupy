# AtomiumL1 — HT scan, 14 nm, 4 distances

ESRF ID16A, proposal `ihmi1591` (`ihmi1591-AtomiumS2-…`), acquired 2025-06-08.
Raw data on **tomo5** at
`/data3/vnikitin/ESRF/atomium/20250607/AtomiumL1/AtomiumL1_HT_014nm_[1-4]_/`,
plus the aux directory `AtomiumL1_HT_014nm_/` (rhapp.mat and the ESRF PyHST
reconstruction).

Structured exactly like `../Y350a_HT`: `steps15.py` (convert → preprocess →
shifts → binned → Paganin+FBP) then a three-level `step6.py` ladder.

## Verified geometry (`python show_geometry.py config_steps15.conf`, 2026-08-22)

| | |
|---|---|
| ndist / ntheta | 4 / 1800 |
| detector | 2048 × 2048, pixel 2.9520 µm |
| energy | 33.35 keV |
| focus→detector | 1.289 m (sx0 = 1.280 mm) |
| z1 | 6.113 / 6.375 / 7.424 / 9.602 mm |
| norm. magnification | 1 / 0.95890 / 0.82341 / 0.63664 |
| voxel size | 14.00 / 14.60 / 17.00 / 21.99 nm |
| nref / ndark | 21 / 20 |
| n / nobj (bin 0) | 2048 / 3264 |

All 1800 projections, 21 ref pairs and 20 darks are present in every distance
directory (checked frame by frame).

Relative dose (`python dose.py config_steps15.conf`): the four distances cost
**3.0028** near-distance projections per angle, 5414 in total.

## Running

On **tomo5** (4 x A100-SXM4-40GB; `/data3` is not mounted on handyn):

```bash
./local_run.sh                                            # steps 1-5, all local GPUs
NP=4 ./local_run.sh
SCRIPT=step6.py CONFIG=config_step6_bin2.conf ./local_run.sh
SCRIPT=step6.py CONFIG=config_step6_bin1.conf ./local_run.sh
SCRIPT=step6.py CONFIG=config_step6_bin0.conf ./local_run.sh
```

`polaris_run.sh` is kept for the case where the 59 GB of EDF is staged to
eagle first — the configs point at `/data3` on tomo5, so it needs the paths
repointed.

Ladder: bin 2 (n=512, nobj=512, iters 0–1024) → bin 1 (1024/1024, 1024–1280)
→ bin 0 (2048/2048, 1280–1536), all sharing `…_rec6` so each level resumes
from the previous checkpoint. Disk: ~600 GB.

## Estimating the sample drift during the scan

`estimate_drift.py` measures how far the sample wandered over the 1800
projections, on top of the ±90 px displacement step 3 already knows about and
removes. It redoes the step-5 Paganin pass in memory for **every** angle (step 5
itself only stores every 10th to `{pfile}_proj.h5`), takes the centre of mass of
each phase projection, subtracts projection 0, and fits polynomials to the two
resulting curves. It reads its geometry back out of `{path_out}/{pfile}.h5`, so
only steps 1–4 have to have run, and flags come after the config:

```bash
mpirun -n 4 ./set_affinity_gpu.sh python estimate_drift.py config_steps15.conf
```

~15 s wall on 4 A100s at bin 2, all 1800 angles, everything at its default. It
writes `{path_out}/drift_bin2/`: `drift_bin2.png` (the six panels below),
`drift_bin2.txt` and `drift_bin2_profiles.npz`. A copy of the figure is checked
in next to this README:

![measured drift and its polynomial fits](drift_bin2.png)

The script is shared with `../AtomiumL1_largedisp`, whose README documents what
the centroid is taken *of* and why — briefly: **not** of the phase but of
`|∇φ|`, because the Paganin phase carries a large low-frequency background and a
per-frame air level that jumps between neighbouring projections, and every
weight built on that level either leaks the applied displacement into the answer
or scatters by tens of px. Keep the two copies in sync; develop in `largedisp`,
which is the harder case, and copy here.

### What it found for this scan

Measured (bin 2, all 1800 angles, defaults, **unbinned** px):

| | dy | dx |
|---|---|---|
| raw centroid peak-to-peak | 45.7 | 33.4 |
| off-axis orbit amplitude | — | 15.8 |
| **drift ptp, deg 2** | **20.0** | **0.5** |
| deg 3 | 19.6 | 6.3 |
| deg 5 | 26.4 | 8.5 |
| rms residual, deg 2 / 3 / 5 | 7.84 / 6.61 / 4.68 | 3.10 / 2.90 / 2.55 |
| point-to-point noise | 2.27 | 1.53 |

Both acceptance tests pass: the leak against the applied `cshifts_final`, which
step 3 already removed and which a correct estimator must therefore ignore, is
corr −0.02 / slope +0.013 on dy and −0.04 / +0.015 on dx; and only 1.2 % of the
sample's edge mass (worst 2.0 %) falls outside the measured window, against
6.9 % / 21.9 % on the largedisp scan — the ±90 px sweep here is small enough
that `nobj = 2048` really does contain the object at every angle.

**The sample drifted about 20 px vertically and essentially not at all
horizontally**, over 180° — twice the largedisp scan's 10 px, which is what you
would expect from a scan that takes about three times as long. As there, almost
all the raw horizontal motion is an `A·cos θ + B·sin θ` orbit (15.8 px, so the
centre of mass sits 15.8 px off the rotation axis), fitted and removed before
the polynomial rather than jointly with it.

### The polynomial is the wrong model here, and the structure function says so

Unlike largedisp, the rms residual keeps falling as the degree goes up
(7.84 → 6.61 → 4.68 px on dy) — but not because there is quintic structure:
degrees 7 and 9 buy 4.68 → 4.64 → 4.58 and then stop, while the fitted drift
keeps growing. The reason is in the structure function the script now prints,
the rms step between projections a given angle apart:

| lag [°] | 0.1 | 0.2 | 0.5 | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| **HT** dy | 2.27 | 2.52 | 3.30 | 3.86 | 4.22 | 4.56 | 5.53 | 7.32 | 10.82 |
| **HT** dx | 1.53 | 1.51 | 1.59 | 1.71 | 1.91 | 2.45 | 2.84 | 3.72 | 6.40 |
| largedisp dy | 3.63 | 3.64 | 3.81 | 3.83 | 4.01 | 4.15 | 4.55 | 4.84 | 4.85 |
| largedisp dx | 3.53 | 3.45 | 3.53 | 3.46 | 3.58 | 4.08 | 4.44 | 5.13 | 6.02 |

On largedisp the curve is **flat** from 0.1° to 5°: on that timescale there is
nothing but estimator noise, and everything above it is the smooth drift the
polynomial fits. On HT it **climbs from the very first lag** — 2.3 px at 0.1°,
3.9 px at 1°, 5.5 px at 10°. The sample was moving at every timescale, a random
walk rather than a drift, and no polynomial of any degree can represent the part
that is faster than the fit. So the 4.7 px residual is *motion*, not noise, and
the deg-2 number below should be read as "about 20 px of slow drift, with
another ~4 px rms of wander underneath it that a per-angle correction would have
to catch".

That is a stronger argument for letting BH refine the positions on this scan
than on largedisp, where the sample really does just drift.

### Feeding the drift back into the reconstruction

`--export-correct3d` writes the fit where step 3 of `steps15.py` looks for it,
`{path}/{pfile}_/correct_correct3D.txt`, and the next run of steps 1–5 folds it
into `shifts_final` along with the random, rhapp and motion shifts:

```bash
mpirun -n 4 ./set_affinity_gpu.sh python estimate_drift.py config_steps15.conf \
    --export-correct3d              # --export-deg 5 by default
```

`../AtomiumL1_largedisp/README.md` documents the file in full; the three things
worth repeating are that the export is the degree-5 fit to the **centre of mass
itself** (the left-hand column of the figure), evaluated at every angle; that it
is a **plain** polynomial with no orbit term taken out (`A·cos θ + B·sin θ` in x
is exactly a rigid translation of the object, so keeping it only re-centres the
volume laterally); and that **nothing is added to or taken off it** — no
re-zeroing at projection 0, no mean-centring, just the fit as `lstsq` returned
it. The columns are **x then y**, because step 3 reads the file as
`np.loadtxt(...)[:ntheta, ::-1]`.

The exported curve is the **black dashed line** on the two left-hand panels of
`drift_bin2.png`, drawn exactly as written.

Written 2026-08-23: ptp x 22.67 px, y 26.37 px; largest value in the file x
18.67, y 15.09 px; rms residual 2.55 / 4.68 px. The x amplitude is larger than
the orbit-free 8.5 px in the table above because the 15.8 px orbit is now inside
the exported curve; y is unchanged at 26.4 px. Given the structure function,
this file removes the slow part and leaves the ~4.7 px rms wander for BH's
position refinement.

Two notes on how this particular file was produced. A `steps15.py` run held the
h5 open when the un-centred version was written, so instead of re-measuring, the
constant was put back into the mean-centred export written earlier the same day
(x −5.3198, y +2.8229 px) — cross-checked against the polynomial coefficients
stored in its own header, the two routes agreeing to 1e-5 px. The previous file
is beside it as `correct_correct3D.txt.meancentred.bak`, and `drift_bin2.png`
in this directory is still the older, mean-centred figure; one 15 s rerun of
`estimate_drift.py --export-correct3d --export-force` once the h5 is free
regenerates both. And because that `steps15.py` run was already past step 3 when
the file changed, **its** output carries the mean-centred shifts, not these.

## Open items

1. **`paganin=120` is inherited, not measured.** Taken from the `AtomiumS2`
   configs (same proposal, same kind of sample). Check it on a bin-2 step-5 FBP
   slice before the long runs.
2. **`rotation_center_shift=-14.375964` assumes PyHST's pixel convention is
   ours.** It comes from `ROTATION_AXIS_POSITION = 1009.624036` on the 2048-wide
   grid in `AtomiumL1_HT_014nm_/AtomiumL1_HT_014nm_rec_.par`. That is a real
   measurement of this scan, but the convention has only been checked for
   Y350a. One bin-2 FBP slice settles it.
3. **No usable NFP probe.** The only companions are `*_NFPwCA_[1-4]_`
   (coded aperture): they carry `cay`/`caz`, not the `spy`/`spz` that
   `step0.py` reads, and those span ~7 px over 50 frames. `config_step0.conf`
   is therefore disabled and the probe starts from the flat field — same as
   Y350a_HT. Enabling it needs a code change in `step0.py` plus a decision
   about modelling the aperture.
4. **`shrink_list.mat` is absent**, so shrink defaults to zeros (steps15
   warns). Same as Y350a_HT. `correct_correct3D.txt` was absent too until
   `estimate_drift.py --export-correct3d` wrote one on 2026-08-23 (above);
   steps 1-5 have to be rerun for it to take effect.
5. **Bin 0 does not fit tomo5.** The 3264³ complex64 object is 278 GB against
   a 160 GB aggregate (4 x 40 GB). Bin 2 (4.3 GB) and bin 1 (35 GB) are fine.
   Either stage to Polaris for bin 0, or stop the ladder at bin 1
   (`start_level_rec=1`).
