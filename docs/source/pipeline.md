# The pipeline

A beamtime lives in one folder under `experimental/`, holding its configs and
a copy of each stage script. The stages are numbered by the config they read.

| stage | script | reads | writes |
|---|---|---|---|
| 0 | `step0.py` | `config_step0.conf` | the probe, by near-field ptychography (optional) |
| 1–5 | `steps15.py` | `config_steps15.conf` | raw → HDF5, preprocess, shifts, binned data, Paganin+FBP init |
| 6 | `step6.py` | `config_step6_bin{2,1,0}.conf` | the BH reconstruction, coarse to fine |
| 7 | `step7.py` | a step-6 config | a per-angle drift correction, then rerun step 6 |

## Step 6 runs as a ladder

Coarse levels are cheap and fix the geometry; fine levels add detail. Each
level resumes from the previous level's last checkpoint, so `start_iter` must
equal it and `checkpoint_step` must divide both `start_iter` and `niter-1`,
or the handoff checkpoint will not exist.

```
bin 2   4x4   iters    0 -> 1024
bin 1   2x2   iters 1024 -> 1280
bin 0   1x1   iters 1280 -> 1536
```

`bin` is a **level**: the binning factor is `2**bin`, so 0 is full resolution,
1 is 2×2, 2 is 4×4. `n`, `nz`, `nobj` and `nzobj` in a level's config are the
values *at that level*, i.e. the full-resolution value divided by `2**bin`.

## Positions and the rotation axis

Positions and `rotation_center_shift` are offsets from the **middle of the
detector**, and the rotation axis sits at `(n-1)/2` at every level. That is
the geometric middle of pixels `0..n-1`, which is what makes binning a plain
factor of two with no half-pixel term.

## Step 7 — per-angle drift

`step7.py` re-projects a finished reconstruction and searches for the
per-angle shift that minimises the entropy of the FBP. It writes
`correct_correct3D_extra.txt` next to the config, and step 6 adds it to the
positions it reads from `cshifts_final`.

```bash
python step7.py config_step6_bin1.conf
```

The correction is applied **once**, by the level that starts fresh
(`start_iter=0`); levels that resume inherit it through the checkpoint, and
re-applying it there would double count. So after step 7, restart the ladder
from bin 2 rather than resuming the level you stopped at.

:::{warning}
Step 7 re-projects a volume that was reconstructed with the current shifts
already in place, so on a converged reconstruction part of what it measures is
the metric's own bias — on one null run that was 1.7 px vertical and 10.7 px
horizontal peak-to-peak. Run `tests/find_shifts_extra` on the same geometry
and compare before shipping an answer.
:::

## Running on a cluster

Each dataset folder carries a `polaris_run.sh`: a PBS script with one literal
`mpiexec` line per stage, each ending in `|| exit $?` so a failed stage stops
the job instead of letting the next one seed from a checkpoint that was never
written. Comment out the lines you do not want. The software environment
comes from `experimental/polaris_env.sh`, sourced inside the job.
