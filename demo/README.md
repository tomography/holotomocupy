# demo — the pipeline end to end on synthetic data

Two steps, entirely synthetic, small enough to run on one GPU in a couple of
minutes. No beamline data, no mounted filesystems: the phantom and the probe
are built here, the data are simulated with the same forward model the solver
inverts, and the truth is kept so every number can be checked.

| | notebook (1 GPU) | script (N GPUs) | what it does |
|---|---|---|---|
| step 0 | [`01_nfp_probe.ipynb`](01_nfp_probe.ipynb) | [`01_nfp_probe.py`](01_nfp_probe.py) | probe retrieval by near-field ptychography from a Siemens star, `npos` scan positions |
| step 6 | [`02_holotomography.ipynb`](02_holotomography.ipynb) | [`02_holotomography.py`](02_holotomography.py) | holotomography of a 3-D phantom, 4 distances |

```bash
jupyter lab                            # then open either notebook
./run.sh 01_nfp_probe.py               # 1 GPU
./run.sh 02_holotomography.py          # 1 GPU
./run_mpi.sh 4 02_holotomography.py    # 4 GPUs
./run.sh 02_holotomography.py --n 256 --ntheta 360 --niter 257
```

The notebooks and the scripts build the same data — both import
[`phantoms.py`](phantoms.py) — so a notebook is the readable version of the
script next to it and they can be compared directly.

## What is skipped, and why

A real run starts at raw detector frames and goes through `steps15.py`:
EDF→HDF5, outlier removal, shift bookkeeping, multi-distance back-projection,
Paganin+FBP. None of that applies here — there are no files to convert and no
shifts to look up, so the demo calls `gen_data` to produce the intensities
directly and starts the solver from zero. Everything downstream of that is the
production code path: `RecNFP` and `Rec` are the same classes
`experimental/*/step0.py` and `step6.py` use, with the same config fields,
and they write the same checkpoints, TIFFs and `conv.csv` into `demo_out/`.

## Measured

| run | GPU | `shift_type` | result |
|---|---|---|---|
| `01_nfp_probe.py` (n 512, 16 positions, 257 iters) | Quadro RTX 8000 | `cubic` | positions 0.951 → **0.031 px**, corr **0.961** |
| `01_nfp_probe.py` | Quadro RTX 8000 | `fft` | positions 0.951 → **0.103 px** |
| `02_holotomography.py` (n 128, 180 angles, 4 distances, 100 iters) | Quadro RTX 8000 | `cubic` | positions 0.248 → **0.048 px**, object corr **0.978** |
| `02_holotomography.py` | **A100-40GB** | `fft` | positions 0.248 → **0.052 px** |
| `02_holotomography.py` | Quadro RTX 8000 | `fft` | **crashes** at iter ~26, see below |

### `shift_type`: `fft` vs `cubic`

`ShiftFFT` is now a complete drop-in -- `Sback` / `coeff_back` / `curlySback`
exist, so the Paganin stitch no longer has to fall back to `Shift`, and the
chirp-z path (magnification != 1, which every multi-distance call takes) is
correct and tested (`tests/unit/test_shift_derivatives.py`,
`tests/shift/test_square_mag.py`).

Two things still argue for `cubic` here:

* **cost.** On the chirp-z path ShiftFFT is 7-30x slower than the B-spline
  gather and needs 8-16x the scratch, because every call runs ~3 FFTs per axis
  of a padded grid instead of one gather. `tests/performance/bench_shift.py`
  has the table.
* **this GPU.** With four distances the chirp-z path dies on the Quadro RTX
  8000 in this box with `CUDA_ERROR_ILLEGAL_ADDRESS` inside
  `_chirpz_lastaxis`, at iteration ~26.  The same commit, same config, runs
  all 100 iterations clean on an A100-40GB and lands at 0.052 px, so the
  failure is the card, not the algorithm.

`SHIFT` at the top of [`02_holotomography.py`](02_holotomography.py) is the one
line to change.

Measured on NFP (single distance, so `m = 1` and no chirp-z):

| `shift_type` | Paganin start corr | final corr | final position error |
|---|---|---|---|
| `cubic` | 0.807 | **0.961** | **0.031 px** |
| `fft` | 0.345 | 0.511 | 0.103 px |

The `fft` number is the same with `--paganin 0`, so that cost is `fft` itself,
not the start.

The `pos abs error` the solver logs during BH is measured from the **starting
guess**, so it grows as the refinement works; the line printed at the end is
the distance to the truth, which is the one to read.

## The pieces

- [`phantoms.py`](phantoms.py) — `siemens_star`, `star_projection`,
  `phantom3d` (nested cube frames, rotated off-axis so no edge lines up with
  the grid, then low-passed) and `id16a_probe` (the measured probe from
  `../data/prb_id16a`, cropped and normalised).
- [`run.sh`](run.sh) / [`run_mpi.sh`](run_mpi.sh) — set `PYTHONPATH`, pick the
  env, keep the cuPy cache off `/tmp`. `DEMO_PYTHON` overrides the interpreter,
  `MPIRUN` the launcher.
- [`bind.sh`](bind.sh) — one GPU per rank, round-robin over the node.
