# Quickstart: the demo

[`demo/`](https://github.com/tomography/holotomocupy/tree/master/demo) runs the
whole chain on synthetic data — no beamline files, no mounted filesystems. The
phantom and the probe are built locally, the data are simulated with the same
forward model the solver inverts, and the truth is kept so every number can be
checked.

| | notebook (1 GPU) | script (N GPUs) | what it does |
|---|---|---|---|
| step 0 | `01_nfp_probe.ipynb` | `01_nfp_probe.py` | probe retrieval by near-field ptychography from a Siemens star |
| step 6 | `02_holotomography.ipynb` | `02_holotomography.py` | holotomography of a 3-D phantom, 4 distances |

```bash
cd demo
./run.sh 01_nfp_probe.py               # 1 GPU
./run.sh 02_holotomography.py          # 1 GPU
./run_mpi.sh 4 02_holotomography.py    # 4 GPUs
./run.sh 02_holotomography.py --n 256 --ntheta 360 --niter 257
```

The notebooks and the scripts build the same data — both import
`phantoms.py` — so a notebook is the readable version of the script beside it.

## What the demo should produce

| run | GPU | `shift_type` | result |
|---|---|---|---|
| `01_nfp_probe.py` (n 512, 16 positions, 257 iters) | Quadro RTX 8000 | `cubic` | positions 0.951 → **0.031 px**, corr **0.961** |
| `02_holotomography.py` (n 128, 180 angles, 4 distances, 100 iters) | Quadro RTX 8000 | `cubic` | positions 0.248 → **0.048 px**, object corr **0.978** |
| `02_holotomography.py` | A100-40GB | `fft` | positions 0.248 → **0.052 px** |

The `pos abs error` logged during the solve is measured from the *starting
guess*, so it grows as the refinement works. The line printed at the end is
the distance to the truth, which is the one to read.

## Choosing `shift_type`: cubic or fft

Two interpolants implement the same operator interface:

* **`cubic`** — B-spline, 4×4 taps, mirrored edges. The default for step 6.
* **`fft`** — Fourier shift theorem; exact for a band-limited input, so no
  interpolation kernel sits between the object and the position gradient.
  The default for step 0.

The split follows the geometry. A multi-distance scan has magnification ≠ 1 at
every plane but the first, so `fft` takes the chirp-z (Bluestein) path on
nearly every call: 7–30× slower and 8–16× the scratch memory. Near-field
ptychography is single-distance, *m* = 1, a plain phase ramp, where `fft` is
cheap and its exactness helps the position gradient.
