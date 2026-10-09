# tests

Scripts, not a pytest suite: each prints its own verdict and exits non-zero on
failure. Run them with `PYTHONPATH=<repo>/src`, or through the folder's
`run.sh` where there is one. Every folder marked GPU needs one CUDA device;
the MPI ones need `mpirun -np N`.

| folder | what it checks | needs | time |
|---|---|---|---|
| **`unit`** | **start here** — imports, GPU, MPI, parallel HDF5, every operator's adjoint, gradients and Hessians of both shift operators and the cascade by Taylor test | GPU, MPI | ~1 min |
| `adjoint` | dot-product (adjoint) test for `dF1*` in `grad(F0 o F1)` | GPU | s |
| `data_mask` | the out-of-grid detector mask, `Rec._build_data_mask` | GPU, MPI 1 | s |
| `disp_study` | dose/displacement study: how `ndist`, amplitude and probe size trade off | GPU | long |
| `find_shifts_extra` | entropy autofocus for the per-angle drift — the engine behind `step7.py` | GPU | ~25 s |
| `model` | amplitude vs intensity misfit on synthetic data: speed and result | GPU | min |
| `mosaic` | mosaic (multi-tile) data generation and reconstruction, notebooks | GPU | min |
| `mosaic_brain` | the synthetic mosaic brain: 5 tiles x 4 distances, full step0/steps15/step6 chain | GPU, MPI | long |
| `mpi_functions` | `MPIClass.redist` correctness and timing | MPI | s |
| `nfp` | near-field ptychography: the intensity+PSF misfit and its derivatives | GPU | min |
| `performance` | shift/chunking benchmarks (`bench_shift.py`, `bench_chunking.py`), BH timing and memory logs, and `mpi_scaling/` | GPU, MPI | long |
| `prbfit` | `PrbfitTerm` follows F0's `model` knob and is K-blurred | GPU | s |
| `propagation` | the Fresnel propagator, cuPy and cuFFTDx backends | GPU | s |
| `psf` | the Gaussian blur operator, F0's derivatives with it, and end-to-end inside `Rec` | GPU | min |
| `rec` | the `\|\|obj^{n+1} - obj^n\|\|^2` convergence metric | GPU | s |
| `shift` | `Shift.curlyS` against scipy, and `ShiftFFT` as its drop-in | GPU | s |
| `tomo` | `Tomo`'s oversampled detector (`nd`) and the `tomo_upsample` chain | GPU | min |

After any change to the solver run `unit/run.sh` first — it is the fast,
complete one. Then `psf`, `prbfit` and `find_shifts_extra` for the parts it
does not reach.
