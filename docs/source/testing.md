# Tests

Scripts, not a pytest suite: each prints its own verdict and exits non-zero on
failure. Run them with `PYTHONPATH=<repo>/src`, or through the folder's
`run.sh` where there is one.

```bash
tests/unit/run.sh                                    # start here, ~1 min
tests/find_shifts_extra/run.sh
PYTHONPATH=src python tests/adjoint/test_adjoint_f1.py
```

## `tests/unit` — run this after any change to the solver

148 checks on one GPU in about a minute:

| file | what |
|---|---|
| `test_env.py` | imports, GPU, MPI, parallel HDF5 |
| `test_operators.py` | adjointness of every linear operator, scored against ‖a‖·‖b‖ |
| `test_derivatives.py` | gradients and Hessians of the data misfit, by Taylor test |
| `test_shift_derivatives.py` | the same for both shift operators, at *m* = 1 and on the chirp-z path |
| `test_cascade.py` | the F0..F4 chain end to end |
| `test_mpi.py` | collectives and the theta↔z redistribution, 2 ranks |

**How the derivative tests work.** An exact first derivative leaves a residual
of O(h²) and an exact second leaves O(h³), so the test fits a slope in log-log
across several step sizes. Residuals flatten once they reach float32 rounding,
and points at that floor are dropped from the fit rather than dragging the
slope down.

## Everything else

| folder | what it checks | needs |
|---|---|---|
| `adjoint` | dot-product test for `dF1*` | GPU |
| `data_mask` | the out-of-grid detector mask | GPU, MPI |
| `find_shifts_extra` | entropy autofocus — the engine behind `step7.py` | GPU |
| `model` | amplitude vs intensity misfit | GPU |
| `mosaic`, `mosaic_brain` | multi-tile generation and reconstruction | GPU, MPI |
| `mpi_functions` | `MPIClass.redist` | MPI |
| `nfp` | near-field ptychography misfit and derivatives | GPU |
| `performance` | `bench_shift.py`, `bench_chunking.py`, MPI scaling | GPU, MPI |
| `prbfit`, `psf` | the probe-fit term and the Gaussian detector PSF | GPU |
| `propagation` | the Fresnel propagator, both backends | GPU |
| `shift`, `tomo` | the shift operators against scipy; `Tomo`'s oversampled detector | GPU |
