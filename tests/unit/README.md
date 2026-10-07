# unit — is the installation sound, and are the operators right?

```bash
./run.sh                 # everything: 106 checks, ~1 min on one GPU
./run.sh operators       # one file
NP=4 ./run.sh mpi        # the MPI part on 4 ranks
```

No pytest, no new dependency: each file is a script that prints one `PASS` or
`FAIL` line per check and exits non-zero if any failed. The functions are
named `test_*` and use plain `assert`, so pytest collects them too if you have
it.

| file | checks |
|---|---|
| [`test_env.py`](test_env.py) | every import, a visible CUDA device, a kernel that compiles and runs, MPI, parallel HDF5, which FFT backend resolved |
| [`test_operators.py`](test_operators.py) | adjointness of `Tomo.R/RT` (both `nd`), `Shift.S/Sadj`, `ShiftFFT`, `Propagation.D/DT`, the PSF blur; `ShiftFFT` against `Shift`; `FBP(R u)` against `u` |
| [`test_derivatives.py`](test_derivatives.py) | Taylor tests on the data misfit `F0` alone — first derivative order 2, second order 3 — for both models with and without the blur; `<gF0, y> == dF0[y]`; Hessian symmetry and its sign at the solution |
| [`test_cascade.py`](test_cascade.py) | the levels above F0 — `F1` probe × propagate, `F2` exp(i psi), `F3` shift, `F4` demagnification — in composition: `<gF chain, y> == dF chain[y]` per variable, and Taylor order 2 and 3 of the composed objective |
| [`test_mpi.py`](test_mpi.py) | `MPIClass.redist` both directions and round-trip, the allreduce helpers, a parallel HDF5 write/read, one device per rank |

## What the tests are actually checking

**Adjointness.** `<A u, v> == <u, A* v>` for random `u`, `v`. It costs two
operator calls and catches a missing conjugate, a transposed index or a wrong
normalisation — none of which a forward-only check can see. Tolerance 1e-5
relative, which is what float32 FFTs and the NUFFT gather give.

**Taylor order, not residual size.** An exact first derivative makes the
linear model's residual fall as `h²`; an exact second derivative makes the
quadratic model's fall as `h³`. The measured log-log slope is the test, since
a wrong sign or a missing factor of 2 drops it to 1 while leaving any single
residual plausible. The amplitude model needs larger steps than the intensity
one — its misfit is ~20× smaller, so its `h³` term reaches float32 rounding
sooner.

**The cascade is tested in composition, not level by level.** The three
loops in `test_cascade.py` are copied from `gradients_cascade` and
`hessian_cascade`, so what runs is the production chain; a direction that
moves only `prb`, only `obj`, only `pos` or only `tp` then names the level
that broke. Two conventions the test had to respect, both real: `gF3` returns
the un-prefiltered `Deltapsi`, so the caller applies `cl_shift.coeff` once
(as `gradients_cascade` does at `if last:`), and `mask_1d` is pinned host
memory that `@gpu_batch` uploads in production.

**Residuals at float32 rounding are not failures.** Once a Taylor residual
reaches ~1e-7 of the objective it stops falling, and fitting a slope through
that flat tail reads as low order. `order()` drops those points and reports
`inf` when none are left, which the output marks as `(at float32 rounding)`.

**FBP is checked up to a constant.** The ramp filter removes the zero
frequency, so `FBP(R u)` matches `u` in shape (correlation > 0.999) but not in
offset. Comparing the raw fields would fail for a reason that is not a bug.

`HTC_PYTHON`, `MPIRUN`, `NP` and `HTC_SCRATCH` override the interpreter, the
launcher, the rank count and the scratch directory.
