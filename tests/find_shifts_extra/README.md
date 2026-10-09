# find_shifts_extra — entropy autofocus for the per-angle drift

The algorithm is `src/holotomocupy/autofocus.py`; `experimental/*/step7.py` is
its driver on real data. These two tests are the only things here, both
synthetic and both self-contained — no mounted data, any GPU host.

```bash
./run.sh test_find_shifts.py           # ~12 s, finds an injected drift
./run.sh test_find_shifts_null.py      # ~12 s, measures the bias with no drift
```

Both print `PASS` or `FAIL` on the last line and exit non-zero on failure.
`--n`, `--nslice`, `--ntheta`, `--deg` and `--maxfev` scale them up;
`--png <file>` writes a summary figure.

## What is being tested

One objective evaluation is a whole reconstruction:

    a  ->  c  ->  s = P c  ->  d' = S(d; -s)  ->  u = FBP(d')  ->  entropy(u)

`c` are Legendre coefficients of the per-angle shift `s`, `S` a Fourier phase
ramp, and the score is the Shannon entropy of a 256-bin histogram over a fixed
cylinder. Blur spreads the grey levels, so a misaligned reconstruction scores
higher. Nelder-Mead minimises it. Nothing needs a reference.

| test | asserts |
|---|---|
| `test_find_shifts.py` | ≥ 80 % of the *identifiable* injected drift comes back, and the entropy drops. 95 % at the defaults. |
| `test_find_shifts_null.py` | with no drift at all, the search invents ≤ 0.1 px rms. 0.02 px at the defaults. |

**Identifiable** means minus the gauge. A rigid object translation
`(dz, dy, dx)` appears as `s_y = dz`, `s_x = dx·cosθ + dy·sinθ`, and gives the
same picture in a different place, so no reference-free score can see it —
two thirds of the injected drift here. It is projected out before the search,
which is also the only reason the descent terminates.

Three other choices are fixed rather than exposed, and each is load-bearing:
the histogram range, taken once from the *uncorrected* FBP with the tails
clipped into the end bins; the initial simplex, written out as `x₀ + 1 px·eᵢ`
because scipy perturbs a zero coordinate by 2.5e-4; and the coarse-to-fine
ladder over the polynomial degree, because one cold descent in the full space
reaches the right minimum for only about two thirds of the (arbitrary) basis
orientations.

`doc/nelder_mead.pdf` is the write-up: what the simplex does in nine variables
rather than one, how it is initialised, and the measurements behind all of the
above. `doc/nm_figs.py` rebuilds its figures from `doc/nm_landscape.npz` (a 49x49
grid of the real objective plus a recorded simplex walk) and the trace CSV
next to it; neither needs a GPU.

## Reading a step7 answer

`step7.py` re-projects a volume that was already reconstructed with the
current shifts, so part of what it finds is the metric's own bias rather than
a residual misalignment. On this phantom that bias is 0.02 px because the data
are exactly consistent with the model; on the real ctxl FT volume it was
0.49 px. Run the null test on the same grid, angle count and slab before
believing a step7 answer of comparable size.
