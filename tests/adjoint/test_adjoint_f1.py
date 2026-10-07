"""Adjoint ("dot-product") tests for dF1^* in grad(F0 o F1) = (4/N) p x.

    PYTHONPATH=<repo>/src python tests/adjoint/test_adjoint_f1.py

F1(x) = |x|^2 maps complex to REAL, so its differential and its adjoint go
between two spaces with two different inner products:

    <a,b>_R = sum a b                       (real fields)
    <a,b>_C = Re sum conj(a) b  = redot     (complex fields)

    dF1  [y] = 2 Re(conj(x) y)              C -> R
    dF1^*[u] = 2 u x                        R -> C      (x, NOT conj(x))

and with the blur K on the intensity, J = K|x|^2,

    dJ  [y] = K(2 Re(conj(x) y))
    dJ^*[u] = 2 x K[u]                      (blur FIRST, then multiply by 2x)

The adjoint identity <u, dJ[y]>_R == <dJ^*[u], y>_C is checked directly, and
then the chain it exists to serve:

    grad(F0 o F1) = dJ^*[(1/N) W g'(J)] = (4/N) x K[W g'(J)/2] = (4/N) p x

which is what gF0 returns, so <gF0(x,d), y>_C must equal dF0(x,y,d) exactly --
both models, K on and off, mask on and off.

A dot-product test passes for the wrong reason easily, so every identity is
also run against the adjoints that are WRONG in ways that survive casual
inspection -- conj(x) instead of x, K applied after the multiply instead of
before, and the mask applied outside K instead of inside -- which must all
fail it.

HOW THE REJECTIONS ARE SCORED.  Not as a relative error: <u, dJ[y]> is a
near-cancelling sum over random u and y, and the wrong adjoints differ from
the right one by another such sum -- conj(x) vs x changes the integrand by
4 u Im(x) Im(y), whose mean is zero -- so a wrong operator can land within a
percent of the right answer on a single random draw and still be wrong.  Each
rejection is scored against the RESIDUAL OF THE IDENTITY ITSELF instead: the
true adjoint closes the identity to float32 rounding, so a candidate that
misses by >= 1000x that residual is separated from it by a margin no rounding
can explain.  In practice the true residual is ~1e-9 and the wrong ones are
~1e-2, a gap of seven orders.
"""
import sys
import numpy as np
import cupy as cp

from holotomocupy.rec_mpi import Rec
from holotomocupy.psf import psf_taps
from holotomocupy.utils import redot, reprod

CHUNK, NZ, N = 2, 48, 40
DATA_SIZE = CHUNK * NZ * N
rng = np.random.default_rng(7)

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        fails.append(name)


def reject(name, truth, wrong, residual, scale):
    """The candidate must miss the identity by >= 1000x its true residual."""
    miss = abs(truth - wrong)
    floor = max(residual, 1e-9 * scale)      # a bit-exact identity is not 0 margin
    check(f"rejects: {name}", miss > 1e3 * floor,
          f"misses by {miss:.3g} = {miss/floor:.1e}x the true residual "
          f"({floor:.2g})")


def cxa(shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype('complex64')


x = cp.asarray(cxa((CHUNK, NZ, N)))
y = cp.asarray(cxa((CHUNK, NZ, N)))
u = cp.asarray(rng.standard_normal((CHUNK, NZ, N)).astype('float32'))  # real!
d = cp.asarray((np.abs(cp.asnumpy(x)) ** 2
                * (1 + 0.05 * rng.standard_normal((CHUNK, NZ, N)))).astype('float32'))

# a real mask: a box per chunk element, as _build_data_mask produces
my = cp.zeros((CHUNK, NZ, 1), dtype='float32')
mx = cp.zeros((CHUNK, 1, N), dtype='float32')
for j, (a0, a1, b0, b1) in enumerate([(0, NZ, 0, N), (5, 40, 3, 33)]):
    my[j, a0:a1, 0] = 1
    mx[j, 0, b0:b1] = 1


def harness(sigma, model, masked):
    r = object.__new__(Rec)
    r.data_size = DATA_SIZE
    r.model = model
    r.psf_w = None if sigma is None else psf_taps(sigma)[1]
    if masked:
        r._mask_y, r._mask_x = my, mx
    else:
        r._mask_y = r._mask_x = cp.ones(1, dtype='float32')
    r.apply_F_from = lambda v, i: v      # gF0 runs the cascade first
    return r


def dotR(a, b):
    """<a,b>_R in float64 -- these sums cancel, float32 would not resolve them."""
    return float(cp.sum(a.astype('float64') * b.astype('float64')))


def dotC(a, b):
    """<a,b>_C = Re sum conj(a) b, in float64."""
    a64, b64 = a.astype('complex128'), b.astype('complex128')
    return float(cp.sum(a64.real * b64.real + a64.imag * b64.imag))


for SIGMA in (None, 1.7):
    tag = "no blur  " if SIGMA is None else "sigma=1.7"
    print(f"\n--- K: {tag} " + "-" * 46)
    r = harness(SIGMA, 'intensity', masked=False)
    K = r._blur

    # ---- 0. K itself must be self-adjoint, or nothing below means anything
    a = cp.asarray(rng.standard_normal((CHUNK, NZ, N)).astype('float32'))
    b = cp.asarray(rng.standard_normal((CHUNK, NZ, N)).astype('float32'))
    lhs, rhs = dotR(a, K(b)), dotR(K(a), b)
    sc = float(cp.sum(cp.abs(a.astype('float64')) * cp.abs(K(b).astype('float64'))))
    check(f"{tag} K is self-adjoint (<a,Kb> == <Ka,b>)",
          abs(lhs - rhs) / sc < 1e-6, f"{lhs:.10g} vs {rhs:.10g}")

    # ---- 1. the adjoint identity itself --------------------------------
    fwd = K(2 * reprod(x, y))                   # dJ[y],  a real field
    lhs = dotR(u, fwd)                          # <u, dJ[y]>_R
    adj = 2 * x * K(u)                          # dJ^*[u], a complex field
    rhs = dotC(adj, y)                          # <dJ^*[u], y>_C
    sc = float(cp.sum(cp.abs(u.astype('float64')) * cp.abs(fwd.astype('float64'))))
    res = abs(lhs - rhs)
    check(f"{tag} <u, dJ[y]>_R == <dJ^*[u], y>_C", res / sc < 1e-6,
          f"{lhs:.10g} vs {rhs:.10g}   (|d|/sum|terms| = {res/sc:.2e})")

    # ---- 2. the same identity must REJECT the near-miss adjoints -------
    candidates = [("conj(x) instead of x", 2 * cp.conj(x) * K(u))]
    if SIGMA is None:
        # K = identity: K(2 x u) and 2 x K(u) are the same array, so this
        # candidate is not wrong here and there is nothing to reject.  It is
        # only distinguishable once K actually mixes neighbouring pixels.
        print("[SKIP] no blur   rejects: K after the multiply   "
              "(identical operators when K = 1)")
    else:
        candidates.append(("K after the multiply", K(2 * x * u)))
    for name, wrong in candidates:
        reject(f"{tag} {name}", lhs, dotC(wrong, y), res, sc)

# ---- 3. the chain: <gF0, y>_C == dF0[y], every model x mask x blur ------
print("\n--- the composed gradient " + "-" * 40)
for MODEL in ('intensity', 'amplitude'):
    # the amplitude model divides by a^3; on a raw complex normal |x| reaches
    # ~0.01, so it runs on the bright field it is actually used on (|psi| ~ 1).
    X = x if MODEL == 'intensity' else (x + 2.0).astype('complex64')
    D = (d if MODEL == 'intensity'
         else cp.asarray((np.abs(cp.asnumpy(X)) ** 2
                          * (1 + 0.05 * rng.standard_normal((CHUNK, NZ, N)))
                          ).astype('float32')))
    for masked in (False, True):
        for SIGMA in (None, 1.7):
            r = harness(SIGMA, MODEL, masked)
            tag = (f"{MODEL[:3]} {'mask' if masked else 'W=1 '} "
                   f"{'K=1  ' if SIGMA is None else 'K=1.7'}")
            g = r.gF0(X, D)                       # (4/N) p x
            got = float(redot(g, y))              # <grad, y>_C
            exp = float(r.dF0(X, y, D))           # dF0[y]
            sc = 4 / DATA_SIZE * float(cp.sum(cp.abs(
                (g / np.float32(4 / DATA_SIZE)).astype('complex128'))
                * cp.abs(y.astype('complex128'))))
            check(f"{tag} <gF0, y>_C == dF0[y]", abs(got - exp) / sc < 1e-6,
                  f"{got:.10g} vs {exp:.10g}")

            # the conjugate trap, on the real thing: conj(grad) has the same
            # norm and is the direction an adjoint with the wrong conjugation
            # would return, so a gradient check on |grad| alone cannot see it.
            reject(f"{tag} conj(grad)", exp, float(redot(cp.conj(g), y)),
                   abs(got - exp), sc)

    # ---- 4. W must sit INSIDE K, not outside -----------------------------
    # p = K[W g'/2].  Masking after the blur is a different operator and is not
    # the adjoint of anything; with K = 1 the two coincide, so this needs a blur.
    r = harness(1.7, MODEL, masked=True)
    W = my * mx
    J = r._blur(reprod(X, X))
    if MODEL == 'amplitude':
        a = cp.maximum(cp.sqrt(J), r._AMP_FLOOR)
        half_gp = 0.5 * (1.0 - cp.sqrt(D) / a)
    else:
        half_gp = J - D
    g_out = (np.float32(4 / DATA_SIZE) * (W * r._blur(half_gp))) * X   # W outside K
    got = float(redot(g_out, y))
    exp = float(r.dF0(X, y, D))
    sc = 4 / DATA_SIZE * float(cp.sum(cp.abs((W * r._blur(half_gp)).astype('float64'))
                                      * cp.abs(y.astype('complex128'))))
    # the residual of the identity in the same configuration, as the yardstick
    res = abs(float(redot(r.gF0(X, D), y)) - exp)
    reject(f"{MODEL[:3]} W outside K in p", exp, got, res, sc)

print("\n" + ("all checks passed" if not fails
              else f"{len(fails)} FAILED: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
