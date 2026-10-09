"""PrbfitTerm is K-blurred and follows F0's `model` knob.  Check every piece.

    python tests/prbfit/test_prbfit.py

    intensity   F1 = (lam/prb_size) * sum_j || K|D.prb_j|^2       - ref_j^2 ||^2
    amplitude   F1 = (lam/prb_size) * sum_j || sqrt(K|D.prb_j|^2) - ref_j   ||^2

Everything below runs for BOTH models x three psf_sigma, except where noted.

gradient(), hessian() and hessian3() are hand-derived and each leans on a
different property, so each gets its own check:

  1. K is self-adjoint          -- both derivations move K across the inner
                                   product; if psf_blur were not symmetric the
                                   gradient would be silently wrong only when
                                   psf_sigma > 0.
  2. D/DT are an adjoint pair   -- the stub must earn the chain rule it is used
                                   to verify, or the test proves nothing.
  3. energy_local is the stated -- against an INDEPENDENT float64 numpy/scipy
     functional                    reference (scipy.ndimage, not cupyx).
  4. Taylor                     -- the only check of the derivatives that does
                                   not re-use their derivation:
                                     F(t) - F0 - t<g,v>            = O(t^2)
                                     F(t) - F0 - t<g,v> - t^2/2 B  = O(t^3)
                                   Halving t must shrink these 4x and 8x.  A
                                   wrong constant factor changes the ratio, so
                                   this catches what a single-point comparison
                                   cannot.
  5. hessian3 == hessian x3, and B(v,w) == B(w,v).

Everything runs in float64: at float32 the 3rd-order residual hits roundoff
before the 8x ratio is established.  Swept over psf_sigma including 0, which
takes the psf_w=None identity path.

Two things are intensity-only, for reasons and not by omission:
  * check 4b, the exact quartic fit.  F(t) is a polynomial in t only for the
    intensity model; the amplitude one has a sqrt in it and no finite fit is
    exact.  The Taylor rates in check 4 cover the amplitude model instead.
  * the raw complex-normal probe.  The amplitude derivatives divide by a^3, and
    |D.prb| for a random normal probe times a random normal m gets close enough
    to zero that a Taylor residual measures the fourth derivative rather than
    the second.  The amplitude arm therefore uses a bright probe and a bright m
    (both offset by 2), which is also the only regime the model is used in --
    flat-field-normalised data means |psi| ~ 1, never ~1e-2.
"""
import os
import sys

import numpy as np
import cupy as cp
from scipy.ndimage import convolve1d as np_convolve1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from holotomocupy.extra_terms import PrbfitTerm          # noqa: E402
from holotomocupy.psf import psf_taps, psf_blur          # noqa: E402

NDIST, NZ, N = 3, 16, 24
LAM, PRB_SIZE = 0.7, 11.0          # deliberately not ndist*nz*n, so lam/prb_size shows up
SIGMAS = (0.0, 0.8, 2.0)
MODELS = ('intensity', 'amplitude')


class StubProp:
    """D(x, j) = m[j] * x, DT(y, j) = conj(m[j]) * y.

    Adjoint under the REAL inner product Re<x,y>, which is the one the gradient
    convention uses; for a complex-linear operator that is the Hermitian adjoint.
    """

    def __init__(self, m):
        self.m = m

    def D(self, x, j):
        return self.m[j:j + 1] * x

    def DT(self, y, j):
        return cp.conj(self.m[j:j + 1]) * y


def taps64(sigma):
    """Production taps, in float64.  psf_taps normalises in float32, so the sum
    is only 1 to ~1e-8; renormalising here keeps K exactly mass-conserving and
    the Taylor residuals free of a constant float32 bias."""
    _, w = psf_taps(sigma)
    if w is None:
        return None
    w = w.astype('float64')
    return w / w.sum()


def redot64(a, b):
    """Re<a,b> summed, in float64 (utils.redot views as float32)."""
    return float(cp.sum(cp.real(a) * cp.real(b) + cp.imag(a) * cp.imag(b)))


def build(sigma, rng, model='intensity'):
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    # bright m for the amplitude model, so |D.prb| stays away from the a^-3
    # blow-up -- see the module docstring
    off = 2.0 if model == 'amplitude' else 0.0
    m = cp.asarray(cplx(NDIST, NZ, N) + off)
    w = taps64(sigma)              # keep the whole test in float64
    t = PrbfitTerm(LAM, PRB_SIZE, NDIST, NZ, N, StubProp(m), psf_w=w, model=model)
    # ref is an AMPLITUDE (>= 0) for both models; the intensity one squares it
    # internally.  Offset so the residual is not tiny and both Hessian terms
    # matter, and kept the same relative size as the brightened field.
    t.ref = cp.asarray(np.abs(cplx(NDIST, NZ, N)) + 0.5 + off)
    return t, w


def ref_energy(t, prb, w):
    """Independent float64 reference for energy_local, on the host."""
    m = cp.asnumpy(t.cl_prop.m)
    ref = cp.asnumpy(t.ref)
    p = cp.asnumpy(prb)
    tot = 0.0
    for j in range(NDIST):
        I = np.abs(m[j] * p[j]) ** 2
        if w is not None:
            wn = cp.asnumpy(w)
            I = np_convolve1d(np_convolve1d(I, wn, axis=-2, mode='wrap'),
                              wn, axis=-1, mode='wrap')
        res = np.sqrt(I) - ref[j] if t.amp else I - ref[j] ** 2
        tot += np.sum(res ** 2)
    return LAM / PRB_SIZE * tot


def main():
    rng = np.random.default_rng(0)

    # --- 1. K self-adjoint ---------------------------------------------------
    for sigma in SIGMAS[1:]:
        w = taps64(sigma)
        a = cp.asarray(rng.standard_normal((2, NZ, N)))
        b = cp.asarray(rng.standard_normal((2, NZ, N)))
        lhs = float(cp.sum(psf_blur(a, w) * b))
        rhs = float(cp.sum(a * psf_blur(b, w)))
        r = abs(lhs - rhs) / abs(lhs)
        print(f"K self-adjoint   sigma={sigma:4.1f}   <Ka,b> vs <a,Kb>   rel {r:.2e}")
        assert r < 1e-12, r
        # and it must conserve the mean (taps sum to 1)
        m = abs(float(cp.sum(psf_blur(a, w))) - float(cp.sum(a))) / float(cp.abs(a).sum())
        print(f"K conserves mass sigma={sigma:4.1f}   sum(Ka) vs sum(a)   rel {m:.2e}")
        assert m < 1e-15, m

    # --- 2. stub D/DT adjoint ------------------------------------------------
    t, w = build(0.0, rng)
    x = cp.asarray(rng.standard_normal((1, NZ, N)) + 1j * rng.standard_normal((1, NZ, N)))
    y = cp.asarray(rng.standard_normal((1, NZ, N)) + 1j * rng.standard_normal((1, NZ, N)))
    r = abs(redot64(t.cl_prop.D(x, 0), y) - redot64(x, t.cl_prop.DT(y, 0)))
    r /= abs(redot64(t.cl_prop.D(x, 0), y))
    print(f"stub D/DT adjoint     Re<Dx,y> vs Re<x,DTy>   rel {r:.2e}")
    assert r < 1e-12, r

    ok = True
    for model in MODELS:
      for sigma in SIGMAS:
        print(f"\n{'='*66}\nmodel = {model}   psf_sigma = {sigma}"
              f"{'   (psf_w is None -- identity path)' if sigma == 0 else ''}\n{'='*66}")
        rng = np.random.default_rng(1)
        t, w = build(sigma, rng, model)
        cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
        prb = cp.asarray(cplx(NDIST, NZ, N) + (2.0 if t.amp else 0.0))
        v = cp.asarray(cplx(NDIST, NZ, N))
        u = cp.asarray(cplx(NDIST, NZ, N))
        if sigma == 0.0:
            assert t.psf_w is None

        # --- 3. energy_local vs independent host reference -------------------
        e0 = float(t.energy_local(prb))
        want = ref_energy(t, prb, w)
        r = abs(e0 - want) / abs(want)
        print(f"energy_local vs numpy/scipy reference        rel {r:.2e}")
        assert r < 1e-11, r

        # --- 4. Taylor -------------------------------------------------------
        g = cp.zeros_like(prb)
        t.gradient(g, prb, cp.ones(1, dtype='float64'))
        gv = redot64(g, v)
        B = t.hessian(prb, v, v)

        print(f"{'t':>10s} {'|F-F0-t<g,v>|':>15s} {'ratio':>6s}"
              f" {'|.. -t^2/2 B|':>15s} {'ratio':>6s}")
        p1 = p2 = None
        r1s, r2s = [], []
        for k in range(9):
            h = 2.0 ** -(k + 2)
            F = float(t.energy_local(prb + h * v))
            a1 = abs(F - e0 - h * gv)
            a2 = abs(F - e0 - h * gv - 0.5 * h * h * B)
            s1 = f"{p1/a1:6.2f}" if p1 else "     -"
            s2 = f"{p2/a2:6.2f}" if p2 else "     -"
            print(f"{h:10.6f} {a1:15.6e} {s1} {a2:15.6e} {s2}")
            if p1: r1s.append(p1 / a1)
            if p2: r2s.append(p2 / a2)
            p1, p2 = a1, a2
        # the last pair is the most converged
        print(f"  -> grad order {r1s[-1]:.3f} (expect 4)   "
              f"hess order {r2s[-1]:.3f} (expect 8)")
        if not (3.8 < r1s[-1] < 4.2):
            print("  FAIL gradient order"); ok = False
        if not (7.3 < r2s[-1] < 8.7):
            print("  FAIL hessian order"); ok = False

        # --- 4b. exact quartic (INTENSITY ONLY) ----------------------------
        # F(t) = ||K|u+tv|^2 - r^2||^2 is EXACTLY a quartic in t (|u+tv|^2 is
        # quadratic, K is linear, the square doubles the degree), so a 5-point
        # fit recovers its coefficients with no truncation error at all.  This
        # pins <g,v> and B to full precision instead of to the O(t) rate above.
        # The amplitude model is not polynomial in t -- sqrt(K|u+tv|^2) -- so
        # there is nothing to fit and check 4 is what covers it.
        if not t.amp:
            ts = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            Fs = np.array([float(t.energy_local(prb + tt * v)) for tt in ts])
            c = np.polyfit(ts, Fs, 4)      # c[4] + c[3] t + c[2] t^2 + ...
            r1 = abs(c[3] - gv) / abs(gv)
            r2 = abs(2 * c[2] - B) / abs(B)
            print(f"quartic fit  dF/dt  vs Re<g,v>                rel {r1:.2e}")
            print(f"quartic fit  d2F/dt2 vs B(v,v)                rel {r2:.2e}")
            if r1 > 1e-12 or r2 > 1e-12:
                print("  FAIL"); ok = False

        # --- 5. hessian3 and symmetry ----------------------------------------
        gg, ge, ee = t.hessian3(prb, v, u)
        for name, got, wnt in (("B(v,v)", gg, t.hessian(prb, v, v)),
                               ("B(v,u)", ge, t.hessian(prb, v, u)),
                               ("B(u,u)", ee, t.hessian(prb, u, u))):
            r = abs(got - wnt) / max(abs(wnt), 1e-30)
            print(f"hessian3 {name} vs hessian                  rel {r:.2e}")
            if r > 1e-11:
                print("  FAIL"); ok = False
        r = abs(t.hessian(prb, v, u) - t.hessian(prb, u, v)) / abs(t.hessian(prb, v, u))
        print(f"hessian symmetric  B(v,u) == B(u,v)          rel {r:.2e}")
        if r > 1e-12:
            print("  FAIL"); ok = False

        # --- gradient is DT of something: check it against a direct chain -----
        # grad = DT(4 p u) for both models, with p = K[g'(J)/2], built here from
        # energy_local's own pieces rather than from _p.
        man = cp.zeros_like(prb)
        for j in range(NDIST):
            uu = t.cl_prop.D(prb[j:j+1], j)
            J = t._blur(cp.real(uu) ** 2 + cp.imag(uu) ** 2)
            if t.amp:
                a = cp.maximum(cp.sqrt(J), t._AMP_FLOOR)
                half_gp = 0.5 * (1.0 - t.ref[j:j+1] / a)
            else:
                half_gp = J - t.ref[j:j+1] ** 2
            man[j:j+1] = LAM / PRB_SIZE * t.cl_prop.DT(4 * t._blur(half_gp) * uu, j)
        r = float(cp.abs(man - g).max()) / float(cp.abs(g).max())
        print(f"gradient == DT(4 K[g-prime(J)/2] u)          rel {r:.2e}")
        if r > 1e-12:
            print("  FAIL"); ok = False

    # --- 6. gen_sqrt_ref is consistent with the term it seeds ----------------
    # A ref made from prb must put prb AT the minimum: F1 = 0 and grad F1 = 0.
    # Unblurred, this fails by O(blur^2) once a PSF is on -- the regularizer
    # would then pull the probe by exactly the PSF the model already applies.
    for model in MODELS:
      for sigma in SIGMAS:
        rng = np.random.default_rng(2)
        t, w = build(sigma, rng, model)
        prb = cp.asarray(rng.standard_normal((NDIST, NZ, N))
                         + 1j * rng.standard_normal((NDIST, NZ, N))
                         + (2.0 if t.amp else 0.0))
        t.ref = cp.empty((NDIST, NZ, N), dtype='float64')
        t.gen_sqrt_ref(prb, t.ref)
        e = float(t.energy_local(prb))
        # ||ref^2||^2 or ||ref||^2, the natural unit of each model's residual
        scale = float(cp.sum(t.ref ** (2 if t.amp else 4))) * LAM / PRB_SIZE
        g = cp.zeros_like(prb)
        t.gradient(g, prb, cp.ones(1, dtype='float64'))
        gn = float(cp.abs(g).max()) / float(cp.abs(prb).max())
        print(f"gen_sqrt_ref {model[:3]} sigma={sigma:4.1f}  F1 at its own probe"
              f" {e/scale:.2e}   |grad| {gn:.2e}")
        if e / scale > 1e-28 or gn > 1e-12:
            print("  FAIL"); ok = False

    # --- lam == 0 is inert ---------------------------------------------------
    t0 = PrbfitTerm(0.0, PRB_SIZE, NDIST, NZ, N, StubProp(cp.ones((NDIST, NZ, N))))
    g0 = cp.zeros((NDIST, NZ, N), dtype='complex128')
    t0.gradient(g0, cp.ones((NDIST, NZ, N), dtype='complex128'), cp.ones(1))
    assert float(cp.abs(g0).max()) == 0.0
    z = cp.ones((NDIST, NZ, N), dtype='complex128')
    assert t0.hessian(z, z, z) == 0 and t0.hessian3(z, z, z) == (0, 0, 0)
    assert t0.energy_local(z) == 0
    print("\nlam=0 short-circuit ok")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
