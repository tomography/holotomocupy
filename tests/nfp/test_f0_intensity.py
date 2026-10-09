"""RecNFP's intensity + PSF misfit: derivatives, and agreement with rec_mpi.

F0 is 1/N sum (K|x|^2 - d)^2 -- the same statement as rec_mpi.Rec.F0 with the
detector mask W == 1, which NFP always has (it never shifts the object off its
own grid).  That makes the strongest check available a DIRECT COMPARISON with
the step-6 implementation, which is in production: they must agree exactly, not
approximately, and they do (rel 0.0 at every sigma, the w term included).

The Taylor half is the independent one -- it never calls rec_mpi, and its
reference F0 is float64 numpy + scipy, so a shared error in the cupy kernels or
in the tap convention would still show up.

Single GPU, no MPI, a few seconds:
    PYTHONPATH=<repo>/src python tests/nfp/test_f0_intensity.py
"""
import numpy as np, cupy as cp
from holotomocupy.rec_nfp_mpi import RecNFP
from holotomocupy.rec_mpi import Rec
from holotomocupy.psf import psf_taps
from holotomocupy.utils import redot, reprod

N, NZ, NX = 3, 32, 32

def make(cls, sigma, masked):
    Stub = type('Stub', (), {})      # fresh class per call: setattr below is
    s = Stub()                       # class-level and would clobber a shared one
    s.data_size = N * NZ * NX
    s.psf_sigma, s.psf_w = psf_taps(sigma)
    s.model = 'intensity'                # the misfit-model knob both classes read
    if masked:
        s._mask_y = cp.ones((N, NZ, 1), 'float32')
        s._mask_x = cp.ones((N, 1, NX), 'float32')
    # rec_mpi's F0 group is now one self-contained function per model behind a
    # dispatcher; RecNFP still carries the _pw/_cw/_F0_p helpers and the fused
    # kernels, and the gF0 check below calls them by name.  Copy whatever the
    # class actually has.
    for m in ('_blur', 'F0', 'dF0', 'd2F_dF0',
              'F0_int', 'dF0_int', 'd2F_dF0_int', 'gF0_int',
              'F0_amp', 'dF0_amp', 'd2F_dF0_amp', 'gF0_amp',
              '_AMP_FLOOR', '_pw', '_cw', '_F0_p',
              '_F0_fused', '_F0_fused_amp', '_r_fused', '_r_fused_amp',
              '_mask_fused', '_c_fused_amp', '_d2F_dF0_fused', '_gF0_fused'):
        if hasattr(cls, m):
            setattr(Stub, m, getattr(cls, m))
    return s

rng = np.random.default_rng(0)
x = cp.asarray(rng.normal(size=(N, NZ, NX)) + 1j*rng.normal(size=(N, NZ, NX)), 'complex64')
v = cp.asarray(rng.normal(size=(N, NZ, NX)) + 1j*rng.normal(size=(N, NZ, NX)), 'complex64')
w = cp.asarray(rng.normal(size=(N, NZ, NX)) + 1j*rng.normal(size=(N, NZ, NX)), 'complex64')
# data near the model intensity, as real data is
d = cp.asarray((cp.abs(x)**2).get() * (1 + 0.05*rng.normal(size=(N, NZ, NX))), 'float32')

ok = True
for sigma in (0.0, 0.8, 2.0):
    nfp = make(RecNFP, sigma, masked=False)

    # --- A. Taylor, against an INDEPENDENT float64 numpy reference ----------
    # Not against nfp.F0 itself: that sums in float32, and with a blur the
    # second difference (f(h)-2f(0)+f(-h))/h^2 loses every digit to
    # cancellation.  scipy's convolve1d on float64 is also a second opinion on
    # cupyx's, and taps_np is the same discrete-Gaussian array.
    from scipy.ndimage import convolve1d as conv64
    taps_np = None if nfp.psf_w is None else cp.asnumpy(nfp.psf_w).astype('float64')
    xn, vn, dn = (cp.asnumpy(a) for a in (x, v, d))
    xn, vn, dn = xn.astype('complex128'), vn.astype('complex128'), dn.astype('float64')
    def blur64(a):
        if taps_np is None: return a
        return conv64(conv64(a, taps_np, axis=-2, mode='wrap'),
                      taps_np, axis=-1, mode='wrap')
    def f(t):
        J = blur64(np.abs(xn + t*vn)**2)
        return float(np.sum((J - dn)**2) / nfp.data_size)
    h  = 1e-3
    d1_fd = (f(h) - f(-h)) / (2*h)
    d2_fd = (f(h) - 2*f(0) + f(-h)) / h**2
    d1 = float(nfp.dF0(x, v, d))
    d2 = float(nfp.d2F_dF0(x, v, v, None, d))
    e1 = abs(d1-d1_fd)/abs(d1_fd); e2 = abs(d2-d2_fd)/abs(d2_fd)
    print(f"sigma={sigma:4.1f}  dF0  {d1: .8e} vs FD {d1_fd: .8e}  rel {e1:.2e}")
    print(f"sigma={sigma:4.1f}  d2F0 {d2: .8e} vs FD {d2_fd: .8e}  rel {e2:.2e}")
    ok &= e1 < 1e-3 and e2 < 1e-3   # float32 finite differences

    # gF0's pointwise part: redot(4/N p x, v) must equal dF0
    g = nfp._gF0_fused(x, nfp._F0_p(x, d), np.float32(4/nfp.data_size))
    eg = abs(float(redot(g, v)) - d1)/abs(d1)
    print(f"sigma={sigma:4.1f}  gF0.v vs dF0                      rel {eg:.2e}")
    ok &= eg < 1e-6

    # symmetry of the Hessian in y,z
    a = float(nfp.d2F_dF0(x, v, w, None, d)); b = float(nfp.d2F_dF0(x, w, v, None, d))
    es = abs(a-b)/abs(a)
    print(f"sigma={sigma:4.1f}  Hessian symmetry y<->z            rel {es:.2e}")
    ok &= es < 1e-5

    # --- B. cross-check vs rec_mpi with W == 1 -------------------------------
    ref = make(Rec, sigma, masked=True)
    for name, got, exp in (
        ("F0     ", float(nfp.F0(x, d)),                  float(ref.F0(x, d))),
        ("dF0    ", d1,                                   float(ref.dF0(x, v, d))),
        ("d2F_dF0", float(nfp.d2F_dF0(x, v, w, w, d)),    float(ref.d2F_dF0(x, v, w, w, d))),
    ):
        e = abs(got-exp)/max(abs(exp), 1e-30)
        print(f"sigma={sigma:4.1f}  {name} vs rec_mpi(W=1)          rel {e:.2e}")
        ok &= e < 1e-6
    print()

print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
