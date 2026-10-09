"""
test_data_mask.py — unit checks for the out-of-grid detector mask.

    mpirun -np 1 python test_data_mask.py
    mpirun -np 4 python test_data_mask.py      # exercises the theta split

eff_demag = (1+shrink)/norm_magnification is > 1 for every plane but the
reference one, so a detector pixel back-maps to an object-grid coordinate that
falls outside the grid as soon as nobj < n*max(eff_demag) -- and the sample
sliding by -r moves that footprint further off the grid on one side. Shrinkage
is fitted per axis, so eff_demag is [ndist, ntheta, 2] with (y, x) differing
slightly and the y and x bounds computed independently.
Rec._build_data_mask gives the unsupported pixels zero weight in the data fit;
the F0 family then carries that weight through F0, dF0, d2F_dF0 and gF0.

F0 is the INTENSITY misfit 1/N sum W (K|x|^2 - d)^2 with W = my*mx, so the mask
enters in exactly two places inside each of F0/dF0/d2F_dF0/gF0: on the misfit
integrand, and on the residual weight p = K[W(J-d)] -- both of which see the
blurred intensity J rather than x -- plus once more between the Hessian's two
convolutions.  Nothing masks the DERIVATIVE terms again: they are weighted by p,
which already carries W, and masking them twice would square it.  That is the
trap this section is guarding.

The mask is per (distance, angle) and axis-separable, so it is stored as the
two 1-D factors (mask_1d) plus the box they came from (mask_box), never as a
dense [ndist, ntheta, nz, n] array.

Four things are checked:

  1. The [chunk, nz, 1] and [chunk, 1, n] mask factors broadcast against the
     [chunk, nz, n] arrays the cascade carries; gF0 comes back exactly zero on
     the masked-out pixels -- not merely small -- and the masked-out pixels of
     x cannot move F0, dF0 or d2F_dF0 at all, however large the perturbation.
  2. dF0 and d2F_dF0 are still the first and second derivatives of the MASKED
     F0 (central differences, float64 accumulation).
  3. The geometry, per angle, against a brute-force evaluation of the sampling
     formula in s_kernel: at margin 0 it is EXACT, and with a margin it is
     strictly conservative -- it never keeps a pixel whose four cubic B-spline
     taps are not all inside the object grid. mask_1d and mask_box agree, and
     each kept set is one contiguous interval.
  4. The boxes do not depend on how the angles are split over ranks: under
     -np 4 each rank reproduces the -np 1 boxes for the angles it owns.
"""

import sys
import numpy as np
import cupy as cp
from mpi4py import MPI

sys.path.insert(0, '../..')
from holotomocupy.rec_mpi import Rec
from holotomocupy.utils import redot, reprod


def check_kernels():
    """1. masked pixels carry zero weight; 2. derivatives of the masked F0."""
    chunk, nz, n = 3, 16, 20
    rng = cp.random.RandomState(0)
    def c(): return (rng.rand(chunk, nz, n) + 1j * rng.rand(chunk, nz, n)).astype('complex64')
    x = c() + 0.5          # +0.5: keep |x| away from 0, as the real x0 is
    y, z, w = c(), c(), c()
    d = rng.rand(chunk, nz, n).astype('float32') + 0.5
    # separable per-angle mask: a random box per chunk element
    my = cp.zeros((chunk, nz, 1), dtype='float32')
    mx = cp.zeros((chunk, 1, n), dtype='float32')
    for j, (y0, y1, x0, x1) in enumerate([(0, nz, 0, n), (3, 12, 2, 17), (5, 6, 0, 9)]):
        my[j, y0:y1, 0] = 1
        mx[j, 0, x0:x1] = 1
    m = my * mx                                        # [chunk, nz, n]
    N = chunk * nz * n

    # No blur here: K = identity, so J = |x|^2 and p = W(J-d).  The blur itself
    # is tests/psf/'s job; what this file owns is the mask riding through
    # F0/dF0/d2F_dF0/gF0, which is the same algebra either way.
    rc = object.__new__(Rec)
    rc.psf_w = None
    rc.model = 'intensity'
    rc._mask_y, rc._mask_x = my, mx
    rc.data_size = N
    rc.apply_F_from = lambda v, i: v    # gF0's cascade step, identity here

    # ---- F0 and gF0 against the explicit masked formulas -----------------
    # Relative bars: the intensity form squares |x|, so the integrand is
    # O(|x|^4) ~ 20 here and a float32 ulp on it is ~2e-6 in absolute terms.
    m64 = m.astype('float64')
    J64 = reprod(x.astype('complex128'), x.astype('complex128'))
    d64 = d.astype('float64')
    ref_F0 = float(cp.sum(m64 * (J64 - d64) ** 2)) / N
    ref_g = (4.0 / N) * (m64 * (J64 - d64)) * x.astype('complex128')

    got_F0 = float(rc.F0(x, d))
    gg = rc.gF0(x, d)
    assert gg.shape == x.shape
    e_f = abs(got_F0 - ref_F0) / abs(ref_F0)
    e_g = float(cp.abs(gg - ref_g).max()) / float(cp.abs(ref_g).max())
    print(f"broadcast vs explicit: F0 {e_f:.3g}  gF0 {e_g:.3g}  (relative)")
    assert e_f < 1e-6 and e_g < 1e-6

    off = cp.broadcast_to((m == 0), x.shape)
    bad = float(cp.abs(gg[off]).max())
    print(f"     gF0: max|value| on masked-out pixels = {bad:g}")
    assert bad == 0.0

    # The masked-out pixels of x must not reach any of the four values.  W = 0
    # there exactly, so a 100x perturbation has to leave them BIT-identical --
    # the summation order does not change.
    xp = x + np.float32(100.0) * c() * (m == 0)
    same = ((float(rc.F0(xp, d)) == got_F0)
            and (float(rc.dF0(xp, y, d)) == float(rc.dF0(x, y, d)))
            and (float(rc.d2F_dF0(xp, y, z, w, d))
                 == float(rc.d2F_dF0(x, y, z, w, d))))
    print(f"  100x perturbation on masked-out pixels: values identical = {same}")
    assert same

    # ---- 2. dF0 and d2F_dF0 are the derivatives of the MASKED F0 ---------
    # The second difference cancels ~6 digits, so F0 is evaluated in float64 --
    # the same method, handed complex128 input (dF0 goes through redot and stays
    # float32, which is 1e-7 relative and far inside the bar below).
    x64, y64 = x.astype('complex128'), y.astype('complex128')
    def F(t):
        return float(rc.F0(x64 + t * y64, d64))
    t = 1e-2
    num1 = (F(t) - F(-t)) / (2 * t)
    ana1 = float(rc.dF0(x, y, d))
    num2 = (F(t) - 2 * F(0) + F(-t)) / t**2
    ana2 = float(rc.d2F_dF0(x, y, y, None, d))
    r1, r2 = abs(num1 - ana1) / abs(ana1), abs(num2 - ana2) / abs(ana2)
    print(f"  dF0: numeric={num1:.8e} analytic={ana1:.8e} rel={r1:.2e}")
    print(f"d2F0 : numeric={num2:.8e} analytic={ana2:.8e} rel={r2:.2e}")
    assert r1 < 1e-3 and r2 < 1e-3


class _Stub:
    """Enough of Rec for _build_data_mask to run unmodified."""
    _build_data_mask = Rec._build_data_mask

    def __init__(self, n, nz, nobj, nzobj, ed, pos, margin, comm):
        self.n, self.nz, self.nobj, self.nzobj = n, nz, nobj, nzobj
        self.ndist, self.local_ntheta = ed.shape[:2]   # ed is [ndist, ntheta, 2]
        self.mask_oob, self.mask_oob_margin = True, margin
        self.eff_demag = cp.asarray(ed.astype('float32'))
        self.mask_1d  = np.ones((self.ndist, self.local_ntheta, nz + n), dtype='float32')
        self.mask_box = np.zeros((self.ndist, self.local_ntheta, 4), dtype='int32')
        self.rank = comm.rank
        self.cl_mpi = type('m', (), {'comm': comm})()

    def dense(self):
        """Materialize the per-angle mask the kernels see: [ndist, ntheta, nz, n]."""
        my = self.mask_1d[:, :, :self.nz]
        mx = self.mask_1d[:, :, self.nz:]
        return (my[..., :, None] * mx[..., None, :]) > 0


def exact_mask(n, nz, nobj, nzobj, ed, pos):
    """Brute force, PER ANGLE: keep a pixel only if all four taps of the cubic
    B-spline are in-grid. Mirrors the index math in s_kernel."""
    ndist, T = ed.shape[:2]
    out = np.zeros((ndist, T, nz, n), bool)
    tx, ty = np.arange(n), np.arange(nz)
    for k in range(ndist):
        for t in range(T):
            xx = ed[k, t, 1] * (tx - (n - 1) * 0.5) - pos[k, t, 1] + (nobj - 1) * 0.5
            yy = ed[k, t, 0] * (ty - (nz - 1) * 0.5) - pos[k, t, 0] + (nzobj - 1) * 0.5
            ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
            out[k, t] = np.outer((iy - 1 >= 0) & (iy + 2 <= nzobj - 1),
                                 (ix - 1 >= 0) & (ix + 2 <= nobj - 1))
    return out


def _geometry():
    """AtomiumL1_HT geometry scaled down, plus a large-displacement distance
    (eff_demag == 1, sample sliding most of a frame) -- the case the per-angle
    box exists for."""
    rng = np.random.default_rng(0)
    nm = np.array([1, 0.95890, 0.82341, 0.63664, 1.0])
    N, T = 128, 32
    # per-axis shrink: y and x drift independently, so ed is [ndist, T, 2]
    ed = (np.tile((1.0 / nm)[:, None, None], (1, T, 2))
          * (1 + rng.normal(0, 0.002, (len(nm), T, 2))))
    pos = rng.uniform(-12.5, 12.5, (len(nm), T, 2))
    pos[-1] = rng.uniform(-19.0, 19.0, (T, 2))          # the large-disp plane
    return N, T, nm, ed, pos


def check_geometry(comm):
    """3. exact at margin 0, conservative with a margin; boxes consistent."""
    N, T, nm, ed_all, pos_all = _geometry()
    lo = T * comm.rank // comm.size
    hi = T * (comm.rank + 1) // comm.size

    for nobj, tag in ((N, "nobj == n  (undersized grid)"),
                      (int(np.ceil(N / nm[1:-1].min() / 64) * 64), "nobj == auto-computed")):
        for margin in (0.0, 2.0):
            st = _Stub(N, N, nobj, nobj, ed_all[:, lo:hi], pos_all[:, lo:hi], margin, comm)
            st._build_data_mask({'pos': pos_all[:, lo:hi].astype('float32')})
            built = st.dense()
            ref   = exact_mask(N, N, nobj, nobj, ed_all[:, lo:hi], pos_all[:, lo:hi])
            wrong = int((built & ~ref).sum())
            extra = int((ref & ~built).sum())

            # mask_1d is exactly the box, i.e. each kept set is one interval
            for k in range(st.ndist):
                for t in range(st.local_ntheta):
                    y0, y1, x0, x1 = st.mask_box[k, t]
                    want = np.zeros((N, N), bool)
                    want[y0:y1, x0:x1] = True
                    assert np.array_equal(want, built[k, t]), \
                        f"mask_1d/mask_box disagree at dist {k}, angle {t}"

            if comm.rank == 0:
                print(f"{tag}, margin={margin:g}: nobj={nobj} "
                      f"kept={built.mean():.4f} exact={ref.mean():.4f} "
                      f"wrongly-kept={wrong} extra-discarded={extra} "
                      f"({100 * extra / max(ref.sum(), 1):.2f}%)")
            assert wrong == 0, "mask keeps a pixel whose B-spline support is out of grid"
            if margin == 0.0:
                assert extra == 0, "at margin 0 the box must be the exact support"

    # the per-angle box must beat one shared centred rectangle on the
    # large-displacement plane -- that is the entire point of the change
    st = _Stub(N, N, N, N, ed_all[:, lo:hi], pos_all[:, lo:hi], 2.0, comm)
    st._build_data_mask({'pos': pos_all[:, lo:hi].astype('float32')})
    loc = float(st.dense()[-1].mean()) * st.local_ntheta
    tot = comm.allreduce(loc, op=MPI.SUM) / T
    r = np.abs(pos_all[-1]).max(axis=0)
    shared = ((N - 2 * (r[0] + 2.0)) / N) * ((N - 2 * (r[1] + 2.0)) / N)
    if comm.rank == 0:
        print(f"large-disp plane: per-angle keeps {100*tot:.1f}%, "
              f"one shared centred box would keep {100*shared:.1f}%")
    assert tot > shared + 0.02


def check_mpi_invariance(comm):
    """4. the boxes do not depend on the theta split over ranks."""
    N, T, nm, ed_all, pos_all = _geometry()
    lo = T * comm.rank // comm.size
    hi = T * (comm.rank + 1) // comm.size

    ser = _Stub(N, N, N, N, ed_all, pos_all, 2.0, MPI.COMM_SELF)
    ser._build_data_mask({'pos': pos_all.astype('float32')})
    par = _Stub(N, N, N, N, ed_all[:, lo:hi], pos_all[:, lo:hi], 2.0, comm)
    par._build_data_mask({'pos': pos_all[:, lo:hi].astype('float32')})

    assert np.array_equal(ser.mask_box[:, lo:hi], par.mask_box), \
        f"rank {comm.rank}: boxes differ from the serial build"
    if comm.rank == 0:
        print(f"mpi invariance: boxes identical for a {comm.size}-way theta split")


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        check_kernels()
    check_geometry(comm)
    check_mpi_invariance(comm)
    comm.Barrier()
    if comm.rank == 0:
        print("ALL OK")
