"""Per-angle drift from the entropy of an FBP reconstruction.

    a  ->  c  ->  s = P c  ->  d' = S(d; -s)  ->  u = FBP(d')  ->  entropy(u)

`a` is the free search vector, `c` Legendre coefficients, `s` the per-angle
shift.  Nelder-Mead minimises the entropy over `a`, coarse to fine in the
polynomial degree.  Nothing here needs a reference or a ground truth.

Used by `experimental/*/step7.py` and `tests/find_shifts_extra`.  The method
and the measurements behind the fixed choices are in
`tests/find_shifts_extra/doc/nelder_mead.pdf`.
"""
import numpy as np
import cupy as cp
from scipy.optimize import minimize

NBINS = 256


# ------------------------------------------------------------ the model ----
def legendre_at(x, deg):
    """[len(x), deg+1] Legendre matrix on an arbitrary abscissa in [-1, 1]."""
    x = np.asarray(x, dtype='float64')
    P = np.empty((len(x), deg + 1))
    P[:, 0] = 1.0
    if deg >= 1:
        P[:, 1] = x
    for k in range(2, deg + 1):
        P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
    return P


def legendre(ntheta, deg):
    """[ntheta, deg+1] Legendre matrix in x = 2i/(ntheta-1) - 1.

    Legendre and not monomials: ``|P_k| <= 1`` on [-1, 1], so a coefficient reads
    as an amplitude in pixels.  P_0 = 1 is part of the gauge below.
    """
    x = np.linspace(-1.0, 1.0, ntheta) if ntheta > 1 else np.zeros(1)
    return legendre_at(x, deg)


def gauge_curves(theta):
    """[3, ntheta, 2]: the drift a rigid object shift (dz, dx, dy) makes."""
    G = np.zeros((3, len(theta), 2))
    G[0, :, 0] = 1.0                 # dz -> constant vertical shift
    G[1, :, 1] = np.cos(theta)       # dx
    G[2, :, 1] = np.sin(theta)       # dy
    return G


def rigid_part(s, G):
    """Split s into (rigid part, the rest, (dz, dy, dx) in px).

    A rigid object shift gives the same picture in a different place, so every
    reference-free score is flat along these three directions.
    """
    A = G.reshape(3, -1).T
    t = np.linalg.lstsq(A, np.asarray(s, dtype='float64').reshape(-1),
                        rcond=None)[0]
    g = (A @ t).reshape(np.shape(s))
    return g, s - g, (t[0], t[2], t[1])


def free_basis(P, G):
    """(V, M): columns of V span the coefficients the search may move.

    M maps the coefficient vector to the shift curve; V is the null space of
    the gauge constraints on it, scaled so one unit of `a` is 1 px rms of
    shift.  That is what makes the simplex edge and xatol readable as pixels.
    """
    M = np.kron(P, np.eye(2))                      # c.ravel() -> s.ravel()
    C = np.linalg.qr(G.reshape(3, -1).T)[0].T @ M
    _, sv, Vt = np.linalg.svd(C)
    V = Vt[int(np.sum(sv > 1e-8 * sv.max())):].T
    return V / np.sqrt(((M @ V) ** 2).mean(axis=0)), M


# ------------------------------------------------------------- the score ---
class ShiftFourier:
    """Per-angle shift of the projections as a Fourier phase ramp.

    Unitary, so undoing a drift s is applying -s.  The Nyquist bin is left
    unrotated: a phase there cannot survive the .real and would cost the norm.
    """

    def __init__(self, nz, nd, chunk=128):
        fy = cp.fft.fftfreq(nz).astype('float32')
        fx = cp.fft.fftfreq(nd).astype('float32')
        if nz % 2 == 0:
            fy[nz // 2] = 0.0
        if nd % 2 == 0:
            fx[nd // 2] = 0.0
        self.fy, self.fx, self.chunk = fy[:, None], fx[None, :], chunk

    def __call__(self, p, s):
        s = np.asarray(s, dtype='float32')
        if not np.any(s):
            return p            # "no correction" stays bit-for-bit the input
        sy, sx = cp.asarray(s[:, 0]), cp.asarray(s[:, 1])
        out = cp.empty_like(p)
        for a in range(0, p.shape[0], self.chunk):
            b = min(a + self.chunk, p.shape[0])
            ang = (-2.0 * np.pi) * (self.fy[None] * sy[a:b, None, None]
                                    + self.fx[None] * sx[a:b, None, None])
            E = cp.empty(ang.shape, dtype='complex64')
            cp.cos(ang, out=E.real)
            cp.sin(ang, out=E.imag)
            out[a:b] = cp.fft.ifft2(cp.fft.fft2(p[a:b]) * E).real
        return out


def hist_of(u, sel, lo, hi, nbins=NBINS):
    """Grey-level histogram of the selected voxels, tails clipped into the ends.

    Clipping stops the search lowering the entropy by pushing mass out of the
    range instead of concentrating it.
    """
    return cp.histogram(cp.clip(u[sel], lo, hi), bins=nbins,
                        range=(lo, hi))[0].astype('float64')


def entropy_of_hist(h):
    """Shannon entropy of a histogram, in nats.  Lower = sharper.

    Histograms add, which is what lets a volume be scored a chunk at a time.
    """
    p = h / max(float(h.sum()), 1.0)
    p = p[p > 0]
    return float(-cp.sum(p * cp.log(p)))


def entropy(u, sel, lo, hi, nbins=NBINS):
    return entropy_of_hist(hist_of(u, sel, lo, hi, nbins))


def cylinder_mask(n, nz):
    """[nz, n, n] bool: the inscribed cylinder, the only region worth scoring."""
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
    return cp.broadcast_to(cp.asarray(np.sqrt(yy**2 + xx**2) < n / 2), (nz, n, n))


def z_split(nz, rank, size):
    """[lo, hi) of the z slices this rank reconstructs; balanced, in order."""
    q, r = divmod(nz, size)
    lo = rank * q + min(rank, r)
    return lo, lo + q + (rank < r)


def project_chunked(cl, obj, nzc):
    """R of a volume taller than the Tomo buffer; obj stays on the host.

    Parallel-beam R is slice-separable, so this is the same sinogram.
    """
    nz = obj.shape[0]
    d = cp.empty((cl.ntheta, nz, cl.nd), 'float32')
    for z0 in range(0, nz, nzc):
        z1 = min(z0 + nzc, nz)
        d[:, z0:z1] = cl.R(cp.asarray(obj[z0:z1]))
    return d


def grey_range(cl, d, sel, nzc, plo=0.05, phi=99.95, nfine=65536):
    """Histogram range for the whole run, from the UNCORRECTED FBP.

    Fixed once: two entropies are comparable only if the bins are the same.
    Two chunked passes (extremes, then a fine histogram) so the whole volume
    never has to be held at once.
    """
    nz = d.shape[1]
    lo, hi = np.inf, -np.inf
    for z0 in range(0, nz, nzc):
        u = cl.fbp(d[:, z0:z0 + nzc], 'ramp')
        v = u[sel[:u.shape[0]]]
        lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
        del u, v
    h = cp.zeros(nfine, 'float64')
    for z0 in range(0, nz, nzc):
        u = cl.fbp(d[:, z0:z0 + nzc], 'ramp')
        h += hist_of(u, sel[:u.shape[0]], lo, hi, nfine)
        del u
    c = cp.cumsum(h)
    c /= c[-1]
    e = np.linspace(lo, hi, nfine + 1)
    # v must be an ARRAY: cupy < 14 raises NotImplementedError on a python
    # float here, while numpy and cupy >= 14 accept one.
    i = cp.searchsorted(c, cp.asarray([plo / 100, phi / 100], dtype=c.dtype))
    ilo, ihi = int(i[0]), int(i[1])
    return float(e[ilo]), float(e[min(ihi + 1, nfine)])


# ------------------------------------------------------------ the search ---
class Focus:
    """The objective: one search vector in, one entropy out.

    `s_true` is optional and only ever used to RECORD how far off the search
    is; the optimiser never sees it.  On real data it is None.

    With `comm`, the z slices are split over the ranks.  Every rank holds the
    whole sinogram and applies the whole shift -- a vertical shift mixes z, so
    that part cannot be split -- and reconstructs only its own slices.
    Histograms add, so one 256-bin allreduce per evaluation is the only
    traffic.  `comm=None` is the old single-GPU path, unchanged.
    """

    def __init__(self, cl, sh, d, theta, sel, lo, hi, nzc=0, s_true=None,
                 comm=None):
        self.cl, self.sh, self.d = cl, sh, d
        self.theta, self.sel, self.lo, self.hi = theta, sel, lo, hi
        self.G = gauge_curves(theta)
        self.s_true = None if s_true is None else np.asarray(s_true, 'float32')
        self.nz = d.shape[1]
        self.nzc = min(nzc or self.nz, self.nz)
        self.zmid = self.nz // 2
        self.comm = comm
        size = 1 if comm is None else comm.size
        self.zlo, self.zhi = z_split(self.nz, 0 if comm is None else comm.rank,
                                     size)
        # who owns the slice the summary picture is made of
        self.zowner = next(k for k in range(size)
                           if z_split(self.nz, k, size)[0] <= self.zmid
                           < z_split(self.nz, k, size)[1])
        if comm is not None:
            from mpi4py import MPI
            self._inplace, self._sum = MPI.IN_PLACE, MPI.SUM
        self.P, self.V = None, None             # set per rung by fit_drift
        self.degmax, self.stage = 0, 0
        self.trace = []                 # (entropy, identifiable error, c.ravel())
        self.nit = 0

    def s_of(self, a):
        return (self.P @ (self.V @ np.asarray(a, dtype='float64')).reshape(-1, 2)
                ).astype('float32')

    def sweep(self, s, want_mid=False):
        """Undo the drift s, reconstruct, score.  One evaluation, in chunks.

        The shift is applied to the whole sinogram because a vertical shift
        mixes z; the FBP and the histogram then go nzc slices at a time, over
        this rank's z range only.
        """
        p = self.sh(self.d, -np.asarray(s, dtype='float32'))
        h, mid = cp.zeros(NBINS, 'float64'), None
        for z0 in range(self.zlo, self.zhi, self.nzc):
            z1 = min(z0 + self.nzc, self.zhi)
            u = self.cl.fbp(p[:, z0:z1], 'ramp')
            h += hist_of(u, self.sel[:u.shape[0]], self.lo, self.hi)
            if want_mid and z0 <= self.zmid < z1:
                mid = cp.asnumpy(u[self.zmid - z0])
            del u
        del p
        if self.comm is not None:
            # allreduce and not reduce: every rank runs its own copy of
            # Nelder-Mead, so they must all see the same number or they walk
            # different paths and stop at different times
            hh = cp.asnumpy(h)
            self.comm.Allreduce(self._inplace, hh, op=self._sum)
            h = cp.asarray(hh)
            if want_mid:
                mid = self.comm.bcast(mid, root=self.zowner)
        return entropy_of_hist(h), mid

    def err(self, s):
        """rms of the part of s - s_true a reconstruction can feel; nan if unknown."""
        if self.s_true is None:
            return float('nan')
        return float(np.sqrt((rigid_part(s - self.s_true, self.G)[1] ** 2).mean()))

    def __call__(self, a):
        s = self.s_of(a)
        q = self.sweep(s)[0]
        c = (self.V @ np.asarray(a, dtype='float64')).reshape(-1, 2)
        cf = np.zeros((self.degmax + 1, 2))
        cf[:len(c)] = c
        self.trace.append((q, self.err(s), *cf.ravel()))
        return q

    def callback(self, a):
        self.nit += 1


def fit_drift(fc, deg, maxfev=400, step=1.0, log=print):
    """Coarse-to-fine Nelder-Mead ladder, degree 1 to `deg`.  Returns (coef, rungs).

    One cold descent in the full space is a coin flip: Nelder-Mead is not
    rotation invariant and the free basis comes out of an SVD, so which local
    minimum it reaches depends on an orientation that carries no information
    (2 of 6 random orientations measured worse than no correction at all).
    Fitting one degree at a time, each rung seeded from the last, removes it.

    `maxfev` is a budget PER RUNG.  The initial simplex is written out because
    scipy perturbs a zero coordinate by 2.5e-4, and every coordinate is zero
    at the start.
    """
    ntheta = len(fc.theta)
    fc.degmax = deg
    coef, rungs = np.zeros((deg + 1, 2)), []
    for k in range(1, deg + 1):
        Pk = legendre(ntheta, k)
        Vk, _ = free_basis(Pk, fc.G)
        fc.P, fc.V, fc.stage = Pk, Vk, k
        ak = np.linalg.lstsq(Vk, coef[:k + 1].ravel(), rcond=None)[0]
        n0 = len(fc.trace)
        res = minimize(fc, ak, method='Nelder-Mead', callback=fc.callback,
                       options=dict(maxfev=maxfev, xatol=1e-3, fatol=1e-6,
                                    adaptive=True, disp=False,
                                    initial_simplex=np.vstack(
                                        [ak, ak + step * np.eye(len(ak))])))
        coef = np.zeros((deg + 1, 2))
        coef[:k + 1] = (Vk @ res.x).reshape(-1, 2)
        rungs.append(dict(deg=k, nfree=len(ak), nfev=len(fc.trace) - n0,
                          entropy=float(res.fun), err=fc.err(fc.s_of(res.x)),
                          success=bool(res.success), message=res.message))
        if log:
            e = rungs[-1]['err']
            log(f'    degree {k}: {len(ak)} free, {rungs[-1]["nfev"]:4d} '
                f'evaluations, entropy {res.fun:.5f}'
                + ('' if np.isnan(e) else f', identifiable error {e:.4f} px')
                + ('' if res.success else f'  ({res.message.lower().rstrip(".")})'))
    return coef, rungs


# ------------------------------------------------------------- the angles --
def scan_theta(h5_path, ntheta, key='/exchange/theta'):
    """(degrees, radians) for the search, taken from the scan and negated.

    Step 5 reconstructs at theta = -theta_raw/180*pi, and the sign is not
    cosmetic: R_{-t} u = R_{+t}(M u) for a reflection M that entropy cannot
    see, so a search at +t finds the drift of the mirrored object and the
    curve does not transfer.  Rows are taken on an even stride.
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            raise SystemExit(f'{h5_path}: no {key}; the search needs the '
                             f"scan's own angles")
        t_all = np.asarray(f[key][:], dtype='float64').ravel()
    if len(t_all) < ntheta:
        raise SystemExit(f'{h5_path}: {len(t_all)} angles, fewer than {ntheta}')
    idx = np.round(np.linspace(0, len(t_all), ntheta, endpoint=False)).astype(int)
    t = t_all[idx]
    return t, (-np.radians(t)).astype('float32')


# ------------------------------------------------------------- the output --
EXTRA_NAME = 'correct_correct3D_extra.txt'


def write_correct3d_extra(path, coef, nin, ntheta_scan, scale, rows=0):
    """Write the fitted drift on the scan's own angle grid, in Peter's layout.

    Two columns (horizontal, vertical) and ntheta+1 rows, because step 3 reads
    `np.loadtxt(p)[:ntheta, ::-1]` and ESRF's grid runs 0..180 inclusive.  The
    file carries the motion PRESENT in the data, like correct_correct3D.txt,
    so the two simply add.

    The polynomial is re-evaluated, not re-fitted: row i of the search sits at
    x = 2i/(nin-1) - 1 and theta = pi*i/nin, hence x(theta) below.
    """
    rows = rows or ntheta_scan + 1
    deg = coef.shape[0] - 1
    theta_out = np.pi * np.arange(rows) / ntheta_scan
    x_out = (2.0 * nin / (nin - 1)) * (theta_out / np.pi) - 1.0
    s_out = (legendre_at(x_out, deg) @ coef) * scale
    np.savetxt(path, s_out[:, ::-1], fmt='%.8e')
    return s_out


# ------------------------------------------------------------- the picture -
def summary_png(path, fc, theta_deg, s_found, mids, title=''):
    """Shifts found, the descent, and the middle slice before/after."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tr = np.array(fc.trace)
    has_true = fc.s_true is not None
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.0))
    for k, lab in enumerate(('vertical $s_y$', 'horizontal $s_x$')):
        a = ax[0][k]
        a.plot(theta_deg, s_found[:, k], 'tab:red', lw=1.6, label='found')
        if has_true:
            a.plot(theta_deg, fc.s_true[:, k], 'k-', lw=1.8, label='true')
            a.plot(theta_deg, (fc.s_true - s_found)[:, k], 'tab:blue', lw=1.0,
                   label='difference (the gauge)')
        a.axhline(0, color='0.7', lw=0.8)
        a.set(title=lab, xlabel='theta (deg)', ylabel='shift (px)')
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    a = ax[0][2]
    a.plot(tr[:, 0], '.', ms=2, color='0.7', label='every evaluation')
    a.plot(np.minimum.accumulate(tr[:, 0]), 'tab:red', lw=1.5, label='best so far')
    a.set(title='the descent', xlabel='evaluation', ylabel='entropy (nats)')
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    if has_true:
        a2 = a.twinx()
        a2.plot(tr[:, 1], '.', ms=2, color='tab:blue')
        a2.set_ylabel('identifiable shift error (px)', color='tab:blue')

    ref = mids[1] if mids[1] is not None else mids[0]
    vmin, vmax = np.percentile(ref, [0.5, 99.5])
    for k, (lab, m) in enumerate(zip(('no correction', 'found', 'the true shifts'),
                                     mids)):
        if m is None:
            ax[1][k].axis('off')
            continue
        ax[1][k].imshow(m, cmap='gray', vmin=vmin, vmax=vmax)
        ax[1][k].set_title(lab, fontsize=10)
        ax[1][k].axis('off')
    if title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975 if title else 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
