"""FFT-based shift operator with chirp-z (Bluestein) magnification.

Drop-in alternative for `holotomocupy.shift.Shift`: same public API
(`coeff`, `S`, `Sadj`, `curlyS`, `curlySc`, `dcurlySc`, `dcurlySadjc`,
`d2curlySc`, the magnification-aware `dcurlySmc`/`dcurlySadjmc`/`d2curlySmc`,
plus the `coeff_cached`/`_reset`/`_stats` triad) and the same constructor
signature, but every shift is implemented via the Fourier shift theorem (m=1)
or a separable chirp-z transform (arbitrary per-projection m).

Cubic-B-spline `Shift` interpolates with a 4x4 tap stencil, so it has a real
interpolation kernel: it low-passes the object slightly and leaks a fraction
of a percent between neighbouring pixels.  The Fourier shift is exact for a
band-limited input -- there is no stencil and no kernel error -- which is what
a position-refining solver wants, because the interpolation error otherwise
biases the position gradient.  The price is the periodic boundary (below) and
one FFT pair per shift instead of one kernel launch.

When all `m[t] == 1`, `S` / `Sadj` take the fast path: a single batched
2-D FFT pair on the input grid with a separable linear-phase multiply.
When any `m[t] != 1`, the chirp-z path runs per axis -- one Bluestein
correlation each, evaluating the Fourier interpolant at the m-scaled,
r-shifted output positions.  `m` is `(ntheta, 2)`, axis 1 = (my, mx), and the
chirp-z path matches the `s_kernel` convention

    x_in = m_x·(tx - (n-1)/2) - r_x + (npsi-1)/2

so the two classes place the output on the same grid.  RecNFP holds m = 1
throughout (the NFP object plane is the detector plane), and so does `Rec` at
ndist = 1 with the shrink model off; the chirp-z path is there for the
magnified callers.

The Bluestein convolution length L is the next power of two of
`N_in + N_out - 1`, so chirp-z costs ~3 batched FFTs of length L per
axis (vs 1 of length N for the m=1 fast path).

⚠ PERIODIC BOUNDARY.  The shift runs directly on the input (nzpsi × npsi)
grid, so whatever leaves one edge re-enters the opposite one -- unlike
`Shift`, whose `sym_idx` mirrors at the border.  The object must therefore
vanish near the boundary, or the shifts must stay well inside the grid: size
the object grid as n + 2·(max |pos|) + margin and this never fires.

`coeff(psi)` is the identity: FFT shifts need no B-spline prefilter.
The derivative methods (`dcurlySc`, `dcurlySadjc`, `d2curlySc`) also
dispatch automatically: at `m=1` they use the existing fast-path
algebra; at `m≠1` they exploit the linearity of `S` in its input --
e.g. `dcurlySc = S(c1 + Δry·∂c/∂y + Δrx·∂c/∂x, r, m)` with the
spatial derivatives computed via FFT differentiation -- then call
the same `S`/`Sadj` chirp-z path used by the forward operator.

`Sback`/`curlySback`/`coeff_back` are intentionally omitted; callers needing
them should hold a `Shift` instance alongside.  Neither `RecNFP` nor `Rec`
calls them -- they belong to the back-projection helpers, not the solve -- so
both classes can be switched over with `shift_type = 'fft'`.

The magnification-aware family at the end of the class exists because
`rec_mpi.Rec` differentiates through m (the shrink variable `tp`), which
`RecNFP` does not.  Those three cannot fuse the way the `*c` family does --
Δm enters as a per-OUTPUT-pixel weight rather than a constant spectral factor
-- so they cost a few extra transforms, and short-circuit back to the fused
`*c` methods whenever Δm is identically zero (i.e. always, when rho[tp] = 0).

PORTED from the holotomocupy `mpi` branch with two changes, both to make it a
true drop-in for THIS branch's `Shift`:
  * the constructor takes `nchunk` in Shift's positional slot;
  * `d2curlySc` uses the own-slot pairing of `d2s_kernel` (caller crosses the
    coefficients) rather than folding the crossing into the formula.  See the
    SLOT PAIRING note on that method -- getting this backwards is invisible to
    a Taylor test and only shows up off the diagonal.
"""

import cupy as cp
import cupyx.scipy.fft as cufft
from .utils import redot


def ascontig(x):
    """cp.ascontiguousarray that also handles numpy/pinned inputs."""
    return cp.ascontiguousarray(cp.asarray(x))


# Fused elementwise kernels — each one is a single CUDA pass with no
# intermediate memory traffic, replacing chains of broadcast multiplies.
apply_sep_phase = cp.ElementwiseKernel(
    'complex64 c, complex64 py, complex64 px',
    'complex64 out',
    'out = c * py * px',
    'shift_fft_apply_sep_phase',
)

# imag(conj(a) * b) = a.real * b.imag - a.imag * b.real, written as a single
# pass over float32 lanes — avoids the big complex intermediate
# `cp.conj(a) * b` would otherwise materialize.
imag_conj_prod = cp.ElementwiseKernel(
    'complex64 a, complex64 b',
    'float32 out',
    'out = a.real() * b.imag() - a.imag() * b.real()',
    'shift_fft_imag_conj_prod',
)

# Combine the four big-array ops of dcurlySc into one pass:
#   out = (c * d + c1) * py * px
combine_dcurlySc = cp.ElementwiseKernel(
    'complex64 c, complex64 d, complex64 c1, complex64 py, complex64 px',
    'complex64 out',
    'out = (c * d + c1) * py * px',
    'shift_fft_combine_dcurlySc',
)

# Combine the d2curlySc terms into one pass:
#   out = (c * d1 * d2 + c1 * d1 + c2 * d2) * py * px
# Each coefficient meets the shift in its OWN slot -- the same pairing
# d2s_kernel uses, so the caller crosses the coefficients exactly as it does
# for Shift.d2curlySc.  See the SLOT PAIRING note on ShiftFFT.d2curlySc.
combine_d2curlySc = cp.ElementwiseKernel(
    'complex64 c, complex64 c1, complex64 c2, '
    'complex64 d1, complex64 d2, complex64 py, complex64 px',
    'complex64 out',
    'out = (c * d1 * d2 + c1 * d1 + c2 * d2) * py * px',
    'shift_fft_combine_d2curlySc',
)


class ShiftFFT():
    """Fourier-shift-theorem shift operator (assumes magnification m = 1).

    Runs the FFT shift directly on the input (nzpsi × npsi) grid — periodic
    BC, so the input should vanish near the boundary or shifts should stay
    well inside the grid; otherwise wrap-around artefacts appear.

    coeff() is the identity (no B-spline prefilter needed for FFT shifts).
    """

    def __init__(self, n, npsi, nz, nzpsi, nchunk=None):
        self.n = n
        self.npsi = npsi
        self.nz = nz
        self.nzpsi = nzpsi

        # cuFFT plan reuse for the batch.
        if nchunk is not None:
            tmp = cp.empty([nchunk, nzpsi, npsi], dtype='complex64')
            self.plan       = cufft.get_fft_plan(tmp, axes=(-2, -1), value_type='C2C')
            self.plan_batch = nchunk
            del tmp
        else:
            self.plan       = None
            self.plan_batch = None

        # 2π·k/N per axis on the input grid (fftfreq order: DC, +, -).
        # The output-region offset (nzpsi-nz)/2, (npsi-n)/2 is folded into
        # the per-call phase so the output is the top-left [:nz, :n] slice
        # of the ifft result.
        self.fy = (2 * cp.pi * cp.fft.fftfreq(nzpsi)).astype('float32')
        self.fx = (2 * cp.pi * cp.fft.fftfreq(npsi )).astype('float32')
        # Precomputed -i·fy and -i·fx as complex64 — used by deriv_factor
        # to skip the per-call (-1j)·real cast.
        self.negi_fy = ((-1j) * self.fy).astype('complex64')
        self.negi_fx = ((-1j) * self.fx).astype('complex64')
        self.eff_dy = (nzpsi - nz) * 0.5
        self.eff_dx = (npsi  - n ) * 0.5
        self.inv_N = cp.float32(1.0 / (nzpsi * npsi))

        # Output-grid pixel coordinate measured from the tile centre.  The
        # sample position the operator interpolates at is
        #     u_a = m_a·(t_a − (N_out−1)/2) − r_a + (N_in−1)/2
        # so ∂u_a/∂m_a = tau_a(t) while ∂u_a/∂r_a = −1, i.e.
        #     ∂/∂m_a = −tau_a(t) · ∂/∂r_a,
        # which is exactly what dsm_kernel / d2sm_kernel / dsmadj_kernel do.
        # tau is a per-OUTPUT-pixel weight, so unlike Δr it cannot be folded
        # into the spectral deriv_factor -- see the *mc methods at the end.
        self.tau_y = (cp.arange(nz, dtype='float32') - (nz - 1) * 0.5)
        self.tau_x = (cp.arange(n , dtype='float32') - (n  - 1) * 0.5)

        # ---- Chirp-z (Bluestein) precomputations -----------------------
        # Used when any m[t] != 1. Per axis:
        #   κ        — signed fftfreq integer    (length N_in)
        #   κ²       — for the pre-chirp
        #   ty²      — for the post-chirp        (length N_out)
        #   (j-N//2)² — for the kernel h         (length N_in + N_out - 1)
        # L is the Bluestein FFT length, next power of two of N_in+N_out-1.
        self.L_x = self._next_pow2(npsi + n  - 1)
        self.L_y = self._next_pow2(nzpsi + nz - 1)
        self.k_signed_x = (cp.fft.fftfreq(npsi ) * cp.float32(npsi )).astype('float32')
        self.k_signed_y = (cp.fft.fftfreq(nzpsi) * cp.float32(nzpsi)).astype('float32')
        self.k_sq_x  = (self.k_signed_x ** 2).astype('float32')
        self.k_sq_y  = (self.k_signed_y ** 2).astype('float32')
        self.ty_sq_x = (cp.arange(n , dtype='float32') ** 2).astype('float32')
        self.ty_sq_y = (cp.arange(nz, dtype='float32') ** 2).astype('float32')
        self.j_sq_x  = ((cp.arange(npsi  + n  - 1, dtype='float32') - npsi  // 2) ** 2).astype('float32')
        self.j_sq_y  = ((cp.arange(nzpsi + nz - 1, dtype='float32') - nzpsi // 2) ** 2).astype('float32')

        # Back direction (Sback: small grid -> large grid).  Same Bluestein
        # machinery with the roles of the two grids swapped, so it needs its
        # own tables; see Sback for the change of variables.
        self.L_xb = self._next_pow2(n  + npsi  - 1)
        self.L_yb = self._next_pow2(nz + nzpsi - 1)
        self.k_signed_xb = (cp.fft.fftfreq(n ) * cp.float32(n )).astype('float32')
        self.k_signed_yb = (cp.fft.fftfreq(nz) * cp.float32(nz)).astype('float32')
        self.k_sq_xb  = (self.k_signed_xb ** 2).astype('float32')
        self.k_sq_yb  = (self.k_signed_yb ** 2).astype('float32')
        self.ty_sq_xb = (cp.arange(npsi , dtype='float32') ** 2).astype('float32')
        self.ty_sq_yb = (cp.arange(nzpsi, dtype='float32') ** 2).astype('float32')
        self.j_sq_xb  = ((cp.arange(n  + npsi  - 1, dtype='float32') - n  // 2) ** 2).astype('float32')
        self.j_sq_yb  = ((cp.arange(nz + nzpsi - 1, dtype='float32') - nz // 2) ** 2).astype('float32')

        self._unit_mag = {}          # ntheta -> owned ones((ntheta, 2))

        # Match Shift's coeff cache surface so this class is drop-in.
        self.coeff_cache  = {}
        self.coeff_hits   = 0
        self.coeff_misses = 0

    @staticmethod
    def _next_pow2(n):
        """Smallest 2-3-5-7-smooth integer >= n -- cuFFT's fast radices.

        Bluestein only needs L >= N_in + N_out - 1 for the circular
        convolution to equal the linear one; rounding up to a power of two is
        an FFT-efficiency habit, not a correctness requirement, and cuFFT is
        just as happy with any 2/3/5/7-smooth length.  These grids land one
        past a smooth number almost exactly -- 319 -> 320 instead of 512,
        2559 -> 2560 instead of 4096 -- so this is ~1.6x off every Bluestein
        transform.  (Name kept: it is what the tables call.)
        """
        if n <= 1:
            return 1
        best = 1 << (n - 1).bit_length()           # the power of two, as a bound
        p5 = 1
        while p5 < best:
            p7 = p5
            while p7 < best:
                p3 = p7
                while p3 < best:
                    v = p3
                    while v < n:                   # climb in 2s to reach n
                        v *= 2
                    best = min(best, v)
                    p3 *= 3
                p7 *= 7
            p5 *= 5
        return best

    def unit_mag(self, ntheta):
        """Cached ones((ntheta, 2)) for callers that never magnify.

        This object owns the buffer for the operator's lifetime, so it is never
        freed and `m is` cannot alias a later array -- which is what makes the
        identity fast path in _is_unit_mag sound.
        """
        m = self._unit_mag.get(ntheta)
        if m is None:
            m = cp.ones((ntheta, 2), dtype='float32')
            self._unit_mag[ntheta] = m
        return m

    def _is_unit_mag(self, m):
        """All entries of m equal to 1 → fast (FFT-shift) path; else chirp-z.

        m is shape (ntheta, 2) — axis 1 is (my, mx).

        The device comparison below syncs, and it sits at the top of every
        S/Sadj/dcurlySc call, so a buffer from unit_mag() short-circuits it.
        """
        if m is self._unit_mag.get(m.shape[0]):
            return True
        m_arr = cp.asarray(m)
        # Tolerant comparison so float32 1.0 hits the fast path even if the
        # caller built m as numpy float64.
        return bool(cp.all(cp.abs(m_arr - 1) < 1e-7))

    # ------------------------------------------------------------------
    # FFT helpers
    # ------------------------------------------------------------------

    def fft2(self, x):
        if self.plan is not None and x.shape[0] == self.plan_batch:
            with self.plan:
                return cufft.fft2(x)
        return cp.fft.fft2(x)

    def ifft2(self, x):
        if self.plan is not None and x.shape[0] == self.plan_batch:
            with self.plan:
                return cufft.ifft2(x)
        return cp.fft.ifft2(x)

    def phase_separable(self, r):
        """Per-theta phase exp(-i (fy*(ry - eff_dy) + fx*(rx - eff_dx))) as
        separable factors py[ntheta, nzpsi] and px[ntheta, npsi]."""
        r = cp.asarray(r)
        ry_eff = (r[:, 0] - self.eff_dy).astype('float32')
        rx_eff = (r[:, 1] - self.eff_dx).astype('float32')
        py = cp.exp((-1j) * self.fy[cp.newaxis, :] * ry_eff[:, cp.newaxis]).astype('complex64')
        px = cp.exp((-1j) * self.fx[cp.newaxis, :] * rx_eff[:, cp.newaxis]).astype('complex64')
        return py, px

    def to_complex(self, x):
        return x if x.dtype == cp.complex64 else x.astype('complex64')

    def from_complex(self, x):
        # A fresh contig array, so the caller does not hold the parent ifft
        # buffer alive through a view.
        return cp.ascontiguousarray(x)

    def pad_output(self, g):
        """Place output g (nz, n) at the top-left of the FFT grid. Used by
        adjoint paths; phase folds in the eff_d offsets so the forward output
        appears at top-left."""
        gc = self.to_complex(ascontig(g))
        padded = cp.zeros([gc.shape[0], self.nzpsi, self.npsi], dtype='complex64')
        padded[:, :self.nz, :self.n] = gc
        return padded

    def deriv_factor(self, Delta):
        """D[t,y,x] = -i (fy[y]·Δy[t] + fx[x]·Δx[t]) on the FFT grid.
        Uses precomputed -i·fy / -i·fx; the sum is built in one expression
        so the full [ntheta, nzpsi, npsi] tensor materializes once
        (in-place += would fail because the (1, nzpsi, 1)+(1, 1, npsi)
        broadcast target shape doesn't match either operand's allocation)."""
        Delta = cp.asarray(Delta)
        Dy = Delta[:, 0].astype('complex64')      # [ntheta]
        Dx = Delta[:, 1].astype('complex64')
        return (self.negi_fy[cp.newaxis, :, cp.newaxis] * Dy[:, cp.newaxis, cp.newaxis]
              + self.negi_fx[cp.newaxis, cp.newaxis, :] * Dx[:, cp.newaxis, cp.newaxis])

    # ------------------------------------------------------------------
    # Chirp-z (Bluestein) magnification — separable per axis
    # ------------------------------------------------------------------
    # All four helpers below evaluate / adjoint-evaluate the band-limited
    # Fourier interpolant of the input grid at the m-scaled, r-shifted
    # sample positions
    #
    #     y_{t,ty} = m[t] · ty + b[t],     ty = 0 … N_out − 1
    #
    # which, after folding the s_kernel center convention into b[t],
    # equals  m[t]·(ty − (N_out−1)/2) − r[t] + (N_in−1)/2.
    #
    # Math:
    #   g[t,ty] = (1/N_in) Σ_κ C[t,κ] · exp(2πi κ (m·ty + b)/N_in)
    #   Identity 2κ·ty = (κ+ty)² − κ² − ty² gives, with β = m/N_in:
    #     g[t,ty] = (1/N_in) e^(−iπ β ty²) · Σ_κ [C[t,κ] · e^(2πi κ b/N_in)
    #                                              · e^(−iπ β κ²)] · e^(iπ β (κ+ty)²)
    #
    # The inner sum is a cross-correlation with the chirp kernel
    # h[j] = exp(iπ β j²), j = (κ+ty), evaluated via FFT-convolution of
    # length L = next_pow2(N_in + N_out − 1).

    def _chirpz_lastaxis(self, X, m, b, axis_xy, adjoint, spectrum=False):
        """Chirp-z transform along the last axis.

        `spectrum` says X is ALREADY the DFT along the last axis, so step 1
        below is skipped.  The derivative paths hold fft2(c) and would
        otherwise inverse-transform it only for this to transform it straight
        back.

        X:        complex64 array, shape [B, K, N_in] for adjoint=False
                  or [B, K, N_out] for adjoint=True (B = ntheta, K may be 1).
        m:        [B] float32 — per-projection magnification.
        b:        [B] float32 — per-projection linear shift offset
                  (= (N_in−1)/2 − r − m·(N_out−1)/2).
        axis_xy:  'x' or 'y' — picks which precomputed index arrays to use.
        adjoint:  True for the L²-adjoint of the forward chirp-z.

        Returns [B, K, N_out] (forward) or [B, K, N_in] (adjoint).
        """
        if axis_xy == 'x':
            N_in, N_out, L = self.npsi,  self.n,  self.L_x
            k_signed = self.k_signed_x
            k_sq, ty_sq, j_sq = self.k_sq_x, self.ty_sq_x, self.j_sq_x
        elif axis_xy == 'y':
            N_in, N_out, L = self.nzpsi, self.nz, self.L_y
            k_signed = self.k_signed_y
            k_sq, ty_sq, j_sq = self.k_sq_y, self.ty_sq_y, self.j_sq_y
        elif axis_xy == 'xb':                      # back: small -> large
            N_in, N_out, L = self.n,  self.npsi,  self.L_xb
            k_signed = self.k_signed_xb
            k_sq, ty_sq, j_sq = self.k_sq_xb, self.ty_sq_xb, self.j_sq_xb
        else:                                      # 'yb'
            N_in, N_out, L = self.nz, self.nzpsi, self.L_yb
            k_signed = self.k_signed_yb
            k_sq, ty_sq, j_sq = self.k_sq_yb, self.ty_sq_yb, self.j_sq_yb

        B = X.shape[0]
        K = X.shape[1]
        beta = (cp.asarray(m).astype('float32') / cp.float32(N_in))   # [B]
        b    = cp.asarray(b).astype('float32')                        # [B]

        # Pre-twist (fftfreq order):
        #   pre[t,k] = exp(2πi κ b/N_in − iπ β κ²)
        # combined into one elementwise exp.
        phase_pre = ((2.0 * cp.pi / N_in) * b[:, None] * k_signed[None, :]
                     - cp.pi * beta[:, None] * k_sq[None, :]).astype('float32')   # [B, N_in]
        pre_twist = cp.exp(1j * phase_pre).astype('complex64')                    # [B, N_in]

        # Post-chirp at output sample positions ty = 0 … N_out − 1:
        #   post[t,ty] = exp(−iπ β ty²)
        post = cp.exp(-1j * cp.pi * beta[:, None] * ty_sq[None, :]).astype('complex64')  # [B, N_out]

        # Bluestein kernel h on physical index j_phys = idx − N_in//2:
        #   h[t,idx] = exp(+iπ β j_phys²),  idx = 0 … N_in+N_out−2
        # zero-padded to length L for circular FFT-correlation.
        h = cp.exp(1j * cp.pi * beta[:, None] * j_sq[None, :]).astype('complex64')   # [B, J]
        h_pad = cp.zeros((B, L), dtype='complex64')
        h_pad[:, : N_in + N_out - 1] = h
        H_hat = cp.fft.fft(h_pad, axis=-1)                                            # [B, L]

        if not adjoint:
            # Forward: X[B,K,N_in] → g[B,K,N_out]
            # 1) FFT along last axis -- unless the caller already has it.
            C = X if spectrum else cp.fft.fft(X, axis=-1)                             # [B, K, N_in]
            # 2) Pre-twist (shift phase + pre-chirp).
            a = C * pre_twist[:, None, :]
            # 3) fftshift along last axis so κ = idx − N_in//2 in centered storage.
            a = cp.fft.fftshift(a, axes=-1)
            # 4) Zero-pad to L.
            a_pad = cp.zeros((B, K, L), dtype='complex64')
            a_pad[:, :, :N_in] = a
            # 5) Compute Σ_n a_pad[n] · h_pad[n+ty] for ty=0..L−1 via FFT.
            #    Derivation (with ω = exp(2πi/L)):
            #      h[m] = (1/L) Σ_j H_hat[j] ω^(jm)  ⇒
            #      Σ_n a_pad[n] · h[n+ty]
            #         = (1/L) Σ_j H_hat[j] · ω^(jty) · Σ_n a_pad[n] ω^(jn)
            #         = ifft( H_hat · A_pos )[ty],  A_pos[j] = Σ_n a_pad[n] ω^(jn).
            #    A_pos is cupy's IFFT with norm='forward' (no 1/L, +sign).
            #    NOTE: the apparent "ifft(conj(FFT(a))·FFT(h))" identity is
            #    only correct for REAL a — for complex a the conj sticks on
            #    the data instead of doing nothing.
            A_pos = cp.fft.ifft(a_pad, axis=-1, norm='forward')
            corr  = cp.fft.ifft(A_pos * H_hat[:, None, :], axis=-1)                   # [B, K, L]
            # 6) Extract first N_out, post-multiply, scale by 1/N_in.
            g = corr[:, :, :N_out] * post[:, None, :] * cp.float32(1.0 / N_in)
            return g

        # Adjoint: g[B,K,N_out] → X[B,K,N_in]
        # Apply the L²-adjoint of each forward step in reverse order.
        # cupy adjoints (sum-of-conj·· inner product):
        #     (fft)*    = ifft(·, norm='forward'),
        #     (ifft)*   = fft(·,  norm='forward'),
        #     (ifft_fwd)* = fft(·)  (since ifft_fwd = N·ifft, adjoint cancels N).
        g = X
        # 8*) g · conj(post) · (1/N_in)
        scaled = g * cp.conj(post)[:, None, :] * cp.float32(1.0 / N_in)               # [B, K, N_out]
        # 7*) Zero-pad to length L (place at the front).
        corr_adj = cp.zeros((B, K, L), dtype='complex64')
        corr_adj[:, :, :N_out] = scaled
        # 6*) Adjoint of  corr = ifft(A_pos · H_hat) :
        #     A_pos_adj = conj(H_hat) · fft(corr_adj, norm='forward')
        A_pos_adj = (cp.conj(H_hat)[:, None, :]
                     * cp.fft.fft(corr_adj, axis=-1, norm='forward'))
        # 5*) Adjoint of  A_pos = ifft(a_pad, norm='forward'):
        a_pad_adj = cp.fft.fft(A_pos_adj, axis=-1)
        # 4*) Adjoint of zero-pad: truncate to N_in.
        a_centered_adj = a_pad_adj[:, :, :N_in]                                       # [B, K, N_in]
        # 3*) ifftshift (adjoint of fftshift).
        a_adj = cp.fft.ifftshift(a_centered_adj, axes=-1)
        # 2*) Adjoint of multiply by twist: multiply by conj(twist).
        C_adj = a_adj * cp.conj(pre_twist)[:, None, :]
        # 1*) Adjoint of FFT = ifft with norm='forward'.
        X_adj = cp.fft.ifft(C_adj, axis=-1, norm='forward')
        return X_adj

    def _chirpz_2d(self, X, m, ry, rx, adjoint, spectrum=False):
        """Separable 2-D chirp-z. Calls the last-axis helper twice — once
        along x, once along y — with appropriate axis swapping to keep
        FFTs on the contiguous last axis.

        m is (ntheta, 2) — axis 1 is (my, mx). Each axis's chirp-z receives
        its own magnification.
        """
        m  = cp.asarray(m).astype('float32')
        my = cp.ascontiguousarray(m[:, 0])
        mx = cp.ascontiguousarray(m[:, 1])
        ry = cp.asarray(ry).astype('float32')
        rx = cp.asarray(rx).astype('float32')
        b_x = (cp.float32((self.npsi  - 1) * 0.5) - rx - mx * cp.float32((self.n  - 1) * 0.5))
        b_y = (cp.float32((self.nzpsi - 1) * 0.5) - ry - my * cp.float32((self.nz - 1) * 0.5))

        if not adjoint:
            # Forward order: x first (last axis), then y (last axis after swap).
            # With `spectrum`, X is fft2(c): the x axis is already transformed,
            # and so is y -- the x pass only touches the last axis, so the y
            # axis is still a spectrum when the y pass receives it.  Both
            # leading FFTs drop out.
            X = self.to_complex(ascontig(X))                       # [B, nzpsi, npsi]
            Y = self._chirpz_lastaxis(X, mx, b_x, 'x', adjoint=False,
                                      spectrum=spectrum)                # [B, nzpsi, n]
            Y = cp.ascontiguousarray(cp.swapaxes(Y, -2, -1))            # [B, n, nzpsi]
            Y = self._chirpz_lastaxis(Y, my, b_y, 'y', adjoint=False,
                                      spectrum=spectrum)                # [B, n, nz]
            Y = cp.ascontiguousarray(cp.swapaxes(Y, -2, -1))            # [B, nz, n]
            return Y

        # Adjoint: reverse the order — y_adj first (on a swapped view),
        # then x_adj.
        X = self.to_complex(ascontig(X))                                # [B, nz, n]
        Y = cp.ascontiguousarray(cp.swapaxes(X, -2, -1))                # [B, n, nz]
        Y = self._chirpz_lastaxis(Y, my, b_y, 'y', adjoint=True)         # [B, n, nzpsi]
        Y = cp.ascontiguousarray(cp.swapaxes(Y, -2, -1))                # [B, nzpsi, n]
        Y = self._chirpz_lastaxis(Y, mx, b_x, 'x', adjoint=True)         # [B, nzpsi, npsi]
        return Y

    # ------------------------------------------------------------------
    # Coefficient-space cache (identity coeff, but API-compatible)
    # ------------------------------------------------------------------

    def coeff(self, psi):
        """FFT shift needs no B-spline prefilter — identity."""
        return psi

    def coeff_cached(self, psi):
        key = id(psi)
        cached = self.coeff_cache.get(key)
        if cached is None:
            self.coeff_misses += 1
            cached = self.coeff(psi)
            self.coeff_cache[key] = cached
        else:
            self.coeff_hits += 1
        return cached

    def coeff_cache_reset(self):
        self.coeff_cache = {}

    def coeff_cache_stats(self, reset=False):
        stats = (self.coeff_hits, self.coeff_misses)
        if reset:
            self.coeff_hits = 0
            self.coeff_misses = 0
        return stats

    # ------------------------------------------------------------------
    # Forward / adjoint shift  S / S*
    # ------------------------------------------------------------------

    def S(self, c, r, m):
        if not self._is_unit_mag(m):
            # Chirp-z magnification path.
            r = cp.asarray(r)
            out = self._chirpz_2d(c, m, r[:, 0], r[:, 1], adjoint=False)
            return self.from_complex(out)
        py, px = self.phase_separable(r)
        C = self.fft2(self.to_complex(ascontig(c)))
        # Fused: C = C · py · px (single elementwise kernel, in-place).
        apply_sep_phase(C, py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])

    def Sadj(self, spsi, r, m):
        if not self._is_unit_mag(m):
            # Chirp-z magnification path (adjoint).
            r = cp.asarray(r)
            out = self._chirpz_2d(spsi, m, r[:, 0], r[:, 1], adjoint=True)
            return self.from_complex(out)
        py, px = self.phase_separable(r)
        S = self.fft2(self.pad_output(spsi))
        # Fused: S = S · conj(py) · conj(px), in-place.
        apply_sep_phase(S, cp.conj(py)[:, :, cp.newaxis],
                           cp.conj(px)[:, cp.newaxis, :], S)
        return self.from_complex(self.ifft2(S))

    def curlyS(self, psi, r, m):
        return self.S(psi, r, m)   # coeff is identity in FFT mode

    # ------------------------------------------------------------------
    # Back-projection shift  (small grid -> large grid), for the Paganin
    # initial guess and step 5's stitch.  This is NOT Sadj: it resamples in
    # the opposite direction rather than transposing the operator.
    # ------------------------------------------------------------------

    def coeff_back(self, psi):
        return psi                 # no B-spline prefilter in FFT mode

    def _chirpz_2d_back(self, X, m, ry, rx):
        """Separable 2-D chirp-z for the back direction.

        Sback samples the small grid at
            u = (t - (N_big-1)/2 + r) / m + (N_small-1)/2
        which is the forward form  u = m'(t - (N_out-1)/2) - r' + (N_in-1)/2
        with N_in = N_small, N_out = N_big, m' = 1/m and r' = -r/m.
        """
        m = cp.asarray(m).astype('float32')
        my = cp.ascontiguousarray(1.0 / m[:, 0])
        mx = cp.ascontiguousarray(1.0 / m[:, 1])
        ry = -cp.asarray(ry).astype('float32') * my
        rx = -cp.asarray(rx).astype('float32') * mx
        b_x = (cp.float32((self.n  - 1) * 0.5) - rx - mx * cp.float32((self.npsi  - 1) * 0.5))
        b_y = (cp.float32((self.nz - 1) * 0.5) - ry - my * cp.float32((self.nzpsi - 1) * 0.5))

        X = self.to_complex(ascontig(X))                            # [B, nz, n]
        Y = self._chirpz_lastaxis(X, mx, b_x, 'xb', adjoint=False)  # [B, nz, npsi]
        Y = cp.ascontiguousarray(cp.swapaxes(Y, -2, -1))            # [B, npsi, nz]
        Y = self._chirpz_lastaxis(Y, my, b_y, 'yb', adjoint=False)  # [B, npsi, nzpsi]
        return cp.ascontiguousarray(cp.swapaxes(Y, -2, -1))         # [B, nzpsi, npsi]

    def Sback(self, c, r, m):
        """Interpolate from the small (nz, n) grid to the large (nzpsi, npsi) one.

        At m = 1 this is a placement plus a Fourier shift by +r, which is
        exact for a band-limited input; otherwise it goes through the
        back-direction chirp-z.

        DIFFERS FROM Shift.Sback OUTSIDE THE SMALL GRID.  Where the back-map
        lands beyond [0, n) the B-spline kernel drops the tap and leaves
        zero, while this one is periodic and wraps.  Inside, the two agree to
        ~1e-4 on a band-limited input.  At m < 1 the large grid reaches well
        outside the small one, so the caller has to mask or weight the result
        (the demo stitch divides by the per-pixel coverage).
        """
        if not self._is_unit_mag(m):
            return self.from_complex(self._chirpz_2d_back(c, m, cp.asarray(r)[:, 0],
                                                          cp.asarray(r)[:, 1]))
        # Place the frame where Sadj places it, then shift by +r rather than
        # -r: pad_output already folds in the (nzpsi-nz)/2, (npsi-n)/2 offset.
        py, px = self.phase_separable(cp.asarray(r))
        C = self.fft2(self.pad_output(c))
        apply_sep_phase(C, cp.conj(py)[:, :, cp.newaxis],
                           cp.conj(px)[:, cp.newaxis, :], C)
        return self.from_complex(self.ifft2(C))

    def curlySback(self, psi, r, m):
        return self.Sback(self.coeff_back(psi), r, m)

    # ------------------------------------------------------------------
    # Coefficient-space variants
    # ------------------------------------------------------------------

    def curlySc(self, c, r, m):
        return self.S(c, r, m)

    def dcurlySc(self, c, r, m, c1, Deltar):
        """∂S(c,r)/∂c · c1 + ∂S(c,r)/∂r · Δr."""
        if not self._is_unit_mag(m):
            # Chirp-z magnification path. By linearity of S in its input,
            #   dcurlySc = S(c1 + Δry·∂c/∂y + Δrx·∂c/∂x, r, m)
            # and the spatial derivatives ∂c/∂(y,x) are computed via FFT
            # differentiation:  ∂c/∂y = ifft2(fft2(c) · −i·fy), and the same
            # for x. Both directional contributions fuse into one ifft2 by
            # using the existing `deriv_factor` (Δry·−i·fy + Δrx·−i·fx).
            c  = self.to_complex(ascontig(c))
            c1 = self.to_complex(ascontig(c1))
            D  = self.deriv_factor(Deltar)
            # Sum in the SPECTRUM and let the chirp-z consume it: the old
            # form ifft2'd here only for S to fft it straight back.
            combined = self.fft2(c1) + self.fft2(c) * D
            r = cp.asarray(r)
            return cp.ascontiguousarray(
                self._chirpz_2d(combined, m, r[:, 0], r[:, 1], adjoint=False,
                                spectrum=True).astype('complex64'))

        py, px = self.phase_separable(r)
        C  = self.fft2(self.to_complex(ascontig(c)))
        C1 = self.fft2(self.to_complex(ascontig(c1)))
        D  = self.deriv_factor(Deltar)
        # Fused: out = (C·D + C1)·py·px, in-place into C, no intermediates.
        combine_dcurlySc(C, D, C1,
                         py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        del D, C1
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])

    def dcurlySadjc(self, c, r, m, Deltaphi):
        """Adjoint of (c1, Δr) → dcurlySc(c, r, m, c1, Δr) applied to Δφ.
        Returns [out1, out2] where out1 = Sadj(Δφ) and
        out2[t, 0/1] = redot(Δφ, ∂S/∂(ry/rx)·c)."""
        if not self._is_unit_mag(m):
            # Chirp-z magnification path.
            #   out1 = Sadj(Δφ).
            #   out2[t,i] = redot(Δφ, S(∂c/∂rᵢ, r, m))  with the spatial
            # derivatives ∂c/∂y, ∂c/∂x computed by FFT differentiation.
            ntheta_loc = c.shape[0]
            Deltaphi_c = self.to_complex(ascontig(Deltaphi))
            out1 = self.Sadj(Deltaphi_c, r, m)

            C = self.fft2(self.to_complex(ascontig(c)))
            d_y_c = self.ifft2(C * self.negi_fy[None, :, None])
            d_x_c = self.ifft2(C * self.negi_fx[None, None, :])
            del C
            dy = self.S(d_y_c, r, m)
            dx = self.S(d_x_c, r, m)

            out2 = cp.empty([ntheta_loc, 2], dtype='float32')
            out2[:, 0] = redot(Deltaphi_c, dy, axis=(1, 2))
            out2[:, 1] = redot(Deltaphi_c, dx, axis=(1, 2))
            return [out1, out2]

        ntheta = c.shape[0]
        py, px = self.phase_separable(r)
        py_b = py[:, :, cp.newaxis]
        px_b = px[:, cp.newaxis, :]

        # fft(ZeroPad(Δφ)) on the internal grid — reused for out1 (Sadj) and
        # out2 (Parseval redots). Kept around past out1 since out2 needs it.
        Phat = self.fft2(self.pad_output(Deltaphi))

        # out1 = Sadj(Δφ). Copy Phat into Sbuf so the original survives.
        Sbuf = Phat.copy()
        apply_sep_phase(Sbuf, cp.conj(py_b), cp.conj(px_b), Sbuf)
        out1 = self.from_complex(self.ifft2(Sbuf))
        del Sbuf

        # Cshift = fft(c) · phase, in-place
        Cshift = self.fft2(self.to_complex(ascontig(c)))
        apply_sep_phase(Cshift, py_b, px_b, Cshift)

        # PhatC_im = Im(conj(Phat) · Cshift), via fused float-arithmetic
        # kernel — avoids the big complex intermediate cp.conj(Phat)*Cshift.
        PhatC_im = imag_conj_prod(Phat, Cshift)
        del Phat, Cshift

        # out2 via Parseval, separable sum-then-dot to avoid the huge
        # fy[None,:,None]*PhatC_im broadcast temp the naive form would build.
        #   out2[t,0] = inv_N · Σ_y fy[y] · Σ_x PhatC_im[t,y,x]
        #   out2[t,1] = inv_N · Σ_x fx[x] · Σ_y PhatC_im[t,y,x]
        sum_x = PhatC_im.sum(axis=2)   # [ntheta, nzpsi]
        sum_y = PhatC_im.sum(axis=1)   # [ntheta, npsi]
        del PhatC_im

        out2 = cp.empty([ntheta, 2], dtype='float32')
        out2[:, 0] = self.inv_N * (sum_x @ self.fy)
        out2[:, 1] = self.inv_N * (sum_y @ self.fx)
        return [out1, out2]

    def d2curlySc(self, c, r, m, c1, Deltar1, c2, Deltar2):
        """Second directional derivative on (c, r), m constant.  ∂²S/∂c² = 0, so

            Crop(ifft((C·D1·D2 + C1·D1 + C2·D2)·P))

        SLOT PAIRING -- the caller must CROSS the coefficients, exactly as for
        Shift.d2curlySc.  Like d2s_kernel, this contracts each coefficient with
        the shift in its OWN slot, i.e. it evaluates

            d2/dr2 (c)[Δ1, Δ2]  +  d/dr (c1)[Δ1]  +  d/dr (c2)[Δ2]

        while the mixed second differential along y = (c_y, dr_y) and
        z = (c_z, dr_z) needs each coefficient against the OTHER direction's
        shift, so a caller wanting B(y, z) passes

            d2curlySc(c, r, m, c_z, dr_y, c_y, dr_z)

        This is the one place where this class deliberately departs from the
        holotomocupy `mpi` branch's ShiftFFT, which folds the crossing into the
        formula instead.  Matching the kernel is what makes ShiftFFT a drop-in
        for Shift here: RecNFP.d2F_dF3 and Rec.d2F_dF3 already cross, and
        on the diagonal (y is z) the two conventions agree, so only an
        off-diagonal check -- the polarization identity
        B(y+z, y+z) == B(y,y) + 2 B(y,z) + B(z,z) -- can see the difference."""
        if not self._is_unit_mag(m):
            # Chirp-z magnification path. By linearity of S, the same algebra
            # as the m=1 spectral combiner holds, applied to a single
            # spatial "combined input":
            #   d2curlySc = S(ifft2(C·D1·D2 + C1·D1 + C2·D2), r, m).
            C  = self.fft2(self.to_complex(ascontig(c )))
            C1 = self.fft2(self.to_complex(ascontig(c1)))
            C2 = self.fft2(self.to_complex(ascontig(c2)))
            D1 = self.deriv_factor(Deltar1)
            D2 = self.deriv_factor(Deltar2)
            combined = C * D1 * D2 + C1 * D1 + C2 * D2
            del C, C1, C2, D1, D2
            # straight into the chirp-z as a spectrum, no ifft2/fft2 pair
            r = cp.asarray(r)
            return cp.ascontiguousarray(
                self._chirpz_2d(combined, m, r[:, 0], r[:, 1], adjoint=False,
                                spectrum=True).astype('complex64'))

        py, px = self.phase_separable(r)
        D1 = self.deriv_factor(Deltar1)
        D2 = self.deriv_factor(Deltar2)
        C  = self.fft2(self.to_complex(ascontig(c)))
        C1 = self.fft2(self.to_complex(ascontig(c1)))
        C2 = self.fft2(self.to_complex(ascontig(c2)))
        # Fused: out = (C·D1·D2 + C1·D1 + C2·D2)·py·px, in-place into C.
        # Own-slot pairing -- see the SLOT PAIRING note above.
        combine_d2curlySc(C, C1, C2, D1, D2,
                          py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        del D1, D2, C1, C2
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])

    # ------------------------------------------------------------------
    # Magnification-aware variants: same as the *c family above, but the
    # magnification m is a differentiable input rather than a constant.
    # Used when the shrinkage (and hence the effective demagnification) is a
    # reconstruction variable -- see Rec.F4 / Rec.d2F_dF3 in rec_mpi.py.
    #
    # WHY THESE CANNOT BE FUSED THE WAY THE *c FAMILY IS.  Every derivative
    # here is still a derivative along the interpolation coordinate u, and
    #     ∂/∂r_a = −∂/∂u_a          ∂/∂m_a = +tau_a(t) · ∂/∂u_a
    # so the whole m-dependence is the per-pixel effective direction
    #     e_a(t, pixel) = Δr_a(t) − tau_a(pixel) · Δm_a(t).
    # Shift's CUDA kernels evaluate the taps pixel by pixel, so they just
    # rebuild e_a inside the loop and stay at one launch.  ShiftFFT applies
    # Δr as a SPECTRAL factor on the (nzpsi × npsi) input grid, which is only
    # possible because Δr is constant over the output; tau is not.  So the
    # derivative fields have to be shifted onto the output grid FIRST and
    # weighted by e_a there, which costs one extra ifft2 per field.
    #
    # Δm == 0 is therefore short-circuited back to the fused *c methods.
    # That is the common case: every config with rho[tp] = 0 (shrinkage off)
    # takes it, and pays exactly what the cubic path pays.
    # ------------------------------------------------------------------

    @staticmethod
    def _is_zero(Delta):
        """True when a direction vector is identically zero -- lets the m-aware
        methods fall back to their fused, m-free counterparts."""
        return not bool(cp.any(cp.asarray(Delta) != 0))

    def _shift_spec(self, C, r, m):
        """S applied to an input whose SPECTRUM on the (nzpsi × npsi) grid is C.

        Saves the fft2 that S(ifft2(C)) would redo on the unit-magnification
        path; the chirp-z path has to come back to real space anyway.

        CONSUMES C -- it is overwritten in place on the m = 1 path.  Every
        caller below passes a freshly built temporary.

        Always complex64: these are derivative fields multiplied by complex
        per-pixel weights, as in every Shift curlySc variant."""
        if not self._is_unit_mag(m):
            # C IS the spectrum: hand it to the chirp-z directly instead of
            # ifft2-ing it so that S can fft it straight back.  Saves the
            # ifft2 pair and the leading FFT of each chirp-z axis -- four
            # transforms of the ten this call used to cost.
            r = cp.asarray(r)
            return cp.ascontiguousarray(
                self._chirpz_2d(C, m, r[:, 0], r[:, 1], adjoint=False,
                                spectrum=True).astype('complex64'))
        py, px = self.phase_separable(r)
        apply_sep_phase(C, py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        s = self.ifft2(C)
        return cp.ascontiguousarray(s[:, :self.nz, :self.n])

    def _eff_dirs(self, Deltar, Deltam):
        """Per-pixel effective r-direction e_a = Δr_a − tau_a · Δm_a, returned
        as broadcastable [ntheta, nz, 1] and [ntheta, 1, n] factors."""
        Deltar = cp.asarray(Deltar)
        Deltam = cp.asarray(Deltam)
        ey = (Deltar[:, 0, cp.newaxis] - self.tau_y[cp.newaxis, :]
              * Deltam[:, 0, cp.newaxis]).astype('complex64')[:, :, cp.newaxis]
        ex = (Deltar[:, 1, cp.newaxis] - self.tau_x[cp.newaxis, :]
              * Deltam[:, 1, cp.newaxis]).astype('complex64')[:, cp.newaxis, :]
        return ey, ex

    def _grad_fields(self, C, r, m):
        """(G_y, G_x) on the output grid, G_a = ∂curlySc(c, r, m)/∂r_a, given
        the input spectrum C = fft2(c)."""
        Gy = self._shift_spec(C * self.negi_fy[cp.newaxis, :, cp.newaxis], r, m)
        Gx = self._shift_spec(C * self.negi_fx[cp.newaxis, cp.newaxis, :], r, m)
        return Gy, Gx

    def dcurlySmc(self, c, r, m, c1, Deltar, Deltam):
        """Single-pass (c, r, m) directional derivative:
            curlySc(c1, r, m)
          + d/dr curlySc(c, r, m) · Δr
          + d/dm curlySc(c, r, m) · Δm

        Same signature as Shift.dcurlySmc: Deltam is (chunk, 2), axis 1 is
        (my, mx).  Equals dcurlySc with Δr replaced by the per-pixel effective
        direction Δr − tau·Δm."""
        if self._is_zero(Deltam):
            return self.dcurlySc(c, r, m, c1, Deltar)

        C   = self.fft2(self.to_complex(ascontig(c)))
        out = self._shift_spec(self.fft2(self.to_complex(ascontig(c1))), r, m)
        Gy, Gx = self._grad_fields(C, r, m)
        del C
        ey, ex = self._eff_dirs(Deltar, Deltam)
        Gy *= ey
        out += Gy
        del Gy
        Gx *= ex
        out += Gx
        del Gx
        return out

    def dcurlySadjmc(self, c, r, m, Deltaphi):
        """Adjoint of dcurlySmc.  Returns [out1, out2_r, out2_m] with
            <dcurlySmc(c, r, m, c1, Δr, Δm), g>
              = <c1, out1> + <Δr, out2_r> + <Δm, out2_m>.

        out2_m[t, a] = −Σ_pixels tau_a(pixel) · Re(conj(g) · G_a), i.e. the same
        reduction as out2_r with the tau weight in front -- see dsmadj_kernel,
        which writes dtm = −tau · dt for exactly this reason."""
        ntheta = c.shape[0]
        Deltaphi_c = self.to_complex(ascontig(Deltaphi))

        out1 = self.Sadj(Deltaphi_c, r, m)

        C = self.fft2(self.to_complex(ascontig(c)))
        Gy, Gx = self._grad_fields(C, r, m)
        del C

        out2_r = cp.empty([ntheta, 2], dtype='float32')
        out2_m = cp.empty([ntheta, 2], dtype='float32')

        # Re(conj(Δφ)·G_a) once per axis, then two reductions over it: the
        # plain sum is out2_r, the tau-weighted sum is −out2_m.
        Ry = (Deltaphi_c.conj() * Gy).real
        del Gy
        out2_r[:, 0] = Ry.sum(axis=(1, 2))
        out2_m[:, 0] = -(Ry.sum(axis=2) @ self.tau_y)
        del Ry

        Rx = (Deltaphi_c.conj() * Gx).real
        del Gx
        out2_r[:, 1] = Rx.sum(axis=(1, 2))
        out2_m[:, 1] = -(Rx.sum(axis=1) @ self.tau_x)
        del Rx

        return [out1, out2_r, out2_m]

    def d2curlySmc(self, c, r, m, c1, Deltar1, Deltam1, c2, Deltar2, Deltam2):
        """Second directional derivative on (c, r, m).  With ∂²S/∂c² = 0 and
        e_a = Δr_a − tau_a·Δm_a the per-pixel effective direction,

            Σ_ab H_ab · e1_a · e2_b  +  Σ_a G_a(c1)·e1_a  +  Σ_a G_a(c2)·e2_a

        where G_a = ∂S/∂r_a and H_ab = ∂²S/∂r_a∂r_b.

        SLOT PAIRING -- the caller must CROSS the coefficients, exactly as for
        d2curlySc and Shift.d2curlySmc: each coefficient is contracted with the
        geometry in its OWN slot, so a caller wanting B(y, z) passes
        (c_z, dr_y, dm_y) in slot 1 and (c_y, dr_z, dm_z) in slot 2.
        Rec.d2F_dF3 is the caller and already does this."""
        if self._is_zero(Deltam1) and self._is_zero(Deltam2):
            return self.d2curlySc(c, r, m, c1, Deltar1, c2, Deltar2)

        e1y, e1x = self._eff_dirs(Deltar1, Deltam1)
        e2y, e2x = self._eff_dirs(Deltar2, Deltam2)

        negi_fy = self.negi_fy[cp.newaxis, :, cp.newaxis]
        negi_fx = self.negi_fx[cp.newaxis, cp.newaxis, :]

        # --- H terms: second spatial derivatives of c, shifted -------------
        C = self.fft2(self.to_complex(ascontig(c)))
        out = self._shift_spec(C * (negi_fy * negi_fy), r, m)
        out *= e1y
        out *= e2y                                             # H_yy·e1y·e2y

        Hyx = self._shift_spec(C * (negi_fy * negi_fx), r, m)
        t = Hyx * e1y
        t *= e2x
        out += t
        t = Hyx * e1x
        t *= e2y
        out += t
        del Hyx, t

        Hxx = self._shift_spec(C * (negi_fx * negi_fx), r, m)
        Hxx *= e1x
        Hxx *= e2x
        out += Hxx
        del Hxx

        # --- G terms: first derivatives against each slot's own coefficient -
        del C
        for ck, ey, ex in ((c1, e1y, e1x), (c2, e2y, e2x)):
            Ck = self.fft2(self.to_complex(ascontig(ck)))
            Gy, Gx = self._grad_fields(Ck, r, m)
            del Ck
            Gy *= ey
            out += Gy
            del Gy
            Gx *= ex
            out += Gx
            del Gx

        return out
