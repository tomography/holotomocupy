"""Regularization terms used by Rec: 3-D biharmonic (Laplacian) and probe-fit."""

import numpy as np
import cupy as cp
from mpi4py import MPI

from .utils import lap, redot, reprod, timer, make_pinned
from .psf import psf_blur


def biharm(pad_chunk):
    """(∇²)² over the owned slices of a chunk carrying 2 ghost rows per z-side.

    Returns an array shaped like pad_chunk[2:-2]: the inner Laplacians consume
    one ghost row, the outer one the second.
    """
    lap_zm1 = lap(pad_chunk[:-4], pad_chunk[1:-3], pad_chunk[2:-2])
    lap_z   = lap(pad_chunk[1:-3], pad_chunk[2:-2], pad_chunk[3:-1])
    lap_zp1 = lap(pad_chunk[2:-2], pad_chunk[3:-1], pad_chunk[4:])
    return lap(lap_zm1, lap_z, lap_zp1)


class LaplacianTerm:
    """3-D biharmonic regularization: (lam / obj_size) * ||∇²u||² where u = vars['obj'].

    Owns the padded scratch buffers u_pad / e_pad / g_pad of shape
    [local_nzobj+4, nobj, nobj] whose middle [2:-2] slices (exposed as `obj_view` /
    `etas_view` / `grads_view`) alias vars['obj'], etas['obj'] and grads['obj'].
    The 2 ghost rows on each z-side let a chunk compute (∇²)² in a single padded
    pass; g_pad exists so the *gradient* direction can be differentiated too,
    which is what lets hessian3 return all three bilinear forms at once. When
    `lam == 0` the term is inactive and no padding is allocated; the views
    return None so the caller can allocate plain obj-shape buffers instead.

    Halo cost over plain obj-shape buffers: 4 z-slabs per padded array.
    """

    def __init__(self, lam, obj_size, local_nzobj, nobj, cl_mpi, gpu_batch,
                 grad_pad=True):
        self.lam       = lam
        self.obj_size  = obj_size
        self.cl_mpi    = cl_mpi
        self.gpu_batch = gpu_batch
        self.u_pad = self.e_pad = self.g_pad = None
        if lam != 0:
            shape = [local_nzobj + 4, nobj, nobj]
            self.u_pad = make_pinned(shape, dtype='complex64'); self.u_pad[:] = 0
            self.e_pad = make_pinned(shape, dtype='complex64'); self.e_pad[:] = 0
            # grads/etas are reconstruction-only; generation (alloc_mode='gen')
            # never touches them, so skip the buffer there.
            if grad_pad:
                self.g_pad = make_pinned(shape, dtype='complex64'); self.g_pad[:] = 0

    @property
    def obj_view(self):
        """Storage for vars['obj'] (view into u_pad); None when this term is inactive."""
        return None if self.u_pad is None else self.u_pad[2:-2]

    @property
    def etas_view(self):
        """Storage for etas['obj'] (view into e_pad); None when this term is inactive."""
        return None if self.e_pad is None else self.e_pad[2:-2]

    @property
    def grads_view(self):
        """Storage for grads['obj'] (view into g_pad); None when inactive."""
        return None if self.g_pad is None else self.g_pad[2:-2]

    def exchange_ghosts(self, pad):
        """Fill pad[0:2] / pad[-2:] from neighbouring ranks; zero at the global boundary."""
        rank, size = self.cl_mpi.rank, self.cl_mpi.size
        left  = rank - 1 if rank > 0        else MPI.PROC_NULL
        right = rank + 1 if rank < size - 1 else MPI.PROC_NULL
        self.cl_mpi.comm.Sendrecv(
            sendbuf=np.ascontiguousarray(pad[-4:-2]), dest=right,
            recvbuf=pad[0:2], source=left)
        self.cl_mpi.comm.Sendrecv(
            sendbuf=np.ascontiguousarray(pad[2:4]), dest=left,
            recvbuf=pad[-2:], source=right)
        if left  == MPI.PROC_NULL: pad[0:2] = 0
        if right == MPI.PROC_NULL: pad[-2:] = 0

    @timer
    def gradient(self, grad_obj):
        """Add 2*lam/obj_size * (∇²)²u to grad_obj in-place."""
        if self.lam == 0:
            return
        scale = np.float32(2.0 * self.lam / self.obj_size)
        self.exchange_ghosts(self.u_pad)

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_grad(self, g, u_pad_chunk, g_in):
            g[:] = g_in + scale * biharm(u_pad_chunk)

        _biharm_grad(self, grad_obj, self.u_pad, grad_obj)

    @timer
    def hessian(self, dobj1):
        """2*lam/obj_size * Re<dobj1, (∇²)²e> over the LOCAL obj slab.

        Local, like energy_local and PrbfitTerm.hessian: Rec.hessian sums the
        three terms and its caller allreduces once. (This used to allreduce
        internally, which made the regularization term count comm.size times
        in the total.  That was latent only while lam_laplacian was 0 in every
        config, which is no longer true -- the step-6 ladders now run it at
        4e-4/2e-4, so do not reintroduce the internal allreduce.)"""
        if self.lam == 0:
            return 0
        scale = np.float32(2.0 * self.lam / self.obj_size)
        self.exchange_ghosts(self.e_pad)
        acc = cp.zeros(1, dtype='float32')

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_dot(self, acc, e_pad_chunk, d1):
            acc[:] += redot(d1, biharm(e_pad_chunk))

        _biharm_dot(self, acc, self.e_pad, dobj1)
        return float(scale * float(acc[0]))

    @timer
    def hessian3(self, *_):
        """The three regularization bilinear forms {B(g,g), B(g,e), B(e,e)} from
        ONE pass — the Laplacian counterpart of Rec.hessian3.

        Takes no directions: g and e are g_pad[2:-2] and e_pad[2:-2], so the pass
        streams the two padded slabs and reads both the biharmonics and the
        contraction vectors out of them. (Any positional arguments are ignored,
        which keeps the call site uniform with the other hessian3's.)

        Both fields carry ghost rows, so both biharmonics are available in the
        same chunk and the three redots cost only arithmetic on top. Local, like
        hessian(); the caller allreduces.

        Half the traffic of the two-pass route it replaces, and it lets the
        caller derive `bottom` arithmetically instead of re-measuring it after
        the etas update."""
        if self.lam == 0:
            return 0.0, 0.0, 0.0
        scale = np.float32(2.0 * self.lam / self.obj_size)
        # The 3-element accumulator is a non-proper output only because 3 is not
        # the chunked axis length; a degenerate slab would silently make it one.
        assert self.g_pad.shape[0] - 4 != 3, "local_nzobj==3 aliases the accumulator shape"
        self.exchange_ghosts(self.g_pad)
        self.exchange_ghosts(self.e_pad)
        acc = cp.zeros(3, dtype='float32')

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_dot3(self, acc, g_pad_chunk, e_pad_chunk):
            bg = biharm(g_pad_chunk)
            be = biharm(e_pad_chunk)
            g  = g_pad_chunk[2:-2]
            acc[0:1] += redot(g, bg)
            acc[1:2] += redot(g, be)
            acc[2:3] += redot(e_pad_chunk[2:-2], be)

        _biharm_dot3(self, acc, self.g_pad, self.e_pad)
        a = acc.get()
        return float(scale * a[0]), float(scale * a[1]), float(scale * a[2])

    def energy_local(self):
        """Local biharmonic energy (lam/obj_size) * ||∇²u||². No allreduce."""
        if self.lam == 0:
            return np.float32(0)
        scale = np.float32(self.lam / self.obj_size)
        self.exchange_ghosts(self.u_pad)
        acc = cp.zeros(1, dtype='float32')

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_e(self, acc, u_pad_chunk):
            l = lap(u_pad_chunk[1:-3], u_pad_chunk[2:-2], u_pad_chunk[3:-1])
            acc[:] += redot(l, l)

        _biharm_e(self, acc, self.u_pad)
        return scale * float(acc[0])


class PrbfitTerm:
    """Probe-fit regularization, the same misfit shape F0 uses (W = 1 here: the
    flat field has no out-of-grid pixels), with the SAME `model` knob::

        intensity   (lam / prb_size) * || K|D·prb|²      - ref² ||²
        amplitude   (lam / prb_size) * || sqrt(K|D·prb|²) - ref  ||²

    It follows rec_mpi's model rather than being fixed, so the regularizer and
    the data term always have the same residual scale and lam stays a plain
    relative weight.  (The history: this term was amplitude-based, F0 moved to
    intensity, and lam had to grow ~4x -- e.g. 3.1e-3 -> 1.2e-2 -- because the
    two shapes had drifted apart.  Tying them together is what stops that
    happening again: BECAUSE this term follows the knob, lam_prbfit carries
    across a model switch unchanged -- both F0 and this residual rescale by the
    same ~4x, so their ratio is fixed.  lam_laplacian is the one that must be
    divided by ~4 for amplitude, since ||∇²u||² does not rescale with the
    model.  rec_mpi warns about exactly this asymmetry at startup.)

    `ref` is stored as an AMPLITUDE either way (see below), so the amplitude
    model compares against it directly and the intensity model squares it.
    BLURRED in both: the measured flat came off the same detector through the
    same partially coherent beam as the data.  See the derivation in gradient().

    K is the same single Gaussian on detector intensity F0 uses -- the measured
    flat came off the same detector through the same partially coherent beam as
    the data, so its model is K``|D·prb|``².  Blurring one side of the misfit and
    not the other would leave the probe absorbing the PSF.  K is self-adjoint
    (psf_blur is a symmetric separable circulant), which is what lets the two
    derivatives below move it across the inner product.  psf_w=None is the
    identity and restores the plain intensity penalty.

    The intensity model divides by nothing; the amplitude model divides by
    a = sqrt(K``|D·prb|``²) and by a³, which blow up wherever the propagated probe
    has a zero, so both denominators are floored at _AMP_FLOOR exactly as in
    rec_mpi's F0.  The energy itself is left exact; only the derivatives are.

    Owns `ref` (shape [ndist, nz, n] float32) — the per-distance reference probe
    AMPLITUDE that the regularizer fits against; the square is taken here, at use.
    It is deliberately still an amplitude: every producer makes one
    (`reader.read_ref` takes sqrt of the measured flat, `gen_sqrt_ref` and
    `disp_study.common.gen_ref` return ``|D·prb|``), and squaring in one place beats
    changing all of them.  Allocated unconditionally so external code can seed it
    regardless of whether the term is active."""

    _AMP_FLOOR = np.float32(1e-3)   # same floor as Rec._AMP_FLOOR

    def __init__(self, lam, prb_size, ndist, nz, n, cl_prop, psf_w=None,
                 model='intensity'):
        self.lam      = lam
        self.prb_size = prb_size
        self.ndist    = ndist
        self.cl_prop  = cl_prop
        self.psf_w    = psf_w
        self.model    = model
        self.amp      = (model == 'amplitude')
        self.ref      = cp.empty([ndist, nz, n], dtype='float32')

    def _blur(self, x):
        """K x.  Identity when no PSF is configured; K^T = K, so one call serves
        both directions, exactly as in Rec._blur."""
        return x if self.psf_w is None else psf_blur(x, self.psf_w)

    def _J(self, u):
        """J = K|u|², the blurred model intensity both models are built on."""
        return self._blur(reprod(u, u))

    def _p(self, u, j, J=None):
        """p = K[g'(J)/2], the pointwise weight both derivatives share --
        (J - ref²) for intensity, (a - ref)/(2a) for amplitude.

        The same quantity rec_mpi.dF0 builds for the data term (with W = 1)."""
        if J is None:
            J = self._J(u)
        if self.amp:
            a = cp.maximum(cp.sqrt(J), self._AMP_FLOOR)
            return self._blur(0.5 * (1.0 - self.ref[j:j+1] / a))
        r = self.ref[j:j+1] * self.ref[j:j+1]
        return self._blur(J - r)

    def _c(self, J, j):
        """g''(J)/2, the pointwise curvature weight: 1 for intensity,
        ref/(4a³) for amplitude.  Returns None when it is identically 1, so the
        intensity path multiplies by nothing."""
        if not self.amp:
            return None
        a = cp.maximum(cp.sqrt(J), self._AMP_FLOOR)
        return 0.25 * self.ref[j:j+1] / (a * a * a)

    @timer
    def gradient(self, grad_prb, prb, rho_sq_prb):
        """Add (lam / prb_size) * D^T(4 p · D·prb) * rho_sq_prb to grad_prb in-place,
        with p = K[K``|D·prb|``² - ref²].

        With u = D·prb, I = |u|², J = K I and f = g(J):
            df = g'(J) · K(dI)     =  K(g'(J)) · dI            (K self-adjoint)
               = 2p · 2Re<u,v>     =  Re<4p·u, v>              (p = K[g'/2])
        so grad = 4 p u for BOTH models, the real gradient (df/da, df/db) packed
        as a complex, and the same shape as rec_mpi.dF0 = 4/N Σ p Re(conj(x) y).
        Only p differs between them -- see _p.

        grad_prb may be pinned numpy (vars/grads/etas['prb'] are all pinned); the per-j
        contribution is .get()'d to host before accumulating."""
        if self.lam == 0:
            return
        for j in range(self.ndist):
            tmp = self.cl_prop.D(prb[j:j+1], j)
            td  = self.lam / self.prb_size * self.cl_prop.DT(
                4 * self._p(tmp, j) * tmp, j)
            contrib = (td * rho_sq_prb)
            if isinstance(grad_prb, cp.ndarray):
                grad_prb[j:j+1] += contrib
            else:
                grad_prb[j:j+1] += cp.asnumpy(contrib)

    @timer
    def hessian(self, prb, dprb1, dprb2):
        """Probe-fit hessian: lam/prb_size * Σ_j Σ [8 c K<u,v> K<u,w> + 4 p Re<v,w>],
        with u = D·prb, J = K|u|², p = K[g'(J)/2], c = g''(J)/2, v = D·dprb1,
        w = D·dprb2 and K<u,v> shorthand for K(Re<u,v>).

        Second variation of f = g(J):  d²f = g''(dJ)² + g' d²J, with
        dJ = K(dI) = K(2Re<u,v>) and d²J = K(d²I) = K(2Re<v,w>).  Moving K off
        the second term (K^T = K) turns g'(J)·K(Re<v,w>) into 2p·Re<v,w>:

            B(v,w) = 8 Σ c K(Re<u,v>) K(Re<u,w>) + 4 Σ p Re<v,w>

        which is rec_mpi.d2F_dF0's two terms with W = 1 -- its first term is
        written 2 K[c K[Re<x,z>]] Re<x,y>, the same thing with K moved the other
        way.  c is identically 1 for the intensity model (_c returns None and
        the multiply is skipped) and ref/(4a³) for the amplitude one."""
        if self.lam == 0:
            return 0
        out = 0
        for j in range(self.ndist):
            Dprb   = self.cl_prop.D(prb[j:j+1], j)
            Ddprb1 = self.cl_prop.D(dprb1[j:j+1], j)
            Ddprb2 = self.cl_prop.D(dprb2[j:j+1], j)
            J  = self._J(Dprb)
            p  = self._p(Dprb, j, J)
            c  = self._c(J, j)
            k1 = self._blur(reprod(Dprb, Ddprb1))
            k2 = self._blur(reprod(Dprb, Ddprb2))
            v1 = cp.sum(k1 * k2 if c is None else c * k1 * k2)
            v2 = cp.sum(p * reprod(Ddprb1, Ddprb2))
            out += 8 * v1 + 4 * v2
        out = self.lam * out / self.prb_size
        return out.get()

    @timer
    def hessian3(self, prb, dg, de):
        """The three probe-fit bilinear forms {B(g,g), B(g,e), B(e,e)}.

        Same contraction as hessian(), but the direction-independent u, p and c
        and the two direction propagations are each computed once per distance
        instead of once per pair — 3 D() calls per j instead of 9, and 2 blurs
        per direction instead of 2 per pair."""
        if self.lam == 0:
            return 0, 0, 0
        ogg = oge = oee = 0
        for j in range(self.ndist):
            Dprb = self.cl_prop.D(prb[j:j+1], j)
            Dg   = self.cl_prop.D(dg[j:j+1], j)
            De   = self.cl_prop.D(de[j:j+1], j)
            J  = self._J(Dprb)
            p  = self._p(Dprb, j, J)
            c  = self._c(J, j)
            kg = self._blur(reprod(Dprb, Dg))
            ke = self._blur(reprod(Dprb, De))
            # fold c in once per direction rather than once per pair
            cg = kg if c is None else c * kg
            ce = ke if c is None else c * ke
            ogg += 8 * cp.sum(cg * kg) + 4 * cp.sum(p * reprod(Dg, Dg))
            oge += 8 * cp.sum(cg * ke) + 4 * cp.sum(p * reprod(Dg, De))
            oee += 8 * cp.sum(ce * ke) + 4 * cp.sum(p * reprod(De, De))
        s = self.lam / self.prb_size
        return (s * ogg).get(), (s * oge).get(), (s * oee).get()

    def energy_local(self, prb):
        """Local probe-fit energy (lam / prb_size) * Σ_j ||·||², the residual
        being K``|D·prb_j|``² - ref_j² (intensity) or sqrt(K``|D·prb_j|``²) - ref_j
        (amplitude).  Exact -- no floor: nothing divides here."""
        if self.lam == 0:
            return 0
        out = 0
        for j in range(self.ndist):
            J = self._J(self.cl_prop.D(prb[j:j+1], j))
            if self.amp:
                res = cp.sqrt(J) - self.ref[j:j+1]
            else:
                res = J - self.ref[j:j+1] * self.ref[j:j+1]
            out += self.lam / self.prb_size * cp.sum(res * res)
        return out

    def gen_sqrt_ref(self, prb, out):
        """Populate `out` with the synthetic reference,
        ``out[j] = sqrt(K``|D·prb_j|``²)``.

        BLURRED, because it stands in for a measured flat and a measured flat came
        through the same PSF as the data -- an unblurred synthetic ref would leave
        F1 = ||K``|D·prb|``² - ref²||² at O(blur²) rather than ~0 at the true probe, i.e.
        it would pull the probe by exactly the amount of the PSF the model already
        accounts for.  With psf_w=None this is ``|D·prb|`` as before.

        Still an AMPLITUDE, and still named for the sqrt: `ref` stays an amplitude and
        PrbfitTerm squares it at use, so every existing seeding call site is unchanged.
        The sqrt-then-square costs an ulp, so F1 at the truth is ~1e-16 relative, not
        bit-zero.
        Used by tests/perf scripts to seed self.ref so the regularizer has something to fit."""
        for j in range(self.ndist):
            u = self.cl_prop.D(prb[j:j+1], j)
            out[j] = cp.sqrt(self._blur(reprod(u, u)))[0]
