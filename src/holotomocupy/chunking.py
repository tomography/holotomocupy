import cupy as cp
import numpy as np
from .utils import redot, timer

def _axpby(out, x, y, a, b):
    """out = a*x + b*y, in place and with as few temporaries as the scalars allow.

    Written as `out[:] = a*x + b*y` this costs three chunk-sized allocations --
    a*x, b*y and their sum -- plus the copy into out, i.e. four full passes over
    a chunk and three buffers the chunking arena never accounted for. On the
    object-shape call sites a chunk is nchunk*nobj*nobj complex64 (hundreds of
    MB), so those temporaries are close to a gigabyte of pool.

    Every call site in the solvers has a in {0, 1} or b in {1, -1, 0}; the
    branches below turn those into pure in-place ufuncs with no temporary at
    all, and only the genuinely general `out += b*y` keeps one.
    """
    if a == 1:
        if out is not x:
            out[:] = x
    elif a == 0:
        out[:] = 0
    else:
        cp.multiply(x, a, out=out)

    if b == 1:
        out += y
    elif b == -1:
        out -= y
    elif b != 0:
        out += b * y
    return out


class Chunking:
    def __init__(self, nbytes, chunk):
        self.gpu_mem = cp.cuda.alloc(nbytes)
        self.nbytes  = nbytes
        self.stream  = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
        self.chunk   = chunk

    def gpu_batch(self, axis_out=0, axis_inp=0, nout=1, inp_pad=0):
        """
        Single-GPU chunked processing of functions with syntax
        f(out1_proper, ..., out1_nonproper, ...,
          inp1_proper, ..., inp1_nonproper, ..., inp1, inp2, ...)

        where
        out*_proper  are numpy or cupy arrays whose shape[axis_out] equals the
                     chunking dimension size. Numpy arrays are transferred D2H
                     per chunk; CuPy arrays are written in-place on the GPU.
        inp*_proper  are numpy or cupy arrays whose shape[axis_inp] equals the
                     chunking dimension size. Numpy arrays are transferred H2D
                     per chunk; CuPy arrays are sliced directly on the GPU.
        out*_nonproper are CuPy arrays of non-chunking shape (filled in-place,
                     no CPU transfer).
        inp*_nonproper are numpy/CuPy arrays of non-chunking shape (replicated
                     to the GPU once).

        inp_pad > 0: the FIRST argument after the nout outputs is a "padded"
                     proper input whose shape[axis_inp] == size + inp_pad.
                     Each chunk transfers (chunk + inp_pad) rows so the kernel
                     receives the full padded window and can slice freely.
                     size is derived as inp[0].shape[axis_inp] - inp_pad.
                     Any FURTHER proper input whose shape[axis_inp] is also
                     size + inp_pad gets the same halo treatment, so a kernel
                     needing two haloed fields (eg a bilinear stencil form
                     against two directions) can take both in one pass.
        """

        def decorator(func):
            def inner(*args):
                # if no numpy arrays present, run the function directly on GPU
                if not any(isinstance(a, np.ndarray) for a in args):
                    func(*args)
                    return

                cl  = args[0]
                out = args[1 : 1 + nout]
                inp = args[1 + nout :]

                # size: actual chunking length (without padding)
                size = inp[0].shape[axis_inp] - inp_pad

                proper_inp,   nonproper_inp   = 0, 0
                proper_out,   nonproper_out   = 0, 0

                for k in range(len(out)):
                    if ((isinstance(out[k], np.ndarray) or isinstance(out[k], cp.ndarray))
                            and out[k].ndim > axis_out
                            and out[k].shape[axis_out] == size):
                        proper_out += 1
                    elif isinstance(out[k], (np.ndarray, cp.ndarray)):
                        nonproper_out += 1

                # inp[0] when inp_pad > 0: always the padded proper input.
                # inp_pads[j] is the halo of proper input j -- inp_pad for a
                # haloed one, 0 for an ordinary one. Classified by shape, the
                # same way proper/non-proper already is.
                inp_pads = [inp_pad] if inp_pad > 0 else []
                for k in range(len(inp_pads), len(inp)):
                    if ((isinstance(inp[k], np.ndarray) or isinstance(inp[k], cp.ndarray))
                            and inp[k].ndim > axis_inp
                            and inp[k].shape[axis_inp] in (size, size + inp_pad)):
                        inp_pads.append(inp[k].shape[axis_inp] - size)
                    elif isinstance(inp[k], np.ndarray) or isinstance(inp[k], cp.ndarray):
                        nonproper_inp += 1
                proper_inp = len(inp_pads)

                # Numpy non-proper outputs: auto-create a cupy scratch (uploaded from
                # the current CPU value so read-modify-write patterns work), swap into
                # out[], then D2H back after run() returns. Device-sync wraps the H2D
                # and the post-run D2H because chunking uses non-blocking streams that
                # don't implicitly sync with the per-thread default stream.
                np_out_refs = []   # list of (numpy_ref, cupy_scratch) pairs
                out = list(out)
                for kk in range(proper_out, proper_out + nonproper_out):
                    if isinstance(out[kk], np.ndarray):
                        numpy_ref = out[kk]
                        cupy_scratch = cp.empty(numpy_ref.shape, dtype=numpy_ref.dtype)
                        cupy_scratch.set(numpy_ref)
                        out[kk] = cupy_scratch
                        np_out_refs.append((numpy_ref, cupy_scratch))
                if np_out_refs:
                    cp.cuda.Device().synchronize()   # ensure H2D done before chunking reads

                # build argument lists for the single GPU
                ginp = [x for x in inp[:proper_inp]]
                ginp.extend(inp[proper_inp:])

                gout = [x for x in out[:proper_out]]
                gout.extend(out[proper_out:])

                if np.prod(gout[0].shape) == 0:
                    return

                self.run(cl, gout, ginp,
                         proper_inp, nonproper_inp,
                         proper_out, nonproper_out,
                         axis_out, axis_inp, func, inp_pads, size)

                # D2H any numpy non-proper outputs back to their original pinned buffers.
                if np_out_refs:
                    cp.cuda.Device().synchronize()   # ensure compute streams visible to default
                    for numpy_ref, cupy_scratch in np_out_refs:
                        if numpy_ref.flags['C_CONTIGUOUS']:
                            cupy_scratch.get(out=numpy_ref)
                        else:
                            numpy_ref[...] = cupy_scratch.get()

            return inner

        return decorator

    def run(self, cl, out, inp, proper_inp, nonproper_inp, proper_out, nonproper_out, axis_out, axis_inp, func, inp_pads=None, size=None):
        """Run by chunks with overlapped H2D / compute / D2H on three streams.

        inp_pads[j] is the halo width of proper input j (0 for most; inp_pad
        for the padded ones). size is the chunking length without padding.
        """
        if inp_pads is None:
            inp_pads = [0] * proper_inp
        if size is None:
            size = inp[0].shape[axis_inp]

        gpu_mem = self.gpu_mem
        stream  = self.stream

        # The chunking streams are non-blocking, so they have NO ordering relation
        # with the caller's stream -- which breaks CuPy's pool invariant that a
        # block freed on stream A is safe to reissue on A. A caller leaving an
        # unsynchronized kernel pending (`out += term.energy_local(prb)` drops the
        # device scalar it still has to read) gets that block recycled into a
        # chunked accumulator and overwritten mid-read. One event restores the
        # pool's assumption. See tests/data_mask/probe_lap.py.
        _ev = cp.cuda.Event(disable_timing=True)
        _ev.record(cp.cuda.get_current_stream())
        for s in stream:
            s.wait_event(_ev)

        nchunk = int(np.ceil(size / self.chunk))

        # pre-allocate double-buffered GPU arrays
        out_gpu, offset = self.alloc_double_buffers(out[:proper_out], axis_out, gpu_mem, 0, self.chunk)

        # each proper input gets chunk + its own halo rows
        inp_gpu = [[], []]
        for j in range(proper_inp):
            bufs, offset = self.alloc_double_buffers(
                inp[j:j + 1], axis_inp, gpu_mem, offset, self.chunk + inp_pads[j])
            inp_gpu[0].append(bufs[0][0])
            inp_gpu[1].append(bufs[1][0])

        # move non-proper numpy inputs to GPU once
        for k in range(proper_inp, proper_inp + nonproper_inp):
            inp[k] = cp.asarray(inp[k])

        def p2g(buf_id, k):
            st  = k * self.chunk
            end = min(size, (k + 1) * self.chunk)
            cur_stream = cp.cuda.get_current_stream()
            for j in range(proper_inp):
                extra = inp_pads[j]
                ndim = inp[j].ndim
                src = self.mk_slices(axis_inp, slice(st, end + extra), ndim)
                dst = self.mk_slices(axis_inp, slice(0, end - st + extra), ndim)
                if axis_inp == 1:
                    c_src = inp[j][src]
                    c_dst = inp_gpu[buf_id][j][dst]
                    rows      = c_src.shape[0]
                    row_bytes = c_src[0].nbytes
                    cp.cuda.runtime.memcpy2DAsync(
                        c_dst.data.ptr,    c_dst.strides[0],
                        c_src.ctypes.data, c_src.strides[0],
                        row_bytes, rows,
                        cp.cuda.runtime.memcpyHostToDevice,
                        cur_stream.ptr,
                    )
                else:
                    if isinstance(inp[j], cp.ndarray):
                        cp.copyto(inp_gpu[buf_id][j][dst], inp[j][src])
                    else:
                        inp_gpu[buf_id][j][dst].set(inp[j][src])

        def g2p(buf_id, k):
            st  = k * self.chunk
            end = min(size, (k + 1) * self.chunk)
            cur_stream = cp.cuda.get_current_stream()
            for j in range(proper_out):
                ndim = out[j].ndim
                src = self.mk_slices(axis_out, slice(0, end - st), ndim)
                dst = self.mk_slices(axis_out, slice(st, end), ndim)
                if axis_out == 1:
                    c_src = out_gpu[buf_id][j][src]
                    c_dst = out[j][dst]
                    rows      = c_src.shape[0]
                    row_bytes = c_src[0].nbytes
                    cp.cuda.runtime.memcpy2DAsync(
                        c_dst.ctypes.data, c_dst.strides[0],
                        c_src.data.ptr,    c_src.strides[0],
                        row_bytes, rows,
                        cp.cuda.runtime.memcpyDeviceToHost,
                        cur_stream.ptr,
                    )
                else:
                    if isinstance(out[j], cp.ndarray):
                        cp.copyto(out[j][dst], out_gpu[buf_id][j][src])
                    else:
                        # cupy's .get(out=...) needs a C-contiguous destination; if the
                        # numpy slice is strided (e.g. data[:, k] view), fall back to a
                        # contiguous intermediate then numpy-assign.
                        host_dst = out[j][dst]
                        if host_dst.flags['C_CONTIGUOUS']:
                            out_gpu[buf_id][j][src].get(out=host_dst, blocking=False)
                        else:
                            host_dst[...] = out_gpu[buf_id][j][src].get()

        def p(buf_id, k):
            st  = k * self.chunk
            end = min(size, (k + 1) * self.chunk)
            n   = end - st
            # Slice each proper input; padded ones get n + their halo rows
            inp_gpu_c = []
            for j in range(proper_inp):
                slc = self.mk_slices(axis_inp, slice(0, n + inp_pads[j]), inp_gpu[buf_id][j].ndim)
                inp_gpu_c.append(inp_gpu[buf_id][j][slc])
            out_gpu_c = self.slice_bufs(out_gpu[buf_id], axis_out, n)
            func(
                cl,
                *out_gpu_c,
                *out[proper_out:],
                *inp_gpu_c,
                *inp[proper_inp : proper_inp + nonproper_inp],
                *inp[proper_inp + nonproper_inp :],
            )

        # All three stages of chunk j run on stream[j % 3], so h2d -> compute
        # -> d2h is already stream-ordered. Only three hazards remain, and all
        # three are events rather than a host synchronize() per step -- that
        # barrier drained the device every iteration and cost 1.1-1.4x
        # (tests/performance/bench_chunking.py).
        ev_p = [cp.cuda.Event(disable_timing=True) for _ in range(3)]
        ev_g = [cp.cuda.Event(disable_timing=True) for _ in range(3)]
        rec_p, rec_g = [False] * 3, [False] * 3

        for k in range(nchunk + 2):
            if k < nchunk:
                s = stream[k % 3]
                with s:
                    # WAR: this input buffer was last read by compute(k-2)
                    if k >= 2 and rec_p[(k - 2) % 3]:
                        s.wait_event(ev_p[(k - 2) % 3])
                    p2g(k % 2, k)
            if 0 < k < nchunk + 1:
                j = k - 1
                s = stream[j % 3]
                with s:
                    # WAR: this output buffer was last read by d2h(j-2)
                    if j >= 2 and rec_g[(j - 2) % 3]:
                        s.wait_event(ev_g[(j - 2) % 3])
                    # non-proper outputs are shared accumulators (res += ...),
                    # so computes must not run concurrently
                    if j >= 1 and rec_p[(j - 1) % 3]:
                        s.wait_event(ev_p[(j - 1) % 3])
                    p(j % 2, j)
                    ev_p[j % 3].record(s)
                    rec_p[j % 3] = True
            if 1 < k:
                j = k - 2
                s = stream[j % 3]
                with s:
                    g2p(j % 2, j)
                    ev_g[j % 3].record(s)
                    rec_g[j % 3] = True
        # one host sync, so numpy outputs are readable and the caller's stream
        # is ordered after everything above
        for s in stream:
            s.synchronize()

    def alloc_double_buffers(self, arrs, axis, gpu_mem, offset, chunk):
        """Allocate double-buffered GPU arrays from the pre-allocated pool.

        Each buffer's offset is rounded up to ALIGN bytes — cuFFT / kernels can
        return CUFFT_INVALID_VALUE / misaligned-access on cupy views whose data
        pointer isn't aligned to the element size or vector load width. Small 1-D
        buffers (eg eff_demag [chunk] float32 at chunk=1 = 4 bytes) used to push
        the next slot 4 bytes past a 16-byte boundary -> complex64 cuFFT failed.
        """
        ALIGN = 128
        gpu = [[], []]
        for j in (0, 1):
            for a in arrs:
                shape0 = list(a.shape)
                shape0[axis] = chunk
                shape0 = tuple(shape0)
                n       = int(np.prod(shape0))
                nbytes  = n * np.dtype(a.dtype).itemsize
                offset  = (offset + ALIGN - 1) & ~(ALIGN - 1)
                try:
                    gpu[j].append(cp.ndarray(shape0, dtype=a.dtype, memptr=gpu_mem + offset))
                except Exception as e:
                    raise RuntimeError("Failed to allocate GPU buffers") from e
                offset += nbytes
        # cp.ndarray over a memptr does no bounds checking, so a pool that is
        # too small for the call site hands back views pointing past the
        # allocation and the failure only shows up later as an opaque
        # cudaErrorInvalidValue inside p2g/g2p.  Say what actually went wrong.
        if offset > self.nbytes:
            raise RuntimeError(
                f"chunking pool too small: {offset} bytes needed for "
                f"double-buffered chunk={chunk} views, pool is {self.nbytes}. "
                f"Rec._pool_bytes_for is missing this call site, or nchunk was "
                f"raised after the pool was allocated.")
        return gpu, offset

    ####################### Slicing #########################
    def slice_bufs(self, bufs, axis, n):
        result = []
        for b in bufs:
            slc = [slice(None)] * b.ndim
            slc[axis] = slice(0, n)
            result.append(b[tuple(slc)])
        return result

    def mk_slices(self, axis, sl, ndim=3):
        res = [slice(None)] * ndim
        res[axis] = sl
        return tuple(res)

    ####################### Simple batched functions #########################
    @timer
    def redot_batch(self, x, y, nout=1):
        """res = Re<x, y>"""
        if isinstance(x, cp.ndarray):
            return redot(x, y).get()
        # 0-d, not shape (1,): gpu_batch calls an output chunked when
        # shape[axis_out] == size, so a (1,) accumulator is misread as chunked
        # whenever size == 1 -- and chunked outputs sit in *uninitialised* arena
        # scratch. `ndim > axis_out` is false for a 0-d array, so it never aliases.
        res = cp.zeros((), dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0)
        def _redot(self, res, x, y):
            res[...] += redot(x, y)

        _redot(self, res, x, y)
        return res.get()[()]

    @timer
    def linear_batch(self, x, y, a, b, out=None):
        """w = ax + by"""
        if out is None:
            out = x
        if isinstance(x, cp.ndarray):
            _axpby(out, x, y, a, b)
            return

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _linear(self, out, x, y, a, b):
            _axpby(out, x, y, a, b)

        _linear(self, out, x, y, a, b)

    @timer
    def linear_redot_batch(self, x, y, a, b):
        """x = ax + by, returns Re<y, x_new> in one pass"""
        if isinstance(x, cp.ndarray):
            _axpby(x, x, y, a, b)
            return redot(y, x).get()
        res = cp.zeros((), dtype="float32")   # 0-d — see redot_batch

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=2)
        def _linear_redot(self, out, res, x, y, a, b):
            _axpby(out, x, y, a, b)
            res[...] += redot(y, out)

        _linear_redot(self, x, res, x, y, a, b)
        return res.get()[()]

    @timer
    def mulc_batch(self, out, x, a):
        """out = ax"""
        if isinstance(x, cp.ndarray):
            cp.multiply(x, a, out=out)
            return

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _mulc(self, out, x, a):
            cp.multiply(x, a, out=out)

        _mulc(self, out, x, a)
