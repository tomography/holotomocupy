import numpy as np
import cupy as cp
import os
import time
import logging
import psutil
from functools import wraps
from .logger_config import logger

# matplotlib and tifffile are imported lazily inside the functions that need
# them: they are only reached from interactive/diagnostic paths, and pulling
# them in at import time costs every MPI rank ~0.5 s and ~100 MB of RSS.

# Cached once — avoids a system call on every timed invocation
_process = psutil.Process()


def make_pinned(shape, dtype):
    n      = int(np.prod(shape))
    nbytes = n * np.dtype(dtype).itemsize
    logger.debug(f'Allocate {shape} {dtype}: {nbytes / 1024**3:.3f} GB')
    buf = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(buf, dtype=dtype, count=n).reshape(shape, copy=False)


def mshow(a, show=False, figsize=(6, 6), **args):
    if show:
        import matplotlib.pyplot as plt
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        im = axs.imshow(a, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_complex(a, show=False, figsize=(14, 6), **args):
    if show:
        import matplotlib.pyplot as plt
        from matplotlib_scalebar.scalebar import ScaleBar
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        im = axs[0].imshow(a.real, cmap="gray", **args)
        scalebar = ScaleBar(0.015, "um", length_fraction=0.25,
                            font_properties={"family": "serif"},
                            location="lower right")
        axs[0].add_artist(scalebar)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(a.imag, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_polar(a, show=False, figsize=(14, 6), **args):
    if show:
        import matplotlib.pyplot as plt
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        im = axs[0].imshow(np.abs(a), cmap="gray", **args)
        axs[0].set_title("abs")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(np.angle(a), cmap="gray", **args)
        axs[1].set_title("phase")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_pos(pos, show=False, figsize=(10, 4), **args):
    if show:
        import matplotlib.pyplot as plt
        if isinstance(pos, cp.ndarray):
            pos = pos.get()
        _, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].plot(pos[..., 1], ".")
        ax[0].set_title("x")
        ax[1].plot(pos[..., 0], ".")
        ax[1].set_title("y")
        ax[0].grid()
        ax[1].grid()
        plt.show()


def mshow_approx(t, err_real, err_approx, show=False):
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4))
        plt.plot(t, err_real,   "o-", label="real")
        plt.plot(t, err_approx, "x-", label="approx")
        plt.legend()
        plt.grid()
        plt.show()


def reprod(a, b):
    return cp.real(a) * cp.real(b) + cp.imag(a) * cp.imag(b)


def redot(a, b, axis=None):
    if axis is None:
        return cp.vdot(a.view('float32'), b.view('float32'))
    return cp.sum(reprod(a, b), axis=axis)


def lap(a, b, c):
    """5-stencil Laplacian on the centre slice b given its z-neighbours a, c.
    Output shape matches b: ∇²b ≈ a + c + roll(b,±1,axis=1) + roll(b,±1,axis=2) - 6·b."""
    return (a + c
            + cp.roll(b, -1, axis=1) + cp.roll(b,  1, axis=1)
            + cp.roll(b, -1, axis=2) + cp.roll(b,  1, axis=2)
            - 6 * b)


def write_tiff(a, name, **args):
    import tifffile
    if isinstance(a, cp.ndarray):
        a = a.get()
    dirname = os.path.dirname(name)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tifffile.imwrite(name + '.tiff', a)


def read_tiff(name):
    import tifffile
    return tifffile.imread(name)[:]



def timer(func):
    """Log wall time and memory for a call, but only when DEBUG is enabled.

    @timer sits on hot paths (redot_batch, linear_batch, hessian_cascade, ...)
    that run many times per iteration, and the RSS query, memGetInfo() call and
    f-string cost real time. The isEnabledFor() guard makes the decorator very
    nearly free when DEBUG logging is off, which is the normal case.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not logger.isEnabledFor(logging.DEBUG):
            return func(*args, **kwargs)
        start  = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        mem = _process.memory_info().rss / 1024**3
        free, total = cp.cuda.runtime.memGetInfo()
        gpu_mem = (total - free) / 1024**3
        logger.debug(f"{func.__name__}: {elapsed:.4f} sec, process memory {mem:.2f} GB, GPU memory {gpu_mem:.2f} GB")
        return result
    return wrapper


def empty_like(x):
    if isinstance(x, cp.ndarray):
        return cp.empty_like(x)
    return np.empty_like(x)

