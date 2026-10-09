"""
conv2d_cufftdx.py
Thin Python wrapper around the cuFFTDx 2D convolution shared library.

Pipeline (per batch): FFT_Y -> (FFT_X -> H*x -> IFFT_X) -> IFFT_Y

NOTE: cuFFTDx IFFT is unnormalized (like FFTW).
      conv2d_run output = NX * NY * cp.fft.ifft2(cp.fft.fft2(x) * H)
      Callers must divide by NX*NY to match cuPy convention.

Arrays:
    x  : [batches, NX, NY] complex64  (contiguous)
    H  : [NX, NY]          complex64  (contiguous)
    y  : [batches, NX, NY] complex64  (contiguous, may alias x)

Environment variables (all optional):
    MATHDX_ROOT   path to the mathDX installation
                  (default: /local/vnikitin/nvidia-mathdx-25.12.1-cuda13/nvidia/mathdx/25.12)
    NVCC          path to nvcc  (default: nvcc, i.e. whatever is on PATH)
    CUFFTDX_SM    CUDA SM version to compile for  (default: 80)
    CUFFTDX_SO_DIR directory to cache compiled .so files  (default: next to this file)
"""

import ctypes
import functools
import os
import pathlib
import subprocess
import warnings

import cupy as cp

from .logger_config import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = pathlib.Path(__file__).parent / "cuda"
_SRC     = _HERE / "conv2d.cu"

_MATHDX      = pathlib.Path(os.environ.get(
    "MATHDX_ROOT",
    "/local/vnikitin/nvidia-mathdx-25.12.1-cuda13/nvidia/mathdx/25.12",
))
_CUTLASS     = _MATHDX / "external/cutlass/include"
_CUFFTDX_EX  = _MATHDX / "example/cufftdx"   # common/ and 07_convolution_3d/ live here
_NVCC    = os.environ.get("NVCC", "nvcc")
_SO_DIR  = pathlib.Path(os.environ.get("CUFFTDX_SO_DIR", str(_HERE)))


@functools.lru_cache(maxsize=None)
def _sm() -> str:
    """SM version to compile for. Queried lazily so that importing this module
    never creates a CUDA context (which would land on the wrong device when
    ranks bind their GPU after import)."""
    return os.environ.get("CUFFTDX_SM") or cp.cuda.Device().compute_capability


@functools.lru_cache(maxsize=None)
def _max_smem() -> int:
    return cp.cuda.Device().attributes['MaxSharedMemoryPerBlockOptin']

# ---------------------------------------------------------------------------
# EPT / FPB tuning table
# ---------------------------------------------------------------------------
_EPT_FPB: dict = {
    512:  (8,  8),
    1024: (16, 8),
    2048: (16, 8),
    4096: (32, 4),
}
_WARP     = 32


def _ept_fpb(n: int):
    # Validate table entry against the current GPU's shared-memory limit before using it.
    if n in _EPT_FPB:
        ept, fpb = _EPT_FPB[n]
        if n * fpb * 8 <= _max_smem():
            return ept, fpb
    fpb = next(f for f in (8, 4, 2, 1) if n * f * 8 <= _max_smem())
    for max_block in (512, 1024):
        for ept in (1, 2, 4, 8, 16, 32, 64, 128):
            if n % ept != 0:
                continue
            threads = n // ept
            if threads % _WARP != 0:
                continue
            if threads * fpb <= max_block:
                return ept, fpb
    raise RuntimeError(f"No valid (EPT, FPB) for size {n}")


# ---------------------------------------------------------------------------
# JIT compile + load
# ---------------------------------------------------------------------------
_lib_cache: dict = {}


def _compile(nx, ny, ept_x, fpb_x, ept_y, fpb_y) -> ctypes.CDLL:
    sm = _sm()
    so = _SO_DIR / f"libconv2d_sm{sm}_{nx}x{ny}_xe{ept_x}f{fpb_x}_ye{ept_y}f{fpb_y}.so"
    if not so.exists() or so.stat().st_mtime < _SRC.stat().st_mtime:
        logger.info(f"cuFFTDx JIT-compiling {nx}x{ny} EPT_X={ept_x} FPB_X={fpb_x} "
                    f"EPT_Y={ept_y} FPB_Y={fpb_y} ...")
        r = subprocess.run([
            _NVCC, "-O3", "--std=c++17", f"-arch=sm_{sm}",
            "-Xcompiler", "-fPIC", "-shared",
            f"-I{_MATHDX}/include", f"-I{_CUTLASS}",
            f"-I{_CUFFTDX_EX}", f"-I{_CUFFTDX_EX}/07_convolution_3d",
            f"-I{_HERE}",
            f"-DNX={nx}", f"-DNY={ny}",
            f"-DEPT_X={ept_x}", f"-DFPB_X={fpb_x}",
            f"-DEPT_Y={ept_y}", f"-DFPB_Y={fpb_y}",
            str(_SRC), "-o", str(so),
        ], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"cuFFTDx build failed for {nx}x{ny}:\n{r.stderr}")
        logger.info("cuFFTDx build done.")
    lib = ctypes.CDLL(str(so))
    lib.conv2d_create.restype   = ctypes.c_void_p
    lib.conv2d_create.argtypes  = [ctypes.c_void_p]
    lib.conv2d_run.restype      = None
    lib.conv2d_run.argtypes     = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.conv2d_destroy.restype  = None
    lib.conv2d_destroy.argtypes = [ctypes.c_void_p]
    lib.conv2d_block_dim.restype  = None
    lib.conv2d_block_dim.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.conv2d_smem_size.restype  = None
    lib.conv2d_smem_size.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    return lib


def _get_lib(nx: int, ny: int) -> ctypes.CDLL:
    key = (nx, ny)
    if key in _lib_cache:
        return _lib_cache[key]
    ept_x, fpb_x = _ept_fpb(nx)
    ept_y, fpb_y = _ept_fpb(ny)
    lib = _compile(nx, ny, ept_x, fpb_x, ept_y, fpb_y)

    # Validate block size
    bx, by = ctypes.c_int(), ctypes.c_int()
    lib.conv2d_block_dim(ctypes.byref(bx), ctypes.byref(by))
    if bx.value * max(fpb_x, fpb_y) > 1024:
        fpb_x = fpb_y = 1024 // bx.value
        lib = _compile(nx, ny, ept_x, fpb_x, ept_y, fpb_y)

    # Validate shared memory
    xs, ys = ctypes.c_int(), ctypes.c_int()
    lib.conv2d_smem_size(ctypes.byref(xs), ctypes.byref(ys))
    while max(xs.value, ys.value) > _max_smem():
        fpb = max(fpb_x, fpb_y)
        if fpb <= 1:
            raise RuntimeError(f"No valid FPB for {nx}x{ny}: smem too large")
        fpb_x = fpb_y = fpb // 2
        lib = _compile(nx, ny, ept_x, fpb_x, ept_y, fpb_y)
        lib.conv2d_smem_size(ctypes.byref(xs), ctypes.byref(ys))

    _lib_cache[key] = lib
    return lib


# ---------------------------------------------------------------------------
# Public: Conv2DCUFFTDX handle
# ---------------------------------------------------------------------------
class Conv2DCUFFTDX:
    """
    Wraps a cuFFTDx conv2d handle for a fixed (nx, ny) spatial size.

    Parameters
    ----------
    nx, ny : int
        Spatial dimensions of the 2D FFT (NX rows, NY columns).
    """

    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny
        self._lib = _get_lib(nx, ny)
        stream = cp.cuda.get_current_stream()
        self._stream_ptr = ctypes.c_void_p(stream.ptr)
        self._handle = ctypes.c_void_p(self._lib.conv2d_create(self._stream_ptr))

    def run(self, x: cp.ndarray, H: cp.ndarray, y: cp.ndarray, stream=None) -> None:
        """
        y[b] = IFFT2_unnorm(FFT2(x[b]) * H)  for each batch b.

        NOTE: result is NX*NY times the normalized cuPy ifft2 result.

        Parameters
        ----------
        x : (batches, NX, NY) complex64, contiguous
        H : (NX, NY)          complex64, contiguous
        y : (batches, NX, NY) complex64, contiguous output buffer
        """
        if stream is None:
            stream = cp.cuda.get_current_stream()
        s_ptr = ctypes.c_void_p(stream.ptr)
        batches = x.shape[0]
        self._lib.conv2d_run(
            self._handle,
            ctypes.c_void_p(x.data.ptr),
            ctypes.c_void_p(H.data.ptr),
            ctypes.c_void_p(y.data.ptr),
            ctypes.c_int(batches),
            s_ptr,
        )

    def __del__(self):
        if hasattr(self, '_lib') and hasattr(self, '_handle') and self._handle:
            self._lib.conv2d_destroy(self._handle)


# ---------------------------------------------------------------------------
# Availability flag  (checked at import time by propagation.py)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def cufftdx_available() -> bool:
    """True if the cuFFTDx path can be used. Evaluated (and cached) on first
    call rather than at import time, so importing the package costs no nvcc
    subprocess and emits no output."""
    if not _SRC.exists():
        warnings.warn(
            f"cuFFTDx source not found at {_SRC}. Falling back to cuPy FFT.",
            stacklevel=2,
        )
        return False
    if not (_MATHDX / "include").exists() or not _CUFFTDX_EX.exists():
        warnings.warn(
            f"mathDX not found at {_MATHDX} (include/ or example/cufftdx/ missing). "
            f"Set MATHDX_ROOT to your mathDX installation. Falling back to cuPy FFT.",
            stacklevel=2,
        )
        return False
    try:
        r = subprocess.run([_NVCC, "--version"], capture_output=True)
        if r.returncode != 0:
            warnings.warn(
                "nvcc returned non-zero exit code. Falling back to cuPy FFT.",
                stacklevel=2,
            )
            return False
    except FileNotFoundError:
        warnings.warn(
            f"nvcc not found (NVCC={_NVCC!r}). Set NVCC env var or add nvcc to PATH. "
            f"Falling back to cuPy FFT.",
            stacklevel=2,
        )
        return False
    return True



def precompile(nx: int, ny: int) -> None:
    """Compile (or verify) the cuFFTDx .so for the given grid size.

    Call this on rank 0 before the MPI barrier so that all other ranks
    can simply dlopen the already-built library when they construct their
    Conv2DCUFFTDX handles.

    No-op if cuFFTDx is unavailable.
    """
    if cufftdx_available():
        _get_lib(nx, ny)
