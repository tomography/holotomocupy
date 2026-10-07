"""Synthetic sample and probe for the demo: a Siemens star and a 3-D phantom.

Shared by the notebooks and by the MPI scripts next to them, so the two always
build the same data.  Pure numpy/scipy except where cupy is faster; no MPI.
"""
import os
import urllib.request

import numpy as np
import scipy.ndimage as ndimage
from scipy.fft import fft2, fftn, fftshift, ifft2, ifftn

PRB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'prb_id16a')
# The measured probe is 2 x 64 MB, too big for git, so it is fetched on first
# use and cached in PRB_DIR (gitignored).  Same files tests/nfp downloads.
PRB_URL = ('https://g-110014.fd635.8443.data.globus.org/holotomocupy/'
           'examples_synthetic/data/prb_id16a')
PRB_FILES = ('prb_abs_2048.tiff', 'prb_phase_2048.tiff')


def fetch_probe():
    """Download the ID16A probe once; rank 0 only, others wait."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, barrier = comm.Get_rank(), comm.Barrier
    except Exception:
        rank, barrier = 0, lambda: None
    if rank == 0:
        os.makedirs(PRB_DIR, exist_ok=True)
        for f in PRB_FILES:
            dest = os.path.join(PRB_DIR, f)
            if os.path.exists(dest):
                continue
            print(f'demo: fetching {f} (64 MB) into {PRB_DIR}', flush=True)
            tmp = dest + '.part'          # so an interrupted get is not cached
            urllib.request.urlretrieve(f'{PRB_URL}/{f}', tmp)
            os.replace(tmp, dest)
    barrier()


# ----------------------------------------------------------- 2-D: the star --
def siemens_star(nobj, step_deg=15):
    """(nobj, nobj) float32 Siemens-star mask, spokes every step_deg."""
    def rotate(pts, ang, c):
        s, co = np.sin(ang), np.cos(ang)
        return (pts - c) @ np.array([[co, -s], [s, co]], 'float32').T + c

    def tri(X, Y, t):
        (x1, y1), (x2, y2), (x3, y3) = t
        d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if d == 0:
            return np.zeros_like(X, dtype=bool)
        a = ((y2 - y3) * (X - x3) + (x3 - x2) * (Y - y3)) / d
        b = ((y3 - y1) * (X - x3) + (x1 - x3) * (Y - y3)) / d
        return (a >= 0) & (b >= 0) & (1 - a - b >= 0)

    t0 = np.array([(1.5 * nobj // 16, nobj // 2 - nobj // 32),
                   (1.5 * nobj // 16, nobj // 2 + nobj // 32),
                   (nobj // 2 - nobj // 128, nobj // 2)], dtype='float32')
    yy, xx = np.mgrid[0:nobj, 0:nobj]
    c = np.array([nobj / 2, nobj / 2], dtype='float32')
    star = np.zeros((nobj, nobj), dtype='float32')
    for deg in range(0, 360, step_deg):
        star += tri(xx, yy, rotate(t0, np.deg2rad(deg), c)).astype('float32')
    return star / (star.max() or 1.0)


def star_projection(nobj, delta_beta=29.0, smooth=8.0):
    """complex64 (nobj, nobj): the star as a transmission, -delta + i*beta."""
    star = siemens_star(nobj)
    v = np.arange(-nobj // 2, nobj // 2, dtype='float32') / nobj
    vx, vy = np.meshgrid(v, v, indexing='ij')
    g = fftshift(np.exp(-smooth * (vx**2 + vy**2)).astype('float32'))
    star = ifft2(fft2(star) * g).real
    star = star / star.max() * (np.pi / 8)          # phase range [0, pi/8]
    return (-star + 1j * star / delta_beta).astype('complex64')


# -------------------------------------------------------- 3-D: the phantom --
def _frame_edges(cube, p1, p2):
    """The 12 edges of the axis-aligned cube [p1, p2]^3, set to 1."""
    for a, b in ((p1, p1), (p1, p2), (p2, p1), (p2, p2)):
        cube[p1:p2, a, b] = 1
        cube[a, p1:p2, b] = 1
        cube[a, b, p1:p2] = 1


def _rotate3d(vol, ang_xy=28, ang_xz=45, order=1):
    a, b = np.deg2rad(ang_xy), np.deg2rad(ang_xz)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    A = np.linalg.inv(Ry @ Rz)
    c = (np.array(vol.shape) - 1) / 2.0
    return ndimage.affine_transform(vol, A, offset=c - A @ c, order=order,
                                    mode='constant', cval=0.0, prefilter=False)


def phantom3d(n, delta=1.0, beta=1e-2):
    """complex64 (n, n, n): nested cube frames, rotated off-axis and smoothed.

    Edges in every direction and a wide range of feature sizes, which is what
    makes it a fair test of resolution and of the position refinement.
    """
    amps = np.array([3, -3, 1, 3, -4, 1, 4], dtype='float32')
    dil = (np.array([33, 28, 25, 21, 16, 10, 3], dtype='float32') / 256.0) * n
    ax = np.arange(-n // 2, n // 2, dtype='float32')
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r2 = x * x + y * y + z * z
    del x, y, z

    obj = np.zeros((n, n, n), dtype='float32')
    cube = np.zeros((n, n, n), dtype='float32')
    r = int(n * 0.2)
    for amp, d in zip(amps, dil):
        cube.fill(0.0)
        _frame_edges(cube, n // 2 - r // 2, n // 2 + r // 2)
        fc = fftn(fftshift(cube), workers=-1)
        fs = fftn(fftshift((r2 < d * d).astype('float32')), workers=-1)
        obj += amp * (fftshift(ifftn(fc * fs, workers=-1)).real > 1.0)

    obj = _rotate3d(obj)
    obj = np.roll(obj, (-10 * n // 256, -15 * n // 256), axis=(1, 2))
    np.maximum(obj, 0, out=obj)

    v = np.arange(-n // 2, n // 2, dtype='float32') / n
    vx, vy, vz = np.meshgrid(v, v, v, indexing='ij')
    filt = fftshift(np.exp(-3.0 * (vx**2 + vy**2 + vz**2)).astype('float32'))
    obj = ifftn(fftn(obj) * filt).real
    obj[obj < 0] = 0
    return (obj * (-delta + 1j * beta)).astype('complex64')


# ------------------------------------------------------------- the probe ----
def id16a_probe(n, ndist=1, smooth=4.0):
    """complex64 (ndist, n, n): the measured ID16A probe, cropped and smoothed.

    Normalised to unit mean amplitude so the data are ~1 and the regularisation
    weights mean the same thing at any size.
    """
    from holotomocupy.utils import read_tiff

    fetch_probe()
    a = read_tiff(os.path.join(PRB_DIR, PRB_FILES[0]))[:ndist]
    p = read_tiff(os.path.join(PRB_DIR, PRB_FILES[1]))[:ndist]
    if a.shape[0] < ndist:
        raise ValueError(f'probe has {a.shape[0]} planes, need ndist={ndist}')
    if n > a.shape[1]:
        raise ValueError(f'probe is {a.shape[1]}^2, too small for n={n}')
    prb = (a * np.exp(1j * p)).astype('complex64')
    c0, c1 = prb.shape[1] // 2, prb.shape[2] // 2
    prb = prb[:, c0 - n // 2:c0 + n // 2, c1 - n // 2:c1 + n // 2]

    v = np.arange(-n // 2, n // 2, dtype='float32') / n
    vx, vy = np.meshgrid(v, v, indexing='ij')
    filt = fftshift(np.exp(-smooth * (vx**2 + vy**2)).astype('float32'))
    prb = ifft2(fft2(prb) * filt).astype('complex64')
    return prb / np.mean(np.abs(prb), axis=(1, 2))[:, None, None]
