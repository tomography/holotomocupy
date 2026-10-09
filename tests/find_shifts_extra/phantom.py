"""A textured phantom for the autofocus tests.

Smooth phantoms make an alignment test vacuous: entropy scores the texture, so
the object has to have some.  This is a few ellipsoids filled with a
band-limited random field, which gives both sharp edges and interior grain.
"""
import numpy as np


def phantom(nz, n, seed=0):
    """[nz, n, n] float32, values ~[0, 1], centred in the frame."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:nz, 0:n, 0:n].astype('float32')
    zz = (zz - (nz - 1) / 2) / (n / 2)
    yy = (yy - (n - 1) / 2) / (n / 2)
    xx = (xx - (n - 1) / 2) / (n / 2)

    # band-limited noise: white field, low-pass in Fourier, normalised
    g = rng.standard_normal((nz, n, n)).astype('float32')
    f = np.fft.fftn(g)
    kz, ky, kx = np.meshgrid(np.fft.fftfreq(nz), np.fft.fftfreq(n),
                             np.fft.fftfreq(n), indexing='ij')
    f *= np.exp(-0.5 * ((kz**2 + ky**2 + kx**2) / 0.04**2))
    g = np.real(np.fft.ifftn(f)).astype('float32')
    g = (g - g.mean()) / (g.std() + 1e-12)

    out = np.zeros((nz, n, n), dtype='float32')
    for cz, cy, cx, rz, ry, rx, amp in (
            (0.0, 0.00, 0.00, 1.4, 0.72, 0.72, 1.0),
            (0.0, -0.18, 0.12, 0.9, 0.34, 0.26, 0.5),
            (0.0, 0.22, -0.20, 1.1, 0.20, 0.30, -0.4),
            (0.0, 0.05, 0.30, 0.7, 0.14, 0.14, 0.8)):
        m = (((zz - cz) / rz)**2 + ((yy - cy) / ry)**2 + ((xx - cx) / rx)**2) < 1
        out[m] += amp
    out *= 1.0 + 0.35 * g
    return np.clip(out, 0.0, None).astype('float32')
