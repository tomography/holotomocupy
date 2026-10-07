"""Detector PSF as one Gaussian on detector INTENSITY.

The detector blurs intensity incoherently, so this cannot be folded into the
coherent propagator and the data misfit is its only legal home.  Shared by
rec_mpi.Rec and rec_nfp_mpi.RecNFP, which both compare an intensity, so the
two solvers cannot drift apart on the tap convention.

This models the DETECTOR only.  Source-size / partial-coherence blur is a
different operator -- it scales with the magnification, so it is per-distance
-- and is NOT implemented.
"""

import numpy as np
import cupy as cp
from scipy.special import ive
from cupyx.scipy.ndimage import convolve1d


def psf_taps(psf_sigma):
    """Taps for ONE Gaussian blur of the detector intensity, or None for no blur.

    psf_sigma is the Gaussian sigma in BINNED DETECTOR PIXELS, the same at
    every distance because it is a property of the detector and not of the
    geometry.  Being in pixels it is per-bin: a fixed physical blur is half as
    many pixels each time you bin, so each rung of a ladder carries its own
    number and nothing is converted on the way in.

    The taps are the DISCRETE Gaussian, w_t = exp(-s^2) I_|t|(s^2), not a sampled
    continuous one.  Below ~0.7 px a sampled Gaussian aliases badly -- sampling
    sigma = 0.3 px on a unit grid gives taps whose realised sigma is 0.09 -- and
    the sigmas here are O(1) px.  The discrete Gaussian has second moment
    exactly sigma^2 at every width and tends to the sampled one for large sigma.
    Truncating at 4 sigma holds the realised sigma within 0.14% over
    0.3 <= sigma <= 6 px.

    Returns (sigma, cupy float32 taps or None).
    """
    sigma = float(psf_sigma)
    if sigma <= 0:
        return 0.0, None                  # identity: skip the convolution
    r = max(2, int(np.ceil(4.0 * sigma)))
    w = ive(np.abs(np.arange(-r, r + 1)), sigma * sigma)   # exp(-s^2) I_|t|(s^2)
    return sigma, cp.asarray(w / w.sum(), dtype='float32')


def psf_blur(x, w):
    """Gaussian blur over the last two axes.  Separable, periodic, SELF-ADJOINT.

    The kernel is symmetric and the wrap makes the operator circulant, so
    K^T = K exactly -- there is no separate adjoint to get wrong, and the
    gradient below can reuse this in both directions.  Periodic is also what the
    rest of the forward model already assumes: cl_prop.D is an FFT, so the field
    reaching the detector is periodic by construction.  (Zero padding would be
    self-adjoint too, but it darkens an r-wide rim toward zero where the data is
    ~1 -- a far bigger error than the sub-pixel blur being modelled.)

    Each convolve1d allocates a fresh output -- a gather stencil reads a
    neighbourhood of the input, so it cannot run in place -- hence two
    detector-shaped temporaries per call.  That is what _estimate_gpu_mem
    charges the cascade for.
    """
    return convolve1d(convolve1d(x, w, axis=-2, mode='wrap'),
                      w, axis=-1, mode='wrap')


