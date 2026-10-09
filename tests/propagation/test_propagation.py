import sys
import os
import numpy as np
import cupy as cp
os.environ['PATH'] = '/local/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin:' + os.environ.get('PATH', '')
sys.path.insert(0, '../../src')
from holotomocupy.propagation import Propagation

# ── Acquisition parameters ───────────────────────────────────────────────────

n          = 1024
nz         = 1024
ntheta     = 4
energy     = 17.1                        # keV
wavelength = 1.24e-9 / energy            # m
voxelsize  = 20e-9                       # m

sx0 = -3.135e-3
z1  = np.array([5.110, 5.464, 6.879, 9.817]) * 1e-3 - sx0
focustodetectordistance = 1.217
z2  = focustodetectordistance - z1
distances = (z1 * z2) / focustodetectordistance
ndist = len(distances)

cl      = Propagation(n, nz, ntheta, ndist, wavelength, voxelsize, distances)

# cuPy-only reference (no cuFFTDx) for correctness and timing comparison.
# _plan_2d is only created when _use_cufftdx is False at construction,
# so build it explicitly after flipping the flag.
import cupyx.scipy.fft as cufft
cl_cupy = Propagation(n, nz, ntheta, ndist, wavelength, voxelsize, distances)
cl_cupy._use_cufftdx = False
if not hasattr(cl_cupy, '_plan_2d'):
    cl_cupy._plan_2d = cufft.get_fft_plan(cl_cupy._buf_big, axes=(-2, -1), value_type='C2C')

print(f'n={n}, nz={nz}, ntheta={ntheta}, ndist={ndist}')
print(f'distances: {distances}')
print(f'cuFFTDx enabled: {cl._use_cufftdx}')

# ── Adjoint test ─────────────────────────────────────────────────────────────
# Verify <D(psi), phi> = <psi, DT(phi)> for each distance.

print('\nAdjoint test  <D psi, phi> vs <psi, DT phi>\n')
print(f'{"j":>3}  {"<D psi, phi>":>24}  {"<psi, DT phi>":>24}  {"rel. error":>12}')
print('-' * 72)

for j in range(ndist):
    psi = (cp.random.randn(ntheta, nz, n) + 1j * cp.random.randn(ntheta, nz, n)).astype('complex64')
    phi = (cp.random.randn(ntheta, nz, n) + 1j * cp.random.randn(ntheta, nz, n)).astype('complex64')

    lhs = cp.sum(cl.D(psi, j) * phi.conj())
    rhs = cp.sum(psi * cl.DT(phi, j).conj())

    lhs_np = complex(lhs.get())
    rhs_np = complex(rhs.get())
    err    = abs(lhs_np - rhs_np) / (abs(lhs_np) + 1e-30)

    print(f'{j:>3}  {lhs_np:>24.6e}  {rhs_np:>24.6e}  {err:>12.2e}')

# ── Energy conservation ───────────────────────────────────────────────────────
# Fresnel propagation is unitary on the infinite domain. With symmetric padding
# the energy of D(psi) should be very close to that of psi.

print('\nEnergy conservation  ||D psi||^2 / ||psi||^2\n')
print(f'{"j":>3}  {"||psi||^2":>16}  {"||D psi||^2":>16}  {"ratio":>10}')
print('-' * 52)

for j in range(ndist):
    psi = (cp.random.randn(ntheta, nz, n) + 1j * cp.random.randn(ntheta, nz, n)).astype('complex64')

    e_in  = float(cp.sum(cp.abs(psi)       ** 2).get())
    e_out = float(cp.sum(cp.abs(cl.D(psi, j)) ** 2).get())
    print(f'{j:>3}  {e_in:>16.4f}  {e_out:>16.4f}  {e_out/e_in:>10.6f}')

# ── Shape test ────────────────────────────────────────────────────────────────
# Check that 2-D and 3-D inputs produce the expected output shapes.

psi_2d = cp.ones([nz, n], dtype='complex64')
psi_3d = cp.ones([ntheta, nz, n], dtype='complex64')

out_2d = cl.D(psi_2d, 0)
out_3d = cl.D(psi_3d, 0)

assert out_2d.shape == (nz, n),         f'Expected ({nz},{n}), got {out_2d.shape}'
assert out_3d.shape == (ntheta, nz, n), f'Expected ({ntheta},{nz},{n}), got {out_3d.shape}'

out_2d_t = cl.DT(psi_2d, 0)
out_3d_t = cl.DT(psi_3d, 0)

assert out_2d_t.shape == (nz, n),         f'Expected ({nz},{n}), got {out_2d_t.shape}'
assert out_3d_t.shape == (ntheta, nz, n), f'Expected ({ntheta},{nz},{n}), got {out_3d_t.shape}'

print('\nShape test passed: D and DT return correct shapes for 2D and 3D inputs.')

# ── D / DT correctness ────────────────────────────────────────────────────────
# Verify that the cuFFTDx-based D and DT match a direct cuPy FFT reference.

print('\nD vs D_cupy  max|D - D_cupy| / max|D_cupy|\n')
print(f'{"j":>3}  {"max|D-D_cupy|":>16}  {"max|D_cupy|":>14}  {"rel. error":>12}')
print('-' * 52)

for j in range(ndist):
    psi = (cp.random.randn(ntheta, nz, n) + 1j * cp.random.randn(ntheta, nz, n)).astype('complex64')
    ref = cl_cupy.D(psi, j)
    out = cl.D(psi, j)
    abs_err = float(cp.abs(out - ref).max().get())
    scale   = float(cp.abs(ref).max().get())
    print(f'{j:>3}  {abs_err:>16.3e}  {scale:>14.3e}  {abs_err/scale:>12.2e}')

print()
print('DT vs DT_cupy  max|DT - DT_cupy| / max|DT_cupy|\n')
print(f'{"j":>3}  {"max|DT-DT_cupy|":>16}  {"max|DT_cupy|":>14}  {"rel. error":>12}')
print('-' * 52)

for j in range(ndist):
    phi = (cp.random.randn(ntheta, nz, n) + 1j * cp.random.randn(ntheta, nz, n)).astype('complex64')
    ref = cl_cupy.DT(phi, j)
    out = cl.DT(phi, j)
    abs_err = float(cp.abs(out - ref).max().get())
    scale   = float(cp.abs(ref).max().get())
    print(f'{j:>3}  {abs_err:>16.3e}  {scale:>14.3e}  {abs_err/scale:>12.2e}')

# ── Benchmark: D / DT throughput ─────────────────────────────────────────────

NREP   = 100
WARM   = 10
j      = 0
stream = cp.cuda.get_current_stream()
psi    = (cp.random.randn(ntheta, nz, n) + 1j * cp.random.randn(ntheta, nz, n)).astype('complex64')

def time_fn(fn, nrep=NREP, warm=WARM):
    for _ in range(warm): fn()
    stream.synchronize()
    t0 = cp.cuda.Event(); t1 = cp.cuda.Event()
    t0.record(stream)
    for _ in range(nrep): fn()
    t1.record(stream); t1.synchronize()
    return cp.cuda.get_elapsed_time(t0, t1) / nrep

t_cufftdx_D  = time_fn(lambda: cl.D(psi, j))
t_cufftdx_DT = time_fn(lambda: cl.DT(psi, j))
t_cupy_D     = time_fn(lambda: cl_cupy.D(psi, j))
t_cupy_DT    = time_fn(lambda: cl_cupy.DT(psi, j))

print(f'\n{"Method":<16}  {"cuFFTDx ms":>12}  {"cuPy ms":>10}  {"speedup":>9}')
print('-' * 54)
print(f'{"D":<16}  {t_cufftdx_D:>12.3f}  {t_cupy_D:>10.3f}  {t_cupy_D/t_cufftdx_D:>9.2f}x')
print(f'{"DT":<16}  {t_cufftdx_DT:>12.3f}  {t_cupy_DT:>10.3f}  {t_cupy_DT/t_cufftdx_DT:>9.2f}x')

# ── Per-step profiling: D vs D_cupy ──────────────────────────────────────────

NREP   = 200
WARM   = 10
j      = 0
psi    = (cp.random.randn(ntheta, nz, n) + 1j * cp.random.randn(ntheta, nz, n)).astype('complex64')
phi    = (cp.random.randn(ntheta, nz, n) + 1j * cp.random.randn(ntheta, nz, n)).astype('complex64')

# Propagation stores the kernel separably (fker_y, fker_x); rebuild the dense
# 2-D kernel here, since these sub-step timings feed it to conv2d.run directly.
fker_j = cp.multiply(cl.fker_y[j][:, None], cl.fker_x[j][None, :])

# D sub-steps (cuFFTDx)
ff_pad = cp.empty([ntheta, 2*nz, 2*n], dtype='complex64')
cl._fwd_pad(psi, ff_pad)
y_buf  = cp.empty_like(ff_pad)

steps_D = {
    '_fwd_pad':    lambda: cl._fwd_pad(psi, ff_pad),
    'conv2d.run':  lambda: cl._conv2d.run(ff_pad, fker_j, y_buf),
    'crop':        lambda: y_buf[:, cl.nz//2:-cl.nz//2, cl.n//2:-cl.n//2],
    'D total':     lambda: cl.D(psi, j),
}

# D sub-steps (cuPy)
ff_pad2 = cp.empty([ntheta, 2*nz, 2*n], dtype='complex64')
cl._fwd_pad(psi, ff_pad2)
ff_fft  = cp.fft.fft2(ff_pad2)
ff_mul  = ff_fft * fker_j

steps_D_cupy = {
    '_fwd_pad':       lambda: cl._fwd_pad(psi, ff_pad2),
    'fft2':           lambda: cp.fft.fft2(ff_pad2),
    'multiply fker':  lambda: ff_fft.__mul__(fker_j),
    'ifft2(unnorm)':  lambda: cp.fft.ifft2(ff_mul, norm="forward"),
    'crop':           lambda: ff_mul[:, cl.nz//2:-cl.nz//2, cl.n//2:-cl.n//2],
    'D total':        lambda: cl_cupy.D(psi, j),
}

# DT sub-steps (cuFFTDx)
ff_zpad = cp.zeros([ntheta, 2*nz, 2*n], dtype='complex64')
ff_zpad[:, cl.nz//2:-cl.nz//2, cl.n//2:-cl.n//2] = phi
y_buf2  = cp.empty_like(ff_zpad)

steps_DT = {
    'zero_pad+copy':  lambda: (cl._buf_big[:ntheta].fill(0),
                               cl._buf_big[:ntheta].__setitem__(
                                   (slice(None), slice(cl.nz//2,-cl.nz//2), slice(cl.n//2,-cl.n//2)), phi)),
    'conv2d.run':     lambda: cl._conv2d.run(ff_zpad, fker_j.conj(), y_buf2),
    '_adj_pad':       lambda: cl._adj_pad(y_buf2, cp.empty([ntheta, nz, n], dtype='complex64')),
    'DT total':       lambda: cl.DT(phi, j),
}

# DT sub-steps (cuPy)
def _zpad(phi, ntheta, nz, n, cl):
    ff = cp.zeros([ntheta, 2*cl.nz, 2*cl.n], dtype='complex64')
    ff[:, cl.nz//2:-cl.nz//2, cl.n//2:-cl.n//2] = phi
    return ff

ff_zpad2 = cp.zeros([ntheta, 2*nz, 2*n], dtype='complex64')
ff_zpad2[:, cl.nz//2:-cl.nz//2, cl.n//2:-cl.n//2] = phi
ff_fft3  = cp.fft.fft2(ff_zpad2)
ff_mul3  = ff_fft3 * fker_j.conj()

steps_DT_cupy = {
    'zeros+copy':      lambda: _zpad(phi, ntheta, nz, n, cl),
    'fft2':            lambda: cp.fft.fft2(ff_zpad2),
    'multiply fker_c': lambda: ff_fft3.__mul__(fker_j.conj()),
    'ifft2(unnorm)':   lambda: cp.fft.ifft2(ff_mul3, norm="forward"),
    '_adj_pad':        lambda: cl._adj_pad(ff_mul3, cp.empty([ntheta, nz, n], dtype='complex64')),
    'DT total':        lambda: cl_cupy.DT(phi, j),
}

print(f'\n{"Step":<22}  {"ms":>8}')
print('── D (cuFFTDx) ───────────────────')
for name, fn in steps_D.items():
    print(f'  {name:<20}  {time_fn(fn):>8.3f}')

print()
print('── D (cuPy) ──────────────────────')
for name, fn in steps_D_cupy.items():
    print(f'  {name:<20}  {time_fn(fn):>8.3f}')

print()
print('── DT (cuFFTDx) ──────────────────')
for name, fn in steps_DT.items():
    print(f'  {name:<20}  {time_fn(fn):>8.3f}')

print()
print('── DT (cuPy) ─────────────────────')
for name, fn in steps_DT_cupy.items():
    print(f'  {name:<20}  {time_fn(fn):>8.3f}')
