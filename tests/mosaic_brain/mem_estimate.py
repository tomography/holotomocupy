#!/usr/bin/env python3
"""
Memory estimator for the mosaic holotomography test (gen_data.py / step6.py).

Pure arithmetic -- no GPU, no MPI, no data needed.  Every term below is taken
from an allocation that actually exists in the code:

  host (pinned, holotomocupy/rec_mpi.py:alloc_arrays)
      vars/grads/etas['obj']   3 x [local_nzobj, nobj, nobj]  complex64
      vars/grads/etas['proj']  3 x [local_ntheta, nzobj, nobj] complex64
      proj_tmp                 1 x [ntheta, local_nzobj, nobj] complex64
      data                     1 x [ndist, local_ntheta, nz, n] float32
      vars/grads/etas['prb']   3 x [ndist, nz, n] complex64      (replicated)
      vars/grads/etas['pos']   3 x [ndist, local_ntheta, 2] float32
      (+2 more obj-sized pinned slabs when lam_laplacian > 0: u_pad / e_pad)

  gpu
      chunking slab   Chunking(nbytes) with nbytes from Rec._chunking_pool_bytes:
                      2.1 * max(3*proj_chunk + data_chunk, 3*obj_chunk)
      Tomo            _buf_fde  [nchunk, 2*nobj, 2*nobj] complex64   <- usually the peak
                      _buf_sino [ntheta, nchunk, nobj]  complex64   <- GLOBAL ntheta
                      + two cuFFT plan work areas of comparable size
      Propagation     _buf_big  [nchunk, 2*nz, 2*n] complex64
      Shift           coeff plan work area ~ [nchunk, nzobj, nobj] complex64
                      + per-call transients (S/Sadj/coeff cache)
      ref             [ndist, nz, n] float32
      + CUDA context / cuFFT / cuFFTDx libraries

Usage
    python mem_estimate.py                          # defaults = config_gen/config_step6
    python mem_estimate.py --bins 0 1 2 3 --nranks 8
    python mem_estimate.py --conf config_step6.conf --ntheta 3000
    python mem_estimate.py --nranks 1 2 4 8 --gpu-gb 40 --host-gb 1024
"""

import argparse
import configparser

GiB = 1024.0 ** 3
C64 = 8      # complex64
F32 = 4


def local(total, rank, size):
    """holotomocupy.mpi_functions.get_local_chunk, size of rank's slab."""
    q, r = divmod(total, size)
    return q + (1 if rank < r else 0)


def geom(a, b):
    """Everything that depends on the binning level b."""
    g = argparse.Namespace()
    g.bin   = b
    g.n     = a.ndet  >> b
    g.nz    = a.ndet  >> b
    g.nobj  = a.nobj0 >> b
    g.nzobj = a.nzobj0 >> b
    g.ndist = a.ntiles * a.ndist_tile
    g.voxel = a.voxel0 * (1 << b)
    return g


def rec_host(a, g, nranks):
    """Pinned host bytes on the heaviest rank (rank 0), and the sum over ranks."""
    lnz = local(g.nzobj,  0, nranks)
    lth = local(a.ntheta, 0, nranks)

    obj   = lnz * g.nobj * g.nobj * C64
    nobj_bufs = 3 + (2 if a.lam_laplacian > 0 else 0)
    proj  = lth * g.nzobj * g.nobj * C64
    ptmp  = a.ntheta * lnz * g.nobj * C64
    data  = g.ndist * lth * g.nz * g.n * F32
    prb   = 3 * g.ndist * g.nz * g.n * C64
    pos   = 4 * g.ndist * lth * 2 * F32          # vars/grads/etas + pos_init
    ref_h = 0

    d = {
        'obj  (%d x %d x %d x %d)' % (nobj_bufs, lnz, g.nobj, g.nobj): nobj_bufs * obj,
        'proj (3 x %d x %d x %d)'  % (lth, g.nzobj, g.nobj):           3 * proj,
        'proj_tmp (%d x %d x %d)'  % (a.ntheta, lnz, g.nobj):          ptmp,
        'data (%d x %d x %d x %d)' % (g.ndist, lth, g.nz, g.n):        data,
        'prb + pos + ref':                                             prb + pos + ref_h,
    }
    peak = sum(d.values())

    # sum over all ranks (what a single fat node must hold if all ranks are on it)
    tot = (nobj_bufs * g.nzobj * g.nobj * g.nobj * C64
           + 4 * a.ntheta * g.nzobj * g.nobj * C64          # 3 proj + proj_tmp
           + g.ndist * a.ntheta * g.nz * g.n * F32
           + nranks * (prb + pos))
    return d, peak, tot


def rec_gpu(a, g, nchunk):
    """Device bytes per rank."""
    proj_chunk = nchunk * g.nzobj * g.nobj * C64
    obj_chunk  = nchunk * g.nobj  * g.nobj * C64
    data_chunk = nchunk * g.nz    * g.n    * F32

    pool  = 2.1 * max(3 * proj_chunk + data_chunk, 3 * obj_chunk)
    fde   = nchunk * (2 * g.nobj) ** 2 * C64
    sino  = a.ntheta * nchunk * g.nobj * C64
    big   = nchunk * (2 * g.nz) * (2 * g.n) * C64
    plans = a.plan_factor * (fde + sino)                 # cuFFT work areas
    shift = (1 + a.transient_factor) * proj_chunk        # coeff plan + transients
    ref   = g.ndist * g.nz * g.n * F32
    ctx   = a.ctx_gb * GiB

    d = {
        'chunking slab  (nchunk=%d)' % nchunk: pool,
        'Tomo _buf_fde  [%d,%d,%d]' % (nchunk, 2*g.nobj, 2*g.nobj): fde,
        'Tomo _buf_sino [%d,%d,%d]' % (a.ntheta, nchunk, g.nobj):   sino,
        'cuFFT plan work areas':               plans,
        'Prop _buf_big + Shift + transients':  big + shift,
        'ref + CUDA context/libs':             ref + ctx,
    }
    return d, sum(d.values())


def gen_host(a, g, nranks):
    """gen_data.py: Rec sized for all ntheta angles, grads/etas cleared."""
    lnz = local(g.nzobj, 0, nranks)
    lth = local(a.ntheta, 0, nranks)
    obj  = lnz * g.nobj * g.nobj * C64
    proj = lth * g.nzobj * g.nobj * C64
    ptmp = a.ntheta * lnz * g.nobj * C64
    data = g.ndist * lth * g.nz * g.n * F32
    prb  = g.ndist * g.nz * g.n * C64
    return {
        'obj  (%d x %d x %d)'      % (lnz, g.nobj, g.nobj):     obj,
        'proj (%d x %d x %d)'      % (lth, g.nzobj, g.nobj):    proj,
        'proj_tmp (%d x %d x %d)'  % (a.ntheta, lnz, g.nobj):   ptmp,
        'data (%d x %d x %d x %d)' % (g.ndist, lth, g.nz, g.n): data,
        'prb + ref + pos':                                      3 * prb,
    }


def traffic(a, g, nranks, nchunk):
    """Per-rank bytes moved per BH iteration, and the time that implies.

    obj/proj live in pinned host memory and stream over PCIe on every
    @gpu_batch pass.  gradients_cascade loops over all ndist distances and
    re-reads the local proj each time (rec_mpi.py, `for k in range(self.ndist)`),
    so one cascade pass costs 3*ndist*proj: vars['proj'] in, grads['proj']
    read-modify-write.  hessian_cascade has the same shape and runs several
    times per iteration (compute_beta x2, compute_alpha, check_approximation).
    fwd_tomo/adj_tomo each stream the local object plus proj_tmp.
    Two Alltoallw redistributions per iteration move proj_tmp over the network.
    """
    lnz = local(g.nzobj,  0, nranks)
    lth = local(a.ntheta, 0, nranks)
    proj = lth * g.nzobj * g.nobj * C64
    obj  = lnz * g.nobj  * g.nobj * C64
    ptmp = a.ntheta * lnz * g.nobj * C64
    data = g.ndist * lth * g.nz * g.n * F32

    cascade = 3 * g.ndist * proj + data
    pcie    = a.cascades_per_iter * cascade + 2 * (obj + ptmp)
    net     = 2 * ptmp
    return {
        'one cascade pass (x%d/iter)' % a.cascades_per_iter: cascade,
        'fwd_tomo + adj_tomo':                               2 * (obj + ptmp),
        'PCIe per iteration':                                pcie,
        'network per iteration (2 x Alltoallw)':             net,
    }, pcie, net


def fmt(nbytes):
    return '%8.2f GiB' % (nbytes / GiB)


def table(title, d, total=None):
    print('  ' + title)
    for k, v in d.items():
        print('    %-42s %s' % (k, fmt(v)))
    if total is None:
        total = sum(d.values())
    print('    %-42s %s' % ('-' * 42, '-' * 12))
    print('    %-42s %s' % ('TOTAL', fmt(total)))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument('--conf', help='config_gen.conf / config_step6.conf to take defaults from')
    p.add_argument('--ndet',   type=int,   default=2048,  help='unbinned detector size')
    p.add_argument('--nobj0',  type=int,   default=12288, help='mosaic width, finest grid')
    p.add_argument('--nzobj0', type=int,   default=5888,  help='mosaic height, finest grid')
    p.add_argument('--voxel0', type=float, default=100.0, help='nm at bin 0')
    p.add_argument('--ntiles',     type=int, default=10)
    p.add_argument('--ndist-tile', type=int, default=4)
    p.add_argument('--ntheta',     type=int, default=3000)
    p.add_argument('--nchunk',     type=int, default=8)
    p.add_argument('--lam-laplacian', type=float, default=0.0)
    p.add_argument('--bins',   type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--nranks', type=int, nargs='+', default=[8])
    p.add_argument('--gpu-gb',  type=float, default=0.0,
                   help='per-GPU memory; if given, flag configurations that do not fit')
    p.add_argument('--host-gb', type=float, default=0.0,
                   help='per-node host RAM; if given, flag configurations that do not fit')
    p.add_argument('--ctx-gb', type=float, default=0.8,
                   help='CUDA context + cuFFT/cuFFTDx library footprint')
    p.add_argument('--plan-factor', type=float, default=1.0,
                   help='cuFFT work area as a multiple of the planned buffers')
    p.add_argument('--transient-factor', type=float, default=6.0,
                   help='Shift/coeff transients as a multiple of one proj chunk')
    p.add_argument('--pcie-gbs', type=float, default=25.0,
                   help='effective host<->device bandwidth per GPU, GB/s')
    p.add_argument('--net-gbs', type=float, default=12.5,
                   help='effective network bandwidth per rank, GB/s')
    p.add_argument('--cascades-per-iter', type=float, default=5.0,
                   help='cascade passes per BH iteration (1 gradient + hessians)')
    p.add_argument('--niter', type=int, default=1025,
                   help='iterations, for the walltime projection')
    p.add_argument('--brief', action='store_true', help='one summary line per (bin, nranks)')
    a = p.parse_args()

    if a.conf:
        c = configparser.ConfigParser(inline_comment_prefixes=('#',))
        c.read_string('[DEFAULT]\n' + open(a.conf).read())
        cfg = c['DEFAULT']
        if cfg.get('ntile_h', fallback=None):
            a.ntiles = cfg.getint('ntile_h') * cfg.getint('ntile_v')
        elif cfg.get('tiles', fallback=None):
            a.ntiles = len([t for t in cfg.get('tiles').split(',') if t.strip()])
        a.ndist_tile = cfg.getint('ndist', fallback=a.ndist_tile)
        a.ntheta     = cfg.getint('ntheta', fallback=a.ntheta)
        a.nchunk     = cfg.getint('nchunk', fallback=a.nchunk)
        a.lam_laplacian = cfg.getfloat('lam_laplacian', fallback=a.lam_laplacian)
        a.ndet   = cfg.getint('ndet', fallback=a.ndet)
        # nobj/nzobj in config_step6 are already binned; scale back to the finest grid
        if cfg.get('bin', fallback=None) and cfg.get('nobj', fallback=None):
            b0 = cfg.getint('bin')
            if cfg.getint('nobj') << b0 != a.nobj0 and not cfg.get('ntile_h', fallback=None):
                a.nobj0  = cfg.getint('nobj')  << b0
                a.nzobj0 = cfg.getint('nzobj') << b0
                a.ndet   = cfg.getint('n')     << b0

    print('=' * 78)
    print('  detector %d^2 unbinned   mosaic %d x %d finest px   voxel %.1f nm'
          % (a.ndet, a.nobj0, a.nzobj0, a.voxel0))
    print('  %d tiles x %d distances = %d, ntheta = %d, nchunk = %d'
          % (a.ntiles, a.ndist_tile, a.ntiles * a.ndist_tile, a.ntheta, a.nchunk))
    print('=' * 78)

    if a.brief:
        print('%4s %6s %6s %6s %6s %8s %11s %11s %11s'
              % ('bin', 'n', 'nobj', 'nzobj', 'ranks', 'disk', 'host/rank',
                 'host total', 'gpu/rank'))
        print('%s %11s %9s' % (' ' * 66, 'PCIe/iter', 'walltime'))
    for b in a.bins:
        g = geom(a, b)
        disk = g.ndist * a.ntheta * g.nz * g.n * F32
        for R in a.nranks:
            hd, hpeak, htot = rec_host(a, g, R)
            gd, gtot        = rec_gpu(a, g, a.nchunk)
            if a.brief:
                warn = ''
                if a.gpu_gb  and gtot  > a.gpu_gb  * GiB: warn += ' GPU!'
                if a.host_gb and hpeak > a.host_gb * GiB: warn += ' HOST!'
                _, pcie, net = traffic(a, g, R, a.nchunk)
                t_it = pcie / (a.pcie_gbs * 1e9) + net / (a.net_gbs * 1e9)
                print('%4d %6d %6d %6d %6d %8.1fG %10.1fG %10.1fG %10.2fG'
                      ' %9.2fTB %7.1fh%s'
                      % (b, g.n, g.nobj, g.nzobj, R, disk / GiB, hpeak / GiB,
                         htot / GiB, gtot / GiB, pcie / 1e12,
                         t_it * a.niter / 3600, warn))
                continue

            print()
            print('-' * 78)
            print('BIN %d   detector %d x %d   object %d x %d x %d   voxel %.1f nm'
                  % (b, g.nz, g.n, g.nzobj, g.nobj, g.nobj, g.voxel))
            print('        %d MPI rank(s): local_nzobj=%d  local_ntheta=%d'
                  % (R, local(g.nzobj, 0, R), local(a.ntheta, 0, R)))
            print('-' * 78)
            print('  raw data on disk (float32): %s total, %s per tile file'
                  % (fmt(disk), fmt(disk / a.ntiles)))
            print('  reconstructed object (complex64): %s'
                  % fmt(g.nzobj * g.nobj * g.nobj * C64))
            print()
            table('RECONSTRUCTION - host RAM, heaviest rank', hd, hpeak)
            print('    %-42s %s' % ('all %d ranks together' % R, fmt(htot)))
            print()
            table('RECONSTRUCTION - GPU per rank', gd, gtot)
            if a.gpu_gb:
                print('    %-42s %s' % ('fits in %.0f GiB GPU?' % a.gpu_gb,
                                        'YES' if gtot <= a.gpu_gb * GiB else 'NO'))
            print()
            td, pcie, net = traffic(a, g, R, a.nchunk)
            t_it = pcie / (a.pcie_gbs * 1e9) + net / (a.net_gbs * 1e9)
            table('DATA MOVEMENT per rank per BH iteration', td, pcie + net)
            print('    %-42s %8.1f s' % ('time/iter @ %.0f GB/s PCIe, %.1f GB/s net'
                                         % (a.pcie_gbs, a.net_gbs), t_it))
            print('    %-42s %8.1f h' % ('%d iterations' % a.niter,
                                         t_it * a.niter / 3600))
            print()
            table('GENERATION (gen_data.py) - host RAM, heaviest rank',
                  gen_host(a, g, R))
    print()


if __name__ == '__main__':
    main()
