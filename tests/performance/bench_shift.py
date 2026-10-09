#!/usr/bin/env python
"""Shift vs ShiftFFT: cost of every shift variant, at the m and n the pipeline uses.

    PYTHONPATH=<repo>/src python tests/performance/bench_shift.py
    ... --n 256,512,1024 --reps 10 --csv out.csv

`shift_type` defaults to 'fft', so this is the table that says what that
costs.  Two things drive it:

  * m == 1 takes a pure phase ramp -- one fft2/ifft2 pair of the LARGE
    npsi x nzpsi grid, against Shift's single gather writing only the small
    n x nz output.
  * m != 1 takes the chirp-z (Bluestein): ~3 FFTs of the smallest 2-3-5-7
    smooth L >= N_in + N_out - 1 per axis, plus a transpose, and the chirp
    kernel and pad buffers are rebuilt on every call.  --mem reports what
    those buffers cost, which is what caps nchunk on the fft path.

The magnifications are the real ones: m[k] = z1[k]/z1[0] per distance, read
off the shipped configs.  A single-distance scan only ever sees m = 1.

Measured on an A100-40GB, n=2048 (npsi=3072), scratch per angle, m=1.51:
Shift 49-281 MiB, ShiftFFT 793-1353 MiB.  Scratch is exactly linear in
nchunk, so on 40 GB the chirp-z path OOMs at nchunk=32 where Shift still
fits ~100.  Pick nchunk from the worst op in play (dcurlySadjc).
"""
import argparse
import time

import numpy as np
import cupy as cp

from holotomocupy.shift import Shift
from holotomocupy.shift_fft import ShiftFFT

# z1 per distance (mm) from the shipped configs; m[k] = z1[k]/z1[0].
SCANS = {
    'ctxl_HT':      [6.164, 6.428, 7.486, 9.682],
    'AtomiumS1_HT': [3.698, 3.857, 4.491, 5.809],
    'demo/step6':   [5.110, 5.464, 6.879, 9.817],
    'single dist':  [1.0],
}


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', default='128,256,512,1024',
                   help='detector sizes; the object grid is 3n/2')
    p.add_argument('--nchunk', type=int, default=16,
                   help='angles per call -- 16 is what the ctxl/Atomium configs '
                        'use; 0 picks the largest that fits --max-gb')
    p.add_argument('--max-gb', type=float, default=1.0,
                   help='cap on one chunk-sized complex64 object array')
    p.add_argument('--reps', type=int, default=10, help='timed repeats')
    p.add_argument('--scan', default='', help='only this scan from SCANS')
    p.add_argument('--m', default='',
                   help='explicit magnifications, e.g. "1,1.1,1.51"; '
                        'overrides the per-scan list')
    p.add_argument('--csv', default='', help='also write the table here')
    p.add_argument('--mem', action='store_true',
                   help='peak scratch per call instead of time')
    return p.parse_args()


def peak_mib(fn):
    """Scratch the call allocates on top of its inputs, in MiB."""
    pool = cp.get_default_memory_pool()
    cp.cuda.Stream.null.synchronize()
    pool.free_all_blocks()
    base = pool.used_bytes()
    fn()
    cp.cuda.Stream.null.synchronize()
    hi = pool.total_bytes()
    pool.free_all_blocks()
    return (hi - base) / 2**20


def timed(fn, reps):
    fn()
    cp.cuda.Stream.null.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    cp.cuda.Stream.null.synchronize()
    return (time.time() - t) / reps * 1e3          # ms per call


def main():
    a = parse()
    sizes = [int(v) for v in a.n.split(',') if v.strip()]
    scans = {k: v for k, v in SCANS.items() if not a.scan or k == a.scan}
    if a.m:
        ms = sorted({round(float(v), 4) for v in a.m.split(',') if v.strip()})
        print(f'magnifications (given): {ms}')
    else:
        ms = sorted({round(float(z) / zs[0], 4) for zs in scans.values() for z in zs})
        print(f'magnifications in play: {ms}')
        print(f'  (m = z1[k]/z1[0];  ' + ';  '.join(
            f'{k}: {[round(z / v[0], 3) for z in v]}' for k, v in scans.items()) + ')')

    rows = []
    for n in sizes:
        npsi = 3 * n // 2
        nchunk = a.nchunk or max(1, int(a.max_gb * 2**30 / (npsi * npsi * 8)))
        if nchunk * npsi * npsi * 8 > a.max_gb * 2**30:
            print(f'  n={n}: skipped, {nchunk} x {npsi}^2 complex64 '
                  f'exceeds --max-gb {a.max_gb}')
            continue
        rng = np.random.default_rng(0)
        c = cp.asarray((rng.standard_normal((nchunk, npsi, npsi))
                        + 1j * rng.standard_normal((nchunk, npsi, npsi))).astype('complex64'))
        g = cp.asarray((rng.standard_normal((nchunk, n, n))
                        + 1j * rng.standard_normal((nchunk, n, n))).astype('complex64'))
        r = cp.asarray((2 * (rng.random((nchunk, 2)) - 0.5)).astype('float32'))
        c1 = cp.ascontiguousarray(c * np.float32(0.5))
        c2 = cp.ascontiguousarray(c * np.float32(-0.25))
        dr1 = cp.asarray((rng.random((nchunk, 2)) - 0.5).astype('float32'))
        dr2 = cp.asarray((rng.random((nchunk, 2)) - 0.5).astype('float32'))
        dm1 = cp.asarray(1e-3 * rng.standard_normal((nchunk, 2)).astype('float32'))
        dm2 = cp.asarray(1e-3 * rng.standard_normal((nchunk, 2)).astype('float32'))
        sh, ff = Shift(n, npsi, n, npsi, nchunk), ShiftFFT(n, npsi, n, npsi, nchunk)

        if a.mem:
            res = (c.nbytes * 3 + g.nbytes) / 2**20
            print(f'\nn={n}  npsi={npsi}  nchunk={nchunk}   '
                  f'(peak scratch MiB; inputs already resident: {res:.0f} MiB)')
        else:
            print(f'\nn={n}  npsi={npsi}  nchunk={nchunk}   (ms per call, '
                  f'{a.reps} reps)')
        print(f'{"m":>7s} {"op":>12s} {"Shift":>10s} {"ShiftFFT":>10s} {"ratio":>8s}')
        for mv in ms:
            m = cp.full((nchunk, 2), mv, dtype='float32')
            # The cascade spends most of its time in the DERIVATIVE variants,
            # not in curlyS: Rec.dF3/d2F_dF3/gF3 call these once per level per
            # distance per chunk.  On the chirp-z path several of them route
            # through _shift_spec, which is an ifft2 plus a full S.
            ops = (
                ('curlyS',       lambda o: o.curlyS(c, r, m)),
                ('Sadj',         lambda o: o.Sadj(g, r, m)),
                ('curlySback',   lambda o: o.curlySback(g, r, m)),
                ('dcurlySc',     lambda o: o.dcurlySc(c, r, m, c1, dr1)),
                ('dcurlySadjc',  lambda o: o.dcurlySadjc(c, r, m, g)),
                ('d2curlySc',    lambda o: o.d2curlySc(c, r, m, c1, dr1, c2, dr2)),
                ('dcurlySmc',    lambda o: o.dcurlySmc(c, r, m, c1, dr1, dm1)),
                ('dcurlySadjmc', lambda o: o.dcurlySadjmc(c, r, m, g)),
                ('d2curlySmc',   lambda o: o.d2curlySmc(c, r, m, c1, dr1, dm1,
                                                        c2, dr2, dm2)),
            )
            for op, fn in ops:
                if not (hasattr(sh, op) and hasattr(ff, op)):
                    continue              # older ShiftFFT lacks curlySback
                f_sh, f_ff = (lambda fn=fn: fn(sh)), (lambda fn=fn: fn(ff))
                if a.mem:
                    t1, t2 = peak_mib(f_sh), peak_mib(f_ff)
                else:
                    t1, t2 = timed(f_sh, a.reps), timed(f_ff, a.reps)
                print(f'{mv:7.4f} {op:>12s} {t1:10.2f} {t2:10.2f} {t2 / t1:7.1f}x')
                rows.append((n, npsi, nchunk, mv, op, t1, t2))
        del sh, ff, c, g, c1, c2
        cp.get_default_memory_pool().free_all_blocks()

    if a.csv:
        unit_ = 'mib' if a.mem else 'ms'
        with open(a.csv, 'w') as f:
            f.write(f'n,npsi,nchunk,m,op,shift_{unit_},shiftfft_{unit_},ratio\n')
            for n, npsi, nc, mv, op, t1, t2 in rows:
                f.write(f'{n},{npsi},{nc},{mv},{op},{t1:.4f},{t2:.4f},{t2 / t1:.3f}\n')
        print(f'\nwrote {a.csv}')

    unit = [r for r in rows if r[3] == 1.0]
    chz = [r for r in rows if r[3] != 1.0]
    for lab, sel in (('m == 1 (phase ramp)', unit), ('m != 1 (chirp-z)', chz)):
        if sel:
            q = [t2 / t1 for *_, t1, t2 in sel]
            print(f'{lab:24s} ShiftFFT/Shift  median {np.median(q):5.1f}x  '
                  f'range {min(q):.1f}-{max(q):.1f}x')
    if a.mem and rows:
        print('\nscratch is linear in nchunk, so the budget per angle is')
        for n, npsi, nc, mv, op, t1, t2 in rows:
            res = (3 * npsi * npsi + n * n) * 8 / 2**20 / nc
            print(f'  n={n} m={mv} {op:>12s}  Shift {t1 / nc + res:7.0f} '
                  f'ShiftFFT {t2 / nc + res:7.0f} MiB/angle')


if __name__ == '__main__':
    main()
