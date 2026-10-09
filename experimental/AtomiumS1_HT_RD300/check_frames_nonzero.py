#!/usr/bin/env python
"""
Do the frames actually contain data?  One read per plane, across all scans.

    python check_frames_nonzero.py ../AtomiumS1_*/config_steps15.conf

A ONE-OFF DIAGNOSTIC, not part of the pipeline.

WHY.  preflight.py checks directory listings and .info sidecars and says so --
"no frames read".  That leaves one failure it cannot see: on the eagle copies
the <sample> level of the ESRF tree is missing, so an NXtomo virtual source
can point at nothing and read back as SILENT ZEROS rather than erroring.
Layout falls back to the EDFs when the sources do not resolve, but nothing
confirms the bytes that come out are not zero.  Six of the queued jobs would
run 18 h on an all-zero stack before anyone noticed.

Three frames per plane -- one projection at mid-scan, one flat, one dark.
Pure I/O, no transforms, which is why it is safe to run on a login node.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holotomocupy.config import parse_args_steps15
from esrf_layout import Layout


def stat(tag, a):
    """One line per frame.  'ALL ZERO' is the thing we are hunting."""
    a = np.asarray(a, dtype='float64')
    bad = '  <-- ALL ZERO' if np.ptp(a) == 0 else ''
    return (f'    {tag:<12} min {a.min():>12.2f}  mean {a.mean():>12.2f}  '
            f'max {a.max():>12.2f}  ptp {np.ptp(a):>12.2f}{bad}')


def main(confs):
    nbad = 0
    for conf in confs:
        cfg = parse_args_steps15(conf)
        lay = Layout(cfg.path, cfg.pfile)
        print(f'\n=== {os.path.basename(os.path.dirname(os.path.abspath(conf)))}'
              f'  [{lay.flavour}]  ndist={lay.ndist} ntheta={lay.ntheta}',
              flush=True)
        if lay.nx_unresolved:
            print(f'  note: {lay.nx_unresolved[0]}/{lay.nx_unresolved[1]} '
                  f'virtual sources unresolved -- reading EDFs')
        for k in range(lay.ndist):
            print(f'  plane {k + 1}:')
            for tag, fn in (('proj mid', lambda: lay.read_proj(k, lay.ntheta // 2)),
                            ('flat start', lambda: lay.read_refs(k, 0, 1)),
                            ('flat end', lambda: lay.read_refs(k, lay.ntheta, 1)),
                            ('dark', lambda: lay.read_darks(k, 1))):
                try:
                    a = fn()
                except Exception as e:
                    print(f'    {tag:<12} FAILED: {type(e).__name__}: {e}')
                    nbad += 1
                    continue
                if isinstance(a, list):
                    # read_refs returns [] for the end batch of an aborted
                    # scan; steps15 handles that by reusing the start batch.
                    if not a:
                        print(f'    {tag:<12} (empty batch -- start batch reused)')
                        continue
                    a = a[0]
                line = stat(tag, a)
                print(line, flush=True)
                if 'ALL ZERO' in line:
                    nbad += 1
    print(f'\n{"FAIL" if nbad else "OK"}: {nbad} zero/unreadable frame(s)')
    return 1 if nbad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
