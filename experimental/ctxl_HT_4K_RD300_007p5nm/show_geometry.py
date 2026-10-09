#!/usr/bin/env python
"""
Print the scan geometry steps15.py auto-detects, without launching an MPI job.

    python show_geometry.py config_steps15.conf [bin]

With no [bin] argument it prints one block per hierarchical level, coarsest
first, for every level config_steps15.conf asks step 5 to build
(range(start_level_rec, nlevels)).

steps15.py derives ndist, ntheta, energy, distances, magnifications, n and nobj
from the raw data at startup and logs them -- but only once a job is running.
This does the same reads on the login node so the config_step6_bin*.conf files
can be filled in first, with every size already divided by 2**bin.

All the filename and geometry knowledge lives in esrf_layout.py, which handles
both the 2025 "bliss" and the 2026 "ewoks" directory conventions; see the
docstring there.

Deliberately standalone: only h5py / numpy / fabio, no cupy and no MPI, so it
runs on a Polaris login node where there is no GPU.
"""

import configparser
import sys

import fabio
import numpy as np

from esrf_layout import Layout

# re-exported for scan_overview.py, which used to import these from here
from esrf_layout import (read_energy, read_sx0, read_sx,                # noqa: F401
                         read_detector_pixelsize,                        # noqa: F401
                         read_focustodetectordistance)                   # noqa: F401


def read_config(config_file):
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    with open(config_file, encoding='utf-8') as f:
        cfg.read_string('[DEFAULT]\n' + f.read())
    return cfg['DEFAULT']


def main(config_file, bin_level):
    cfg = read_config(config_file)

    lay = Layout(cfg.get('path').rstrip('/'), cfg.get('pfile'))
    geo = lay.geometry()

    n_override = cfg.getint('n', fallback=0)
    nobj_override = cfg.getint('nobj', fallback=0)

    norm_magnifications = geo['norm_magnifications']
    n0, n1 = fabio.open(lay.refs(0, 0)[0]).data.shape
    n = n_override if n_override > 0 else n0
    nobj = (nobj_override if nobj_override > 0
            else int(np.ceil(n / norm_magnifications[-1] / 64)) * 64)

    print(f'path                    = {lay.path}')
    print(f'pfile                   = {lay.pfile}')
    print(f'layout flavour          = {lay.flavour}')
    print(f'distance dirs           = {[d.split("/")[-1] for d in lay.dirs]}')
    print(f'ndist                   = {lay.ndist}')
    print(f'ntheta                  = {lay.ntheta}')
    print(f'energy                  = {geo["energy"]} keV')
    print(f'detector_pixelsize      = {geo["detector_pixelsize"]} m')
    print(f'focustodetectordistance = {geo["focustodetectordistance"]} m')
    print(f'sx0                     = {geo["sx0"]} m')
    print(f'z1                      = {geo["z1"]} m')
    print(f'z2                      = {geo["z2"]} m')
    print(f'magnifications          = {geo["magnifications"]}')
    print(f'norm_magnifications     = {norm_magnifications}')
    print(f'distances               = {geo["distances"]} m')
    print(f'voxelsizes              = {geo["voxelsizes"] * 1e9} nm')
    print(f'EDF frame               = {n0} x {n1}')
    print(f'n                       = {n}   ({"config override" if n_override > 0 else "auto"})')
    print(f'nobj                    = {nobj} ({"config override" if nobj_override > 0 else "auto"})')
    print(f'nref                    = {lay.nref}')
    print(f'ndark                   = {lay.ndark}')

    # The .info sidecars are written by a different piece of beamline software
    # than the geometry source above, so their PixelSize is an independent
    # check that z1 has been read with the right sign and units.
    bad = lay.info_check(geo)
    print(f'.info cross-check       = ' +
          ('PixelSize agrees with the derived voxel sizes' if not bad
           else 'MISMATCH\n    ' + '\n    '.join(bad)))

    # What nobj has to hold is the SAMPLE plus the whole displacement sweep.
    # The sample is what plane 1 sees, n px at plane-1 voxels; the commanded
    # random shifts are in each plane's own detector pixels and become
    # 1/norm_mag[k] times larger in the object frame, worst at the last plane.
    # (The object-plane footprint of the demagnified planes, n/norm_mag[-1], is
    # larger still and deliberately NOT covered -- step 4 clamps its pad offsets
    # there, outside the sample, where the clamp costs nothing.  See the nobj
    # note in config_steps15.conf.)
    print(f'\nobject-frame footprint of plane 1        = {n} px')
    print(f'object-frame footprint of plane {lay.ndist}        = '
          f'{n / norm_magnifications[-1]:.0f} px  (not covered by nobj; clamped)')
    try:
        r = np.array([np.loadtxt(lay.shift_source(k), dtype='float32')[:lay.ntheta]
                      for k in range(lay.ndist)])           # (ndist, ntheta, 2)
        rmax = np.abs(r / norm_magnifications[:, None, None]).max()
        print(f'max |random shift|                      = {rmax:.1f} object px  '
              f'(commanded max {np.abs(r).max():.1f} detector px)')
        print(f'nobj should be >= {int(np.ceil((n + 2 * rmax) / 64)) * 64} '
              f'(= ceil64({n} + 2*{rmax:.0f}));  config says {nobj}')
    except OSError as e:
        print(f'(random shifts unavailable: {e})')

    # --- per-level blocks for the hierarchical ladder ----------------------
    # Step 5 writes an initial volume for every bin in
    # range(start_level_rec, nlevels); each of those levels has one
    # config_step6_bin{bin}.conf.  nzobj is set equal to nobj (the FBP volume
    # step 5 writes is nobj^3); reduce it to reconstruct a slab.
    nlevels         = cfg.getint('nlevels', fallback=3)
    start_level_rec = cfg.getint('start_level_rec', fallback=0)
    levels = ([bin_level] if bin_level is not None
              else list(range(nlevels - 1, start_level_rec - 1, -1)))

    for lvl in levels:
        b = 2**lvl
        print(f'\n--- config_step6_bin{lvl}.conf block (bin={lvl}, {b}x{b}) ---')
        print(f'ntheta={lay.ntheta}')
        print(f'ndist={lay.ndist}')
        print(f'nz={n // b}')
        print(f'n={n // b}')
        print(f'nzobj={nobj // b}')
        print(f'nobj={nobj // b}')
        print(f'bin={lvl}')
        if n % b or nobj % b:
            print(f'WARNING: n={n} or nobj={nobj} is not divisible by 2**{lvl}; '
                  f'the division above truncates.')

    print(f'\nnzobj is set equal to nobj (the step-5 FBP volume is nobj^3). '
          f'Reduce it to reconstruct a slab.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
