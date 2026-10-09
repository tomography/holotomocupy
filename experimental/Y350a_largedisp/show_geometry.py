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

Deliberately standalone: only h5py / numpy / fabio, no cupy and no MPI, so it
runs on a Polaris login node where there is no GPU.
"""

import configparser
import glob
import json
import sys

import fabio
import h5py
import numpy as np


def _read_h5_field(h5path, suffix):
    """Return the value of the first dataset whose path ends with `suffix`."""
    result = {}

    def _visit(name, obj):
        if not result and isinstance(obj, h5py.Dataset) and name.endswith(suffix):
            result['val'] = obj[()]

    with h5py.File(h5path, 'r') as f:
        f.visititems(_visit)
    if not result:
        raise KeyError(f'{suffix!r} not found in {h5path}')
    return result['val']


def read_energy(p):
    return float(_read_h5_field(p, 'TOMO/energy'))


def read_sx0(p):
    return float(_read_h5_field(p, 'TOMO/sx0')) * 1e-3


def read_sx(p):
    names = _read_h5_field(p, 'sample/positioners/name').decode().split()
    values = _read_h5_field(p, 'sample/positioners/value').decode().split()
    if 'sx' not in names:
        raise ValueError(f"'sx' not found in positioners for {p}\nAvailable: {names}")
    return float(values[names.index('sx')]) * 1e-3


def read_detector_pixelsize(p):
    par = json.loads(_read_h5_field(p, 'TOMO/FTOMO_PAR').decode())
    return float(par['image_pixel_size']) * 1e-6


def read_focustodetectordistance(p):
    return float(_read_h5_field(p, 'PTYCHO/focusToDetectorDistance')) * 1e-3


def main(config_file, bin_level):
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    with open(config_file, encoding='utf-8') as f:
        cfg.read_string('[DEFAULT]\n' + f.read())
    cfg = cfg['DEFAULT']

    path = cfg.get('path').rstrip('/')
    pfile = cfg.get('pfile')
    n_override = cfg.getint('n', fallback=0)
    nobj_override = cfg.getint('nobj', fallback=0)

    dirs = sorted(glob.glob(f'{path}/{pfile}_[0-9]_/'))
    if not dirs:
        sys.exit(f'no distance directories match {path}/{pfile}_[0-9]_/\n'
                 f'check `path` and `pfile`; try: ls -d {path}/{pfile}*')
    h5files = [sorted(glob.glob(f'{d}/*.h5'))[0] for d in dirs]
    ndist = len(h5files)

    dname0 = f'{path}/{pfile}_1_'
    ntheta = max(int(f.split('_')[-1].split('.')[0])
                 for f in glob.glob(f'{dname0}/ref0000_*.edf'))

    energy = read_energy(h5files[0])
    detector_pixelsize = read_detector_pixelsize(h5files[0])
    focustodetectordistance = read_focustodetectordistance(h5files[0])
    sx0 = read_sx0(h5files[0])
    z1 = np.array([read_sx(f) for f in h5files]) - sx0

    z2 = focustodetectordistance - z1
    magnifications = focustodetectordistance / z1
    norm_magnifications = magnifications / magnifications[0]
    distances = (z1 * z2) / focustodetectordistance * norm_magnifications**2
    voxelsizes = np.abs(detector_pixelsize / magnifications)

    n0, n1 = fabio.open(f'{dname0}/ref0000_0000.edf').data.shape
    n = n_override if n_override > 0 else n0
    nref = len(glob.glob(f'{dname0}/ref[0-9]*_0000.edf'))
    ndark = len(glob.glob(f'{dname0}/darkend[0-9]*.edf'))
    nobj = (nobj_override if nobj_override > 0
            else int(np.ceil(n / norm_magnifications[-1] / 64)) * 64)

    print(f'path                    = {path}')
    print(f'pfile                   = {pfile}')
    print(f'distance dirs           = {[d.rstrip("/").split("/")[-1] for d in dirs]}')
    print(f'ndist                   = {ndist}')
    print(f'ntheta                  = {ntheta}')
    print(f'energy                  = {energy} keV')
    print(f'detector_pixelsize      = {detector_pixelsize} m')
    print(f'focustodetectordistance = {focustodetectordistance} m')
    print(f'sx0                     = {sx0} m')
    print(f'z1                      = {z1} m')
    print(f'magnifications          = {magnifications}')
    print(f'norm_magnifications     = {norm_magnifications}')
    print(f'distances               = {distances} m')
    print(f'voxelsizes              = {voxelsizes * 1e9} nm')
    print(f'EDF frame               = {n0} x {n1}')
    print(f'n                       = {n}   ({"config override" if n_override > 0 else "auto"})')
    print(f'nobj                    = {nobj} ({"config override" if nobj_override > 0 else "auto"})')
    print(f'nref                    = {nref}')
    print(f'ndark                   = {ndark}')

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
        print(f'ntheta={ntheta}')
        print(f'ndist={ndist}')
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
