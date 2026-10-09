#!/usr/bin/env python
"""
Radiation dose delivered to the sample, per distance, for a multi-distance
holotomography scan.

    python dose.py config_steps15.conf [--adu-per-photon G] [--muen M]

WHY DOSE DEPENDS ON THE DISTANCE
--------------------------------
The beam comes out of a focus, so at sample-to-focus distance z1 the same
photons are spread over an area ~ z1^2 and the fluence the sample sees goes as
1/z1^2.  Equivalently: the counts per DETECTOR pixel are roughly the same at
every distance (that is how the scan is set up), but one detector pixel maps
back to a sample area (p/M)^2 with M = z2/z1, so the dose per unit sample area
scales as M^2, i.e. as 1/z1^2.  A projection at the far position therefore
costs a fraction of the dose of one at the near position.

Dose contributed by distance k, relative to one projection at distance 0:

    D_k = (z1_0/z1_k)^2  x  (count_time_k / count_time_0)  x  nproj_k

Same reads as show_geometry.py -- h5py / numpy / fabio only, no cupy, no MPI,
so it runs on a Polaris login node.

ABSOLUTE DOSE (optional)
------------------------
Relative numbers need no calibration.  For grays you need the detector gain:
pass --adu-per-photon G (ADU per detected X-ray photon, including scintillator
and optics efficiency -- a beamline calibration, not something in the files).
The script then measures the mean flat-field level, converts to photons per
detector pixel, projects it back to the sample plane, and applies

    D [Gy] = Phi [photons/m^2] x E [J] x (mu_en/rho) [m^2/kg]

with mu_en/rho log-log interpolated from the NIST table for liquid water below
(override with --muen, in cm^2/g).  Water stands in for soft tissue; swap the
table if the sample is something else.  Treat the absolute number as an
order-of-magnitude estimate -- the gain calibration dominates its error.
"""

import argparse
import configparser
import glob
import json
import os
import sys

import fabio
import h5py
import numpy as np

# NIST X-Ray Mass Attenuation Coefficients, liquid water: (keV, mu_en/rho cm^2/g)
MUEN_WATER = np.array([
    [10.0, 4.944], [15.0, 1.374], [20.0, 0.5503], [30.0, 0.1557],
    [40.0, 0.06947], [50.0, 0.04223], [60.0, 0.03190], [80.0, 0.02597],
    [100.0, 0.02546],
])


def muen_water(energy_kev):
    """log-log interpolation of mu_en/rho (cm^2/g) at `energy_kev`."""
    e, m = np.log(MUEN_WATER[:, 0]), np.log(MUEN_WATER[:, 1])
    return float(np.exp(np.interp(np.log(energy_kev), e, m)))


# --- the same metadata reads steps15.py / show_geometry.py do ---------------
def _read_h5_field(h5path, suffix):
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
    return float(values[names.index('sx')]) * 1e-3


def read_detector_pixelsize(p):
    par = json.loads(_read_h5_field(p, 'TOMO/FTOMO_PAR').decode())
    return float(par['image_pixel_size']) * 1e-6


def read_focustodetectordistance(p):
    return float(_read_h5_field(p, 'PTYCHO/focusToDetectorDistance')) * 1e-3


def edf_count_time(edf_path):
    """Exposure per frame (s) from the EDF header keyword `count_time`."""
    with open(edf_path, 'rb') as fh:
        head = fh.read(4096).decode('latin-1')
    for field in head.split(';'):
        if 'count_time' in field.lower():
            return float(field.split('=')[1])
    raise KeyError(f'count_time not in the EDF header of {edf_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='config_steps15.conf')
    ap.add_argument('--adu-per-photon', type=float, default=None,
                    help='detector gain; enables the absolute-dose column')
    ap.add_argument('--muen', type=float, default=None,
                    help='mu_en/rho in cm^2/g (default: NIST water at this energy)')
    a = ap.parse_args()

    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    with open(a.config, encoding='utf-8') as f:
        cfg.read_string('[DEFAULT]\n' + f.read())
    cfg = cfg['DEFAULT']
    path, pfile = cfg.get('path').rstrip('/'), cfg.get('pfile')

    dirs = sorted(glob.glob(f'{path}/{pfile}_[0-9]_/'))
    if not dirs:
        sys.exit(f'no distance directories matching {path}/{pfile}_[0-9]_/')
    h5files = [sorted(glob.glob(f'{d}/*.h5'))[0] for d in dirs]
    ndist = len(dirs)

    energy = read_energy(h5files[0])                       # keV
    f2d    = read_focustodetectordistance(h5files[0])      # m
    pdet   = read_detector_pixelsize(h5files[0])           # m
    sx0    = read_sx0(h5files[0])                          # m
    z1     = np.array([read_sx(h) for h in h5files]) - sx0 # m
    mag    = f2d / z1
    voxel  = pdet / mag

    ct, nproj, flat = [], [], []
    for d in dirs:
        edfs = [f for f in sorted(glob.glob(f'{d}/*.edf'))
                if 'ref' not in os.path.basename(f) and 'dark' not in os.path.basename(f)]
        refs = sorted(glob.glob(f'{d}/ref*.edf'))
        ct.append(edf_count_time(edfs[0]))
        nproj.append(len(edfs))
        flat.append(float(fabio.open(refs[0]).data.mean()) if refs else np.nan)
    ct, nproj, flat = np.array(ct), np.array(nproj, dtype=float), np.array(flat)

    # --- relative dose ------------------------------------------------------
    w   = (z1[0] / z1) ** 2                 # fluence per frame, rel. to distance 1
    per = w * (ct / ct[0])                  # dose per projection, rel. to distance 1
    tot = per * nproj                       # dose contributed by the whole distance
    total = tot.sum()

    print(f'{pfile}   ndist={ndist}   energy={energy:.3f} keV   '
          f'focus-to-detector={f2d*100:.3f} cm   detector pixel={pdet*1e6:.4f} um')
    print()
    hdr = (f'{"k":>2} {"z1 [mm]":>9} {"mag":>8} {"voxel [nm]":>11} {"expo [s]":>9} '
           f'{"nproj":>6} {"flat [ADU]":>11} {"1/z1^2":>8} {"D/proj":>8} '
           f'{"D_total":>9} {"share":>7}')
    print(hdr); print('-' * len(hdr))
    for k in range(ndist):
        print(f'{k+1:>2} {z1[k]*1e3:>9.4f} {mag[k]:>8.2f} {voxel[k]*1e9:>11.3f} '
              f'{ct[k]:>9.4g} {nproj[k]:>6.0f} {flat[k]:>11.1f} {w[k]:>8.4f} '
              f'{per[k]:>8.4f} {tot[k]:>9.1f} {100*tot[k]/total:>6.1f}%')
    print('-' * len(hdr))
    print(f'{"":>2} {"":>9} {"":>8} {"":>11} {"":>9} {nproj.sum():>6.0f} '
          f'{"":>11} {"":>8} {per.sum():>8.4f} {total:>9.1f} {100:>6.1f}%')
    print()
    print(f'Units: one projection at distance 1 (z1={z1[0]*1e3:.4f} mm, '
          f'{ct[0]:g} s) = 1.0.')
    print(f'The whole {ndist}-distance scan costs {total:.1f} such projections, '
          f'i.e. {per.sum():.4f} per angle')
    print(f'  -- not {ndist}, because the far distances are individually cheap.')
    print(f'A single-distance scan at distance 1 needs '
          f'{per.sum()/per[0]:.4f} x ntheta angles to match this dose.')
    if not np.allclose(nproj, nproj[0]):
        print(f'NOTE: the distances do not have equal projection counts: '
              f'{nproj.astype(int).tolist()}')
    if np.isfinite(flat).all() and flat.std() / flat.mean() > 0.15:
        print(f'NOTE: flat-field levels differ by more than 15% across distances, '
              f'so "counts per detector pixel are the same everywhere" does not '
              f'hold here -- scale the 1/z1^2 column by flat/flat[0] as well.')

    # --- absolute dose ------------------------------------------------------
    if a.adu_per_photon:
        muen = a.muen if a.muen is not None else muen_water(energy)
        muen_si = muen * 0.1                              # cm^2/g -> m^2/kg
        e_j = energy * 1e3 * 1.602176634e-19              # keV -> J
        # photons per detector pixel per frame, mapped back to the sample plane:
        # one detector pixel covers (pdet/mag)^2 of sample area.
        ph_px = flat / a.adu_per_photon
        phi   = ph_px / (pdet / mag) ** 2                 # photons / m^2 / frame
        d_frame = phi * e_j * muen_si                     # Gy per frame
        d_dist  = d_frame * nproj
        print()
        print(f'Absolute dose  (gain {a.adu_per_photon:g} ADU/photon, '
              f'mu_en/rho {muen:.4g} cm^2/g '
              f'{"(--muen)" if a.muen is not None else "(NIST water)"} at '
              f'{energy:.3f} keV)')
        h2 = (f'{"k":>2} {"ph/px/frame":>12} {"Phi [ph/m^2]":>14} '
              f'{"Gy/frame":>11} {"Gy (all proj)":>14}')
        print(h2); print('-' * len(h2))
        for k in range(ndist):
            print(f'{k+1:>2} {ph_px[k]:>12.1f} {phi[k]:>14.4e} '
                  f'{d_frame[k]:>11.4e} {d_dist[k]:>14.4e}')
        print('-' * len(h2))
        print(f'{"":>2} {"":>12} {"":>14} {"":>11} {d_dist.sum():>14.4e}')
        print(f'\nTotal absorbed dose for the whole scan: {d_dist.sum():.4g} Gy '
              f'= {d_dist.sum()/1e6:.4g} MGy.')
        print('Order of magnitude only: the ADU/photon calibration dominates.')
    else:
        print('\n(Pass --adu-per-photon G for absolute grays; without the '
              'detector gain only the relative column is defensible.)')


if __name__ == '__main__':
    main()
