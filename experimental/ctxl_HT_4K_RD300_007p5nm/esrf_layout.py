#!/usr/bin/env python
"""
One place that knows how an ID16A scan directory is laid out on disk.

Every other standalone script in this folder (show_geometry, scan_overview,
estimate_center / _motion / _shrink) and steps15.py itself go through this
module instead of globbing filenames of their own, because ESRF changed the
layout between the 2025 and 2026 beamtimes and the two are not compatible.

    2025 ("bliss", e.g. ../Y350a_largedisp_006nm)   2026 ("ewoks", this scan)
    -------------------------------------------    --------------------------
    <pfile>_k_/ls3231-...-<pfile>_k_.h5             (no HDF5 in the scan dir;
      -> TOMO/energy, TOMO/sx0, sample/positioners,   geometry comes from the
         TOMO/FTOMO_PAR, PTYCHO/focusToDetector...    NXtomo written next door)
    <pfile>_k_/darkend0000.edf ...                  <pfile>_k_/dark0000.edf ...
    <pfile>_k_/ref{BATCH}_{ANGLE}.edf               <pfile>_k_/ref{ANGLE}_{BATCH}.edf
    <pfile>_k_/correct.txt                          <pfile>/projections/<pfile>_000k.txt
    EDF header: motor_mne = dummy somega sx sy ...  EDF header: motor_mne = somega

The ref naming is the nastiest of these: both flavours produce files called
ref<4 digits>_<4 digits>.edf and the two fields are simply swapped, so a glob
that is right for one silently returns the wrong count for the other rather
than failing.  `ntheta` used to be recovered as the largest suffix of
ref0000_*.edf; under the 2026 naming that glob enumerates the 20 flat frames of
the theta=0 batch and returns 19.

GEOMETRY.  For the 2026 flavour it is read from the NXtomo file that
nxtomomill writes per distance,

    <path>/<pfile>/projections/<pfile>_000k.nx

which carries the same numbers the 2025 HDF5 did:
    instrument/beam/incident_energy        keV
    instrument/detector/x_pixel_size       um   (physical detector pixel)
    instrument/source/distance             mm   (negative; focus -> sample)
    instrument/detector/distance           mm   (sample -> detector)
so z1 = -source.distance and the focus-to-detector distance is
detector.distance - source.distance, constant across the four planes (it is,
to the last digit: 1212.9965 mm).  This is exactly what
holotomocupy.reader.read_nxtomo_meta already does; that function is not used
here only because these scripts must stay importable without cupy.

The <pfile>_k_/<pfile>_k_.info sidecar carries the same geometry in a third
convention (Distance = sample-to-detector in 2026, but focus-to-sample in
2025) and is used only as a cross-check: info_check() compares its PixelSize
against the voxel size derived from the NXtomo and complains if they disagree.

Deliberately standalone: numpy / h5py / fabio, no cupy and no MPI.
"""

import glob
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# .info sidecar
# ---------------------------------------------------------------------------

def read_info(path):
    """Parse an ESRF <prefix>.info sidecar into a dict of strings."""
    out = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# 2025 flavour: geometry out of the per-distance bliss HDF5
# ---------------------------------------------------------------------------

def _h5_field(h5path, suffix):
    """Value of the first dataset whose path ends with `suffix`."""
    import h5py
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
    return float(_h5_field(p, 'TOMO/energy'))


def read_sx0(p):
    return float(_h5_field(p, 'TOMO/sx0')) * 1e-3


def read_sx(p):
    names = _h5_field(p, 'sample/positioners/name').decode().split()
    values = _h5_field(p, 'sample/positioners/value').decode().split()
    if 'sx' not in names:
        raise ValueError(f"'sx' not found in positioners for {p}\nAvailable: {names}")
    return float(values[names.index('sx')]) * 1e-3


def read_detector_pixelsize(p):
    par = json.loads(_h5_field(p, 'TOMO/FTOMO_PAR').decode())
    return float(par['image_pixel_size']) * 1e-6


def read_focustodetectordistance(p):
    return float(_h5_field(p, 'PTYCHO/focusToDetectorDistance')) * 1e-3


# ---------------------------------------------------------------------------
# 2026 flavour: geometry out of the NXtomo written by nxtomomill
# ---------------------------------------------------------------------------

def read_nx_geometry(nx_path):
    """(energy keV, detector pixel m, z1 m, focus-to-detector m) from an NXtomo."""
    import h5py
    with h5py.File(nx_path, 'r') as f:
        g = f[next(k for k in f if k.startswith('entry'))]
        energy = float(g['instrument/beam/incident_energy'][()])
        det_px = float(g['instrument/detector/x_pixel_size'][()]) * 1e-6
        src    = float(g['instrument/source/distance'][()])          # mm, negative
        det    = float(g['instrument/detector/distance'][()])        # mm
    return energy, det_px, -src * 1e-3, (det - src) * 1e-3


def read_nx_translations(nx_path):
    """(x_translation, y_translation) in um for the projection frames only.

    These are the ~spy / ~spz stage readings the 2025 EDF headers carried and
    the 2026 ones do not.  Only image_key == 0 rows are returned, so the
    indexing matches the projection numbering (0 .. ntheta+2).
    """
    import h5py
    with h5py.File(nx_path, 'r') as f:
        g = f[next(k for k in f if k.startswith('entry'))]
        ids = np.where(g['instrument/detector/image_key'][:] == 0)[0]
        x = g['sample/x_translation'][:][ids].astype('float64') * 1e3   # mm -> um
        y = g['sample/y_translation'][:][ids].astype('float64') * 1e3
    return x, y


# ---------------------------------------------------------------------------

class Layout:
    """Filenames and geometry of one multi-distance ID16A scan.

    Distance planes are numbered k = 0 .. ndist-1 here and 1 .. ndist on disk.
    """

    def __init__(self, path, pfile):
        self.path = path.rstrip('/')
        self.pfile = pfile
        self.dirs = [d.rstrip('/') for d in
                     sorted(glob.glob(f'{self.path}/{pfile}_[0-9]_/'))]
        if not self.dirs:
            raise SystemExit(
                f'no distance directories match {self.path}/{pfile}_[0-9]_/\n'
                f'check `path` and `pfile`; try: ls -d {self.path}/{pfile}*')
        self.ndist = len(self.dirs)

        self.h5files = [(sorted(glob.glob(f'{d}/*.h5')) or [None])[0]
                        for d in self.dirs]
        self.nxfiles = [f'{self.path}/{pfile}/projections/{pfile}_{k+1:04d}.nx'
                        for k in range(self.ndist)]
        if self.h5files[0] is not None:
            self.flavour = 'bliss'
        elif os.path.exists(self.nxfiles[0]):
            self.flavour = 'ewoks'
        else:
            raise SystemExit(
                f'{self.dirs[0]} has no *.h5 and {self.nxfiles[0]} does not '
                f'exist -- cannot find the scan geometry for {pfile}')

        # ntheta: TOMO_N in the .info is authoritative in both flavours and
        # needs no filename archaeology.  The old ref0000_*.edf trick is kept
        # only as a fallback for a directory with no sidecar.
        info = self.info(0)
        if 'TOMO_N' in info:
            self.ntheta = int(float(info['TOMO_N']))
        else:
            self.ntheta = max(int(f.split('_')[-1].split('.')[0])
                              for f in glob.glob(f'{self.dirs[0]}/ref0000_*.edf'))

        self.nref = len(self.refs(0, 0))
        self.ndark = len(self.darks(0))

    # -- filenames ---------------------------------------------------------

    def dname(self, k=0):
        return f'{self.path}/{self.pfile}_{k + 1}_'

    def proj(self, k, j):
        return f'{self.dname(k)}/{self.pfile}_{k + 1}_{j:04d}.edf'

    def ref(self, k, batch, angle):
        if self.flavour == 'bliss':
            return f'{self.dname(k)}/ref{batch:04d}_{angle:04d}.edf'
        return f'{self.dname(k)}/ref{angle:04d}_{batch:04d}.edf'

    def refs(self, k, angle, nmax=None):
        """Sorted flat frames of the batch taken at frame index `angle`."""
        pat = (f'{self.dname(k)}/ref[0-9]*_{angle:04d}.edf' if self.flavour == 'bliss'
               else f'{self.dname(k)}/ref{angle:04d}_[0-9]*.edf')
        return sorted(glob.glob(pat))[:nmax]

    def darks(self, k, nmax=None):
        # `dark[0-9]*` also matches darkend0000.edf, so one pattern covers both
        # flavours; the plain dark.edf average is excluded by the digit.
        return sorted(glob.glob(f'{self.dname(k)}/dark[0-9]*.edf'))[:nmax]

    def info(self, k):
        return read_info(f'{self.dname(k)}/{self.pfile}_{k + 1}_.info')

    def exposure(self, k=0):
        """(count time s, latency s) per frame, or (None, None).

        Count_time / Latency_time are in the 2025 .info sidecars but not the
        2026 ones; NXtomo carries count_time per frame instead, in seconds.
        """
        info = self.info(k)
        if 'Count_time' in info:
            return (float(info['Count_time']),
                    float(info.get('Latency_time', 0.0)))
        if self.flavour == 'ewoks':
            import h5py
            try:
                with h5py.File(self.nxfiles[k], 'r') as f:
                    g = f[next(x for x in f if x.startswith('entry'))]
                    ct = g['instrument/detector/count_time'][:]
                return float(np.median(ct)), None
            except (OSError, KeyError, StopIteration):
                pass
        return None, None

    def angles(self, k=0):
        """Rotation angle of every written frame, from angles_file.txt."""
        f = f'{self.dname(k)}/angles_file.txt'
        return np.loadtxt(f) if os.path.exists(f) else None

    def nproj_files(self, k=0):
        return len(glob.glob(f'{self.dname(k)}/{self.pfile}_{k + 1}_[0-9]*.edf'))

    def shift_source(self, k):
        """Where this flavour keeps the commanded random displacement.

        2025 writes it into the distance directory as correct.txt; 2026 leaves
        it beside the NXtomo as <pfile>_000k.txt.  prepare_shifts.py copies the
        2026 file into the distance directory so step 3 of steps15.py -- which
        only ever looks for correct.txt -- finds it in the usual place.
        """
        if self.flavour == 'bliss':
            return f'{self.dname(k)}/correct.txt'
        return f'{self.path}/{self.pfile}/projections/{self.pfile}_{k + 1:04d}.txt'

    # -- geometry ----------------------------------------------------------

    def geometry(self):
        """dict(energy keV, detector_pixelsize m, focustodetectordistance m,
        z1 [ndist] m) plus everything derived from them."""
        if self.flavour == 'bliss':
            f0 = self.h5files[0]
            energy = read_energy(f0)
            det_px = read_detector_pixelsize(f0)
            f2d    = read_focustodetectordistance(f0)
            sx0    = read_sx0(f0)
            z1     = np.array([read_sx(f) for f in self.h5files]) - sx0
        else:
            per = [read_nx_geometry(f) for f in self.nxfiles]
            energy = per[0][0]
            det_px = per[0][1]
            z1     = np.array([p[2] for p in per])
            f2d_k  = np.array([p[3] for p in per])
            # focus-to-detector is one number; the four planes agree to <1 um
            # because only the sample moved.  Averaging documents that instead
            # of silently trusting plane 1.
            if np.ptp(f2d_k) > 1e-5:
                raise SystemExit(
                    f'focus-to-detector distance is not constant across the '
                    f'{self.ndist} planes: {f2d_k} m -- the NXtomo geometry '
                    f'does not describe a fixed detector')
            f2d = float(f2d_k.mean())
            sx0 = 0.0

        z2                  = f2d - z1
        magnifications      = f2d / z1
        norm_magnifications = magnifications / magnifications[0]
        distances           = (z1 * z2) / f2d * norm_magnifications**2
        voxelsizes          = np.abs(det_px / magnifications)
        return dict(energy=energy, detector_pixelsize=det_px,
                    focustodetectordistance=f2d, sx0=sx0, z1=z1, z2=z2,
                    magnifications=magnifications,
                    norm_magnifications=norm_magnifications,
                    distances=distances, voxelsizes=voxelsizes,
                    voxelsize=voxelsizes[0])

    def info_check(self, geo, tol=0.02):
        """Compare derived voxel sizes with PixelSize in each .info sidecar.

        Returns a list of human-readable lines, empty when everything agrees.
        The sidecar is written by a different part of the beamline software
        than the NXtomo, so agreement is a real independent check that the
        source/detector distances have been read with the right sign.
        """
        msgs = []
        for k in range(self.ndist):
            info = self.info(k)
            if 'PixelSize' not in info:
                continue
            want = float(info['PixelSize']) * 1e-6
            got = geo['voxelsizes'][k]
            if abs(got - want) > tol * 1e-6 * max(1.0, want * 1e6):
                msgs.append(f'plane {k + 1}: .info PixelSize {want * 1e9:.4f} nm '
                            f'vs derived {got * 1e9:.4f} nm')
        return msgs

    # -- EDF headers -------------------------------------------------------

    def motors(self, k, j):
        """{motor: value} from the EDF header of projection j at distance k."""
        import fabio
        h = fabio.open(self.proj(k, j)).header
        names = h['motor_mne'].split()
        vals = [float(x) for x in h['motor_pos'].split()]
        return dict(zip(names, vals))

    def omega(self, k, j):
        return self.motors(k, j)['somega']

    def __str__(self):
        return (f'{self.pfile}  flavour={self.flavour}  ndist={self.ndist}  '
                f'ntheta={self.ntheta}  nref={self.nref}  ndark={self.ndark}')


def from_config(cfg, path=None, pfile=None):
    """Layout for a parsed config_steps15.conf section, with CLI overrides."""
    return Layout((path or cfg.get('path')).rstrip('/'),
                  pfile or cfg.get('pfile'))
