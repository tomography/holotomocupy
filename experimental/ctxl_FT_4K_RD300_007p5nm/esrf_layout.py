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

THE THIRD FLAVOUR, "nxvds" (this scan).  ctxl_FT_* has BOTH 12003 projection
EDFs and the NXtomo, and the NXtomo's instrument/detector/data is a VIRTUAL
dataset over 124 raw balor HDF5 files under RAW_DATA/.  When a virtual NXtomo
is present the flavour is reported as `nxvds` and the frame readers go through
it rather than through the EDFs, which is what steps15.py step 1 and step0.py
are written against.  Everything else -- geometry, .info, shift_source, the
ref/dark naming -- behaves exactly as `ewoks`, which `nxvds` is a special case
of.  `NxFrames` below does the virtual-source rebasing; see its docstring for
why reading the VDS naively returns silent zeros.

A scan with the NXtomo but no virtual dataset stays `ewoks` and reads EDFs, so
this file also drives ../ctxl_HT_4K_RD300_007p5nm unchanged.

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

def _nx_is_virtual(nx_path):
    """True when this NXtomo's detector/data is an HDF5 virtual dataset.

    False for a plain dataset and for anything unreadable, so a damaged or
    half-copied .nx degrades to the EDF path instead of raising at import.
    """
    import h5py
    try:
        with h5py.File(nx_path, 'r') as f:
            g = f[next(k for k in f if k.startswith('entry'))]
            return bool(g['instrument/detector/data'].is_virtual)
    except (OSError, KeyError, StopIteration):
        return False


# ---------------------------------------------------------------------------
# NXtomo backed by a virtual dataset
#
# nxtomomill writes instrument/detector/data as an HDF5 virtual dataset whose
# sources are recorded RELATIVE TO THE .nx FILE and count on the full beamline
# tree (./../../../../RAW_DATA/...).  Our copy drops the PROCESSED_DATA level
# those four `..` are written against, so the recorded path resolves to
# nothing and HDF5 serves the dataset's FILL VALUE -- zeros, with no error and
# no warning -- and every downstream step runs to completion on them.  That
# silent failure is the only reason this is not one line of h5py.
#
# NxFrames takes the mapping apart, rebases each source onto the real tree
# (recorded path first, then the same tail against every ancestor of the .nx,
# nearest first) and reads the sources directly.  `missing` lists whatever
# still could not be found; callers must REFUSE TO START rather than read
# zeros.  ../AtomiumS1_FT_RD300/nx_frames.py is a standalone copy of this class.
# ---------------------------------------------------------------------------

# Source files kept open at once.  An NFP companion has two (darks + frames);
# a projection scan has one per bliss scan and can have dozens.
MAX_OPEN = 8


def nx_entry(f):
    """The single NXtomo entry group name of an open file.

    `entry_0001` for a projection scan and `entry_NFP_before_0001` for an NFP
    companion, so it cannot be hard-coded.
    """
    return next(k for k in f if k.startswith('entry'))


def _tail(src):
    """`src` with its leading `.` / `..` components dropped."""
    parts = [p for p in os.path.normpath(src).split(os.sep) if p not in ('.', '')]
    return os.sep.join(p for p in parts if p != '..')


def rebase_candidates(nxdir, src):
    """Every absolute path `_rebase` would accept for `src`, nearest first.

    [0] is the recorded path resolved as written (an untouched beamline tree
    lands there); the rest strip the leading `..` and hang the same tail off
    each ancestor of the NXtomo in turn.  A missing source has to be copied to
    one of these -- which is what makes this worth exposing: the recorded path
    is where ESRF had it, not where it goes here.
    """
    tail = _tail(src)
    out = [os.path.normpath(os.path.join(nxdir, src))]
    anc = os.path.abspath(nxdir)
    while True:
        out.append(os.path.join(anc, tail))
        parent = os.path.dirname(anc)
        if parent == anc:
            return out
        anc = parent


def _rebase(nxdir, src):
    """Absolute path of a virtual source, as copied rather than as recorded.

    Returns the recorded path if nothing matches, so the caller reports the
    path that was actually written in the file.
    """
    cands = rebase_candidates(nxdir, src)
    for cand in cands:
        if os.path.exists(cand):
            return cand
    return cands[0]


class NxFrames:
    """Frame-level reader for an NXtomo backed by a virtual dataset.

        fr = NxFrames(nx_path)
        if fr.missing: ...            # refuse; reading would return zeros
        dark = fr.frames(fr.idark).mean(axis=0)
        raw  = fr.frames(fr.iproj[a:b])
        fr.close()

    Attributes: shape (ny, nx), nframes_total, dtype, image_key, angles,
    count_time, tomo_n, iproj / iflat / idark (indices by image_key), sources
    (start, count, file, dataset) sorted by start, missing (unresolved files).
    """

    def __init__(self, nx_path):
        import h5py
        self.path = nx_path
        self.dir = os.path.dirname(os.path.abspath(nx_path))
        self._open = {}

        with h5py.File(nx_path, 'r') as f:
            g = f[nx_entry(f)]
            det = g['instrument/detector']
            self.image_key = det['image_key'][:].astype('int64')
            self.angles = g['sample/rotation_angle'][:].astype('float64')
            self.count_time = det['count_time'][:]
            self.tomo_n = int(det['tomo_n'][()])

            d = det['data']
            self.nframes_total = d.shape[0]
            self.shape = d.shape[1:]
            self.dtype = d.dtype
            if not d.is_virtual:
                raise ValueError(
                    f'{nx_path}: detector/data is not a virtual dataset; read '
                    'it with plain h5py instead of NxFrames.')

            rows = []
            self.recorded = {}
            for s in d.virtual_sources():
                start, _, _, block = s.vspace.get_regular_hyperslab()
                fn = _rebase(self.dir, s.file_name)
                self.recorded[fn] = s.file_name
                rows.append((int(start[0]), int(block[0]), fn, s.dset_name))
            rows.sort()
            self.sources = rows

        self.iproj = np.where(self.image_key == 0)[0]
        self.iflat = np.where(self.image_key == 1)[0]
        self.idark = np.where(self.image_key == 2)[0]
        self.missing = [fn for _, _, fn, _ in self.sources if not os.path.exists(fn)]

    def destinations(self, path):
        """Where a source reported in `missing` may be copied, nearest first.

        `path` is as it appears in `self.missing`.  Index 0 is the recorded
        beamline path; the later entries are the ancestors of this NXtomo, and
        those are the ones a local copy normally uses.
        """
        return rebase_candidates(self.dir, self.recorded[path])

    def suggest_destination(self, path):
        """Where to copy a missing source: the nearest candidate whose raw root
        (`RAW_DATA/`, say) already exists, so the copy joins the samples that
        are there rather than starting a second tree one level off.  Falls back
        to the outermost candidate when no root exists yet.
        """
        tail = _tail(self.recorded[path])
        top = tail.split(os.sep)[0]
        cands = self.destinations(path)
        for c in cands[1:]:
            if os.path.isdir(os.path.join(c[:-len(tail)], top)):
                return c
        return cands[-1]

    def _file(self, path):
        import h5py
        f = self._open.get(path)
        if f is None:
            if len(self._open) >= MAX_OPEN:
                self._open.popitem()[1].close()
            f = self._open[path] = h5py.File(path, 'r')
        return f

    def raw(self, gid):
        """Frame `gid` of the NXtomo's detector/data, as stored (uint16)."""
        for start, count, fname, dset in self.sources:
            if start <= gid < start + count:
                return self._file(fname)[dset][gid - start]
        raise IndexError(f'frame {gid} is outside the {self.nframes_total} '
                         f'frames of {self.path}')

    def frames(self, gids):
        """float32 [len(gids), ny, nx] for a list of global frame indices."""
        out = np.empty((len(gids),) + self.shape, dtype='float32')
        for i, gid in enumerate(gids):
            out[i] = self.raw(int(gid))
        return out

    def close(self):
        for f in self._open.values():
            f.close()
        self._open.clear()

    def __str__(self):
        s = [f'{self.path}',
             f'  {self.nframes_total} frames of {self.shape[0]}x{self.shape[1]} '
             f'{self.dtype}  (proj {len(self.iproj)}, flat {len(self.iflat)}, '
             f'dark {len(self.idark)})']
        for start, count, fname, dset in self.sources:
            mark = '' if os.path.exists(fname) else '   <-- MISSING'
            s.append(f'  [{start:6d}:{start+count:6d}] {fname}:{dset}{mark}')
        return '\n'.join(s)


if __name__ == '__main__':
    import sys
    fr = NxFrames(sys.argv[1])
    print(fr)
    for m in fr.missing:
        print(f'  copy it to: {fr.suggest_destination(m)}')
    raise SystemExit(1 if fr.missing else 0)


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
            # A virtual detector/data means the frames live in the raw balor
            # files, not in the EDFs, even when both are on disk.
            self.flavour = 'nxvds' if _nx_is_virtual(self.nxfiles[0]) else 'ewoks'
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

        self._nx = {}       # k -> NxFrames, opened on the first frame read

        # Prefer the EDF count: it is free, and where both exist the two agree
        # (20 flats per batch, 20 darks on this scan).  A pure-NXtomo drop has
        # no ref*/dark* EDFs at all, and then the image_key decides.
        self.nref = len(self.refs(0, 0))
        self.ndark = len(self.darks(0))
        if self.flavour == 'nxvds' and (self.nref == 0 or self.ndark == 0):
            fr = self.nx(0)
            self.nref = self.nref or len(self._flat_blocks(0)[0])
            self.ndark = self.ndark or len(fr.idark)

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
        if self.flavour in ('ewoks', 'nxvds'):
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
        """Rotation angle of every written PROJECTION.

        For `nxvds` this comes from the NXtomo's sample/rotation_angle taken
        at image_key == 0, so it is indexed exactly like read_proj() -- the
        flats and darks interleaved in detector/data are not in it.  The other
        flavours read angles_file.txt.
        """
        if self.flavour == 'nxvds':
            fr = self.nx(k)
            return fr.angles[fr.iproj]
        f = f'{self.dname(k)}/angles_file.txt'
        return np.loadtxt(f) if os.path.exists(f) else None

    def nproj_files(self, k=0):
        return len(glob.glob(f'{self.dname(k)}/{self.pfile}_{k + 1}_[0-9]*.edf'))

    # -- frames ------------------------------------------------------------

    def nx(self, k=0):
        """The NxFrames of distance k, opened once and cached."""
        if self.flavour != 'nxvds':
            raise RuntimeError(f'{self.pfile} is flavour {self.flavour}, '
                               'which has no NXtomo virtual dataset')
        fr = self._nx.get(k)
        if fr is None:
            fr = self._nx[k] = NxFrames(self.nxfiles[k])
            if fr.missing:
                raise SystemExit(
                    f'{self.nxfiles[k]}: {len(fr.missing)} of '
                    f'{len(fr.sources)} virtual sources do not resolve, e.g.\n'
                    f'  {fr.missing[0]}\n'
                    'reading them would return zeros with no error; copy the '
                    'raw balor files (fr.suggest_destination) before running.')
        return fr

    def _flat_blocks(self, k=0):
        """The flat batches of an NXtomo, as lists of global frame indices.

        image_key == 1 comes in contiguous runs, one per batch: [0] is the
        start-of-scan batch and [-1] the end-of-scan one.  An aborted scan has
        only the first.
        """
        iflat = self.nx(k).iflat
        if len(iflat) == 0:
            return [[]]
        cuts = np.where(np.diff(iflat) > 1)[0] + 1
        return [b.tolist() for b in np.split(iflat, cuts)]

    def frame_shape(self, k=0):
        """(ny, nx) of one detector frame, without reading a whole one."""
        if self.flavour == 'nxvds':
            return tuple(int(x) for x in self.nx(k).shape)
        import fabio
        return tuple(int(x) for x in fabio.open(self.proj(k, 0)).data.shape)

    def read_proj(self, k, j):
        """Projection j (0-based, excluding flats/darks) as float32 [ny, nx]."""
        if self.flavour == 'nxvds':
            fr = self.nx(k)
            return fr.frames([fr.iproj[j]])[0]
        import fabio
        return fabio.open(self.proj(k, j)).data.astype('float32')

    def read_refs(self, k, angle, nmax=None):
        """Flat frames of the batch at frame index `angle`, as a list.

        `angle == 0` is the start-of-scan batch and anything else the
        end-of-scan one, which is how steps15.py calls it (0 and ntheta).  An
        aborted scan returns [] for the end batch and the caller reuses the
        start batch -- the same contract as the EDF glob it replaces.
        """
        if self.flavour == 'nxvds':
            blocks = self._flat_blocks(k)
            gids = blocks[0] if angle == 0 else (blocks[-1] if len(blocks) > 1 else [])
            return list(self.nx(k).frames(gids[:nmax])) if gids else []
        import fabio
        return [fabio.open(f).data.astype('float32')
                for f in self.refs(k, angle, nmax)]

    def read_darks(self, k, nmax=None):
        """Dark frames as a list of float32 [ny, nx]."""
        if self.flavour == 'nxvds':
            fr = self.nx(k)
            gids = list(fr.idark[:nmax])
            return list(fr.frames(gids)) if gids else []
        import fabio
        return [fabio.open(f).data.astype('float32') for f in self.darks(k, nmax)]

    def close(self):
        for fr in self._nx.values():
            fr.close()
        self._nx.clear()

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
