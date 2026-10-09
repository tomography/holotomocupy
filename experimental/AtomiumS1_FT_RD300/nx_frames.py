#!/usr/bin/env python
"""
Read detector frames out of an ESRF 2026 NXtomo whose data is a VIRTUAL dataset.

WHY THIS EXISTS.  nxtomomill writes `instrument/detector/data` as an HDF5
virtual dataset whose sources are stored RELATIVE TO THE .nx FILE and count on
the full beamline tree:

    ./../../../../RAW_DATA/<sample>/<pfile>/scan000k/balor_0000.h5

Our copy of the 2026 tree drops the PROCESSED_DATA level those four `..` are
written against, so the recorded path lands one directory too high and resolves
to nothing.  HDF5 reports that as the dataset's FILL VALUE -- zeros, with no
error and no warning at all -- so `f[entry + '/instrument/detector/data'][ids]`
returns a perfectly shaped array of zeros and every downstream step runs to
completion on it.  That silent failure is the only reason this module is not
one line of h5py.

WHAT IT DOES.  Takes the virtual mapping apart (each source covers a contiguous
block of output frames), rebases each source file onto the real tree -- the
recorded path first, so an untouched beamline tree keeps working, then the same
tail against every ancestor of the .nx, nearest first -- and reads through the
source files directly.  `missing` lists any source that still could not be
found, and callers are expected to REFUSE TO START rather than read zeros.

This is ../ctxl_FT_4K_RD300_007p5nm/esrf_layout.py's NxFrames, kept as a
module of its own because this folder's scripts read EDF through
esrf_layout.Layout and need none of the rest of it.

Deliberately standalone: numpy / h5py, no cupy and no MPI.
"""

import os

import numpy as np

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
