"""Reader over the per-tile files gen_data.py writes, presented as one scan.

    {path_out}/{pfile}_{tile}.h5    one per tile: its own energy / z1 /
                                    detector_pixelsize / theta / cshifts_final /
                                    pref_{bin} / pdata0..{ndist_tile-1}_{bin}
    {path_out}/{pfile}_obj.h5       the shared initial object
    {path_out}/{pfile}_prb.h5       the probe, ndist entries, tile-major

There is no metadata file: every tile file is self-contained, so the mosaic is
exactly the set of tile files and nothing else -- the same arrangement as the
real YY037A scan, where step6's MosaicReader also takes its metadata from the
tile files and uses ``mosaic_file`` only as the name stem for ``_obj.h5``.

The tiles are flattened onto the distance axis, tile-major: entry
``t*ndist_tile + k`` is tile ``t``, distance ``k``.  rec_mpi.Rec derives
magnifications, propagation distances and voxel size from ``z1`` alone, so
tiling ``z1`` across the tiles is all the solver needs.

Each tile's place on the mosaic is already inside its ``cshifts_final`` (that
is how gen_data.py writes it), so read_pos needs no separate tile_offsets term.
The offsets are kept as an attribute for diagnostics only.
"""

import numpy as np
import cupy as cp
import h5py

from holotomocupy.reader import Reader


class MosaicReader(Reader):
    def __init__(self, tile_files, mosaic_file, comm,
                 st_obj, end_obj, nzobj, nobj,
                 st_theta, end_theta, ntheta,
                 ndist_tile, nz, n,
                 paganin, rotation_center_shift, start_theta, bin,
                 tiles=None):
        # Scalars, the angle list and z1 of one tile come from the first file;
        # _check_tiles then verifies the rest match rather than assuming it.
        super().__init__(tile_files[0], comm,
                         st_obj, end_obj, nzobj, nobj,
                         st_theta, end_theta, ntheta,
                         ndist_tile, nz, n,
                         paganin, rotation_center_shift, start_theta, bin)

        self.tile_files  = list(tile_files)
        self.ntiles      = len(self.tile_files)
        self.ndist_tile  = ndist_tile
        self.mosaic_file = mosaic_file
        self.tiles       = ([str(t) for t in tiles] if tiles
                            else [str(i) for i in range(self.ntiles)])
        if len(self.tiles) != self.ntiles:
            raise ValueError(f'{len(self.tiles)} tile names for '
                             f'{self.ntiles} tile files')

        z1_tile     = np.asarray(self.z1, dtype='float32')
        self._check_tiles(z1_tile)
        self.z1     = np.tile(z1_tile, self.ntiles)
        self.ndist  = self.ntiles * ndist_tile

        # Diagnostics only; the placement is already inside cshifts_final.
        self.tile_offsets = np.zeros((self.ntiles, 2), dtype='float32')
        for t, path in enumerate(self.tile_files):
            with h5py.File(path, 'r') as fid:
                off = fid['/exchange'].attrs.get('tile_offset')
            if off is not None:
                self.tile_offsets[t] = np.asarray(off, dtype='float32')

    def _check_tiles(self, z1_tile):
        """Fail loudly if the tiles do not share the same acquisition geometry."""
        for path in self.tile_files[1:]:
            with h5py.File(path, 'r', driver="mpio", comm=self.comm) as fid:
                z1_t = fid['/exchange/z1'][:self.ndist_tile]
                dpx  = float(fid['/exchange/detector_pixelsize'][0]) * 2**self.bin
                fdd  = float(fid['/exchange/focusdetectordistance'][0])
                nth  = len(fid['/exchange/theta'])
            if len(z1_t) != self.ndist_tile:
                raise ValueError(f'{path}: /exchange/z1 has {len(z1_t)} entries, '
                                 f'need ndist_tile={self.ndist_tile}')
            if not np.allclose(z1_t, z1_tile, rtol=0, atol=1e-9):
                raise ValueError(f'{path}: z1 differs from {self.tile_files[0]}')
            if not np.isclose(dpx, self.detector_pixelsize, rtol=1e-9, atol=0):
                raise ValueError(f'{path}: detector_pixelsize {dpx} differs from '
                                 f'tile 0 {self.detector_pixelsize}')
            if not np.isclose(fdd, self.focustodetectordistance, rtol=0, atol=1e-9):
                raise ValueError(f'{path}: focustodetectordistance {fdd} differs '
                                 f'from tile 0 {self.focustodetectordistance}')
            if nth < self.ids[-1] + 1:
                raise ValueError(f'{path}: only {nth} angles, but angle index '
                                 f'{self.ids[-1]} is requested')

    # ------------------------------------------------------------------ reads

    def read_obj(self, out=None):
        """The initial object is shared, so it comes from mosaic_file's _obj.h5."""
        saved, self.in_file = self.in_file, self.mosaic_file
        try:
            return super().read_obj(out=out)
        finally:
            self.in_file = saved

    def read_pos(self, out=None):
        """[ndist, local_ntheta, 2], each tile's cshifts_final in its own file."""
        nl  = self.end_theta - self.st_theta
        nd  = self.ndist_tile
        ids = self.ids[self.st_theta:self.end_theta]
        raw = np.empty([self.ndist, nl, 2], dtype='float32')
        for t, path in enumerate(self.tile_files):
            with h5py.File(path, 'r', driver="mpio", comm=self.comm) as fid:
                cs = fid['/exchange/cshifts_final'][ids, :nd].astype('float32')
            raw[t * nd:(t + 1) * nd] = cs.transpose(1, 0, 2)
        if out is None:
            out = raw
        else:
            out[:] = cp.array(raw) if isinstance(out, cp.ndarray) else raw

        scale = np.float32(1.0 / 2**self.bin)
        out *= scale
        out[..., 1] += np.float32(self.rotation_center_shift * scale)
        return out

    def read_ref(self, out=None):
        """[ndist, nz, n] flat field, read on rank 0 and broadcast."""
        nz, n  = self.nz, self.n
        nd     = self.ndist_tile
        raw_np = np.empty((self.ndist, nz, n), dtype='float32')
        if self.rank == 0:
            key     = f'/exchange/pref_{self.bin}'
            key_end = f'/exchange/pref_end_{self.bin}'
            for t, path in enumerate(self.tile_files):
                sl = slice(t * nd, (t + 1) * nd)
                with h5py.File(path, 'r') as fid:
                    nz0 = fid[key].shape[1]
                    st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                    raw_np[sl] = fid[key][:nd, st:end]
                    if key_end in fid:
                        raw_np[sl] = 0.5 * (raw_np[sl] + fid[key_end][:nd, st:end])
        self.comm.Bcast(raw_np, root=0)
        raw = cp.array(raw_np)
        if out is None:
            out = cp.sqrt(raw)
        else:
            cp.sqrt(raw, out=out)
        return out

    def read_data(self, out=None):
        """[ndist, local_ntheta, nz, n], pdata{k} of tile t at index t*ndist_tile+k.

        Measured INTENSITY, matching holotomocupy.reader.Reader.read_data: Rec.F0
        is an intensity misfit, so no sqrt is taken here."""
        nz, n = self.nz, self.n
        nl    = self.end_theta - self.st_theta
        nd    = self.ndist_tile
        if out is None:
            out = np.empty([self.ndist, nl, nz, n], dtype='float32')
        # Batch reads to stay under 2^31 bytes (MPI-IO uses int for transfer sizes)
        batch = max(1, (1 << 28) // (nz * n))
        for t, path in enumerate(self.tile_files):
            with h5py.File(path, 'r', driver="mpio", comm=self.comm) as fid:
                for kk in range(nd):
                    k   = t * nd + kk
                    ds  = fid[f'/exchange/pdata{kk}_{self.bin}']
                    nz0 = ds.shape[1]
                    st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                    for i0 in range(0, nl, batch):
                        i1 = min(i0 + batch, nl)
                        out[k, i0:i1] = ds[self.ids[self.st_theta + i0:
                                                    self.st_theta + i1], st:end]
        return out
