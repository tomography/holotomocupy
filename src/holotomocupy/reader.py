import glob
import math
import os
import h5py
import numpy as np
import cupy as cp
from .logger_config import logger


def load_octave_text_mat(fpath, varname):
    """Parse Octave/MATLAB text-format .mat file and return named variable as ndarray."""
    with open(fpath, 'r') as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == f'# name: {varname}':
            i += 1
            meta = {}
            while i < len(lines) and lines[i].startswith('#'):
                parts = lines[i][1:].strip().split(':', 1)
                if len(parts) == 2:
                    meta[parts[0].strip()] = parts[1].strip()
                i += 1
            if 'ndims' in meta:
                shape = tuple(int(x) for x in lines[i].split())
                i += 1
                n = 1
                for s in shape:
                    n *= s
                vals = []
                while len(vals) < n:
                    vals.extend(lines[i].split()); i += 1
                return np.array(vals, dtype='float64').reshape(shape, order='F')
            else:
                rows = int(meta.get('rows', 1))
                cols = int(meta.get('columns', 1))
                vals = []
                for _ in range(rows):
                    vals.extend(lines[i].split()); i += 1
                return np.array(vals, dtype='float64').reshape(rows, cols)
        i += 1
    raise KeyError(f'{varname!r} not found in {fpath}')


def load_shrink_from_mats(path, pfile, ndist, ntheta):
    """Build [ntheta, ndist, 2] shrink array from per-distance shrink_list.mat files.

    Each shrink_list.mat contains a (3,2) matrix; row 0 gives [h, v] incremental
    shrink from the previous distance plane. The per-angle shrink for distance k at
    angle j is linearly interpolated: cumulative[k] + increments[k] * j / ntheta.

    The trailing axis is (y, x) = (v, h), matching the position convention: the
    two directions are kept apart rather than averaged, because a sample that
    settles vertically and one that dries horizontally are different geometries
    and the reconstruction can now tell them apart.
    Returns a zero array if any mat file is missing.
    """
    increments = []
    for k in range(ndist):
        mat_path = f'{path}/{pfile}_{k + 1}_/shrink_list.mat'
        if not os.path.exists(mat_path):
            logger.warning(f'shrink_list.mat not found, returning zeros: {mat_path}')
            return np.zeros((ntheta, ndist, 2), dtype='float32')
        sl = load_octave_text_mat(mat_path, 'shrink_list')
        increments.append([float(sl[0, 1]), float(sl[0, 0])])   # (v, h) -> (y, x)
    increments = np.array(increments, dtype='float64')                 # [ndist, 2]
    cumulative = np.concatenate([np.zeros((1, 2)),
                                 np.cumsum(increments, axis=0)])[:ndist]
    j_frac = np.arange(ntheta) / ntheta
    shrink_nd = cumulative[None] + increments[None] * j_frac[:, None, None]
    return shrink_nd.astype('float32')


def read_nxtomo_meta(nx_path):
    """Read geometry and scan metadata from an ESRF NXtomo (.nx) file.

    Returns a dict with:
      entry           str   — HDF5 entry group name
      energy          float — keV
      pixel_size      float — m  (physical detector pixel size)
      z1              float — m  (focus-to-sample propagation distance)
      z_total         float — m  (focus-to-detector distance)
      magnification   float
      voxelsize       float — m
      ny, nx          int   — full detector frame size
      data_ids        ndarray[int] — frame indices where image_key == 0
      flat_ids        ndarray[int] — frame indices where image_key == 1
      dark_ids        ndarray[int] — frame indices where image_key == 2
      x_trans         ndarray[float64] — mm, sample x_translation for data frames (≈ spy)
      y_trans         ndarray[float64] — mm, sample y_translation for data frames (≈ spz)
    """
    with h5py.File(nx_path, 'r') as f:
        entry = next(k for k in f if k.startswith('entry'))
        g = f[entry]

        energy     = float(g['instrument/beam/incident_energy'][()])           # keV
        pixel_size = float(g['instrument/detector/x_pixel_size'][()]) * 1e-6  # µm → m
        _src_dist  = float(g['instrument/source/distance'][()])                 # mm (negative)
        z1         = -_src_dist * 1e-3                                        # mm → m
        z_total    = (float(g['instrument/detector/distance'][()]) - _src_dist) * 1e-3  # mm → m

        image_key = g['instrument/detector/image_key'][:]
        data_ids  = np.where(image_key == 0)[0]
        flat_ids  = np.where(image_key == 1)[0]
        dark_ids  = np.where(image_key == 2)[0]

        ny, nx = g['instrument/detector/data'].shape[1:3]

        x_trans = g['sample/x_translation'][data_ids].astype('float64')  # mm (≈ spy)
        y_trans = g['sample/y_translation'][data_ids].astype('float64')  # mm (≈ spz)

    magnification = z_total / z1
    voxelsize     = pixel_size / magnification

    return dict(
        entry=entry, energy=energy, pixel_size=pixel_size,
        z1=z1, z_total=z_total, magnification=magnification, voxelsize=voxelsize,
        ny=ny, nx=nx,
        data_ids=data_ids, flat_ids=flat_ids, dark_ids=dark_ids,
        x_trans=x_trans, y_trans=y_trans,
    )


def find_latest_checkpoint(path_out, start_iter):
    """Return the path to the most recent checkpoint in path_out, or None."""
    if start_iter > 0:
        files = sorted(glob.glob(os.path.join(path_out, 'checkpoints', f'checkpoint_*{start_iter:04}.h5')))
        return files[-1] if files else None
    else:
        return None


class Reader:
    """MPI-aware HDF5 reader for holotomography data.

    Mirrors Writer: captures all fixed parameters at construction time so each
    read_* method needs no extra arguments beyond what is rank-specific.

    Acquisition parameters (detector_pixelsize, focustodetectordistance, z1,
    energy, ids, theta) are read once in __init__ and stored as attributes.

    File datasets:
      /exchange/obj_init_re{paganin}_{bin}   initial object
      /exchange/cshifts_final                positions
      /exchange/pdata{k}_{bin}               projection data per distance
      /exchange/pref_{bin}                   reference (flat-field)
    """

    def __init__(self, in_file, comm,
                 st_obj, end_obj, nzobj, nobj,
                 st_theta, end_theta, ntheta,
                 ndist, nz, n,
                 paganin, rotation_center_shift, start_theta, bin,
                 tomo_upsample=1, correct3d_extra=0,
                 correct3d_extra_file='correct_correct3D_extra.txt',
                 correct3d_bin=1):
        self.in_file   = in_file
        self.comm      = comm
        self.rank      = comm.Get_rank()
        self.st_obj    = st_obj
        self.end_obj   = end_obj
        self.nzobj     = nzobj
        self.nobj      = nobj
        self.st_theta  = st_theta
        self.end_theta = end_theta
        self.ntheta    = ntheta
        self.ndist     = ndist
        self.nz        = nz
        self.n         = n
        self.paganin   = paganin
        self.rotation_center_shift = rotation_center_shift
        self.bin       = bin
        # Step 5 is untouched by tomo_upsample: it writes its Paganin+FBP init
        # on the PROJECTION grid, nobj*tomo_upsample wide, exactly as it always
        # has, and both arms read the same datasets.  read_obj bins that init
        # down to the object grid.
        self.tomo_upsample = int(tomo_upsample)
        # Step 7's drift refinement, added to the positions in read_pos.
        self.correct3d_extra      = int(correct3d_extra)
        self.correct3d_extra_file = correct3d_extra_file
        self.correct3d_bin        = int(correct3d_bin)

        # Read acquisition parameters once and store as attributes
        with h5py.File(in_file, 'r', driver="mpio", comm=self.comm) as fid:
            self.detector_pixelsize      = fid['/exchange/detector_pixelsize'][0]
            self.focustodetectordistance = fid['/exchange/focusdetectordistance'][0]
            self.z1                      = fid['/exchange/z1'][:ndist]
            self.energy                  = fid['/exchange/energy'][0]
            ntheta0 = len(fid['/exchange/theta'])
            # FIX: clip to exactly ntheta to avoid float-step off-by-one
            ids = np.arange(start_theta, ntheta0, ntheta0 / ntheta)
            self.ids   = ids[:ntheta].astype('int')
            self.theta = -fid['/exchange/theta'][:, 0][self.ids] / 180 * np.pi
            self.detector_pixelsize *= 2**self.bin
            
    def read_obj(self, out=None):
        """Read initial object guess for this rank's z-slice into out."""
        # obj_init may be in a separate _obj.h5 file (written there by step 5
        # to avoid the ~16 TiB Lustre per-file size limit).
        obj_file = self.in_file.replace('.h5', '_obj.h5')
        if not os.path.exists(obj_file):
            obj_file = self.in_file
        logger.info(f"read object from {obj_file}")
        # Step 5 knows nothing about tomo_upsample: its FBP init is on the
        # PROJECTION grid, nobj*tomo_upsample wide.  Read that width and bin it
        # down to the object grid here.  Binning is an AVERAGE, not a sum,
        # because obj is a mean-valued quantity -- the forward model gives
        # proj = C * mean_j(obj_j) along the ray, independent of the object
        # grid, which is also why obj carries between bin levels unrescaled and
        # why bin_z.py averages.  Both arms therefore read the same datasets and
        # step 5 never has to be re-run for a change of tomo_upsample.
        ups  = self.tomo_upsample
        nread = self.nobj * ups                     # projection-grid width
        with h5py.File(obj_file, 'r', driver="mpio", comm=self.comm) as fid:
            obj_ds_re = fid[f'/exchange/obj_init_re{self.paganin}_{self.bin}']
            im_key = f'/exchange/obj_init_im{self.paganin}_{self.bin}'
            obj_ds_im = fid[im_key] if im_key in fid else None
            nzobj0, nobj0 = obj_ds_re.shape[:2]
            if nread > nobj0:
                raise ValueError(
                    f"{obj_ds_re.name} is {nobj0} wide, too small for "
                    f"nobj={self.nobj} x tomo_upsample={ups} = {nread}")
            if self.nzobj > nzobj0:
                # Without this, stz goes negative and the h5py slice wraps
                # round to the end of the dataset, handing back short or empty
                # blocks -- a broadcast error deep in the read loop rather than
                # a statement of what is actually wrong.
                raise ValueError(
                    f"{obj_ds_re.name} has {nzobj0} z slices, too few for "
                    f"nzobj={self.nzobj}")
            stz  = nzobj0 // 2 - self.nzobj // 2
            stx  = nobj0  // 2 - nread // 2
            endx = nobj0  // 2 + nread // 2
            local_nz = self.end_obj - self.st_obj
            if out is None:
                out = np.empty([local_nz, self.nobj, self.nobj], dtype='complex64')
            if self.rank == 0 and ups > 1:
                logger.info(
                    f"obj_init [{self.nzobj}, {nread}, {nread}] -> object grid "
                    f"[{self.nzobj}, {self.nobj}, {self.nobj}] "
                    f"(x/y averaged {ups}x{ups}; z untouched)")
            batch = max(1, (1 << 28) // (nread * nread * obj_ds_re.dtype.itemsize))
            for i0 in range(0, local_nz, batch):
                i1 = min(i0 + batch, local_nz)
                sl = (slice(stz + self.st_obj + i0, stz + self.st_obj + i1),
                      slice(stx, endx), slice(stx, endx))
                re = self._bin_xy(obj_ds_re[sl], ups)
                if out.dtype == np.complex64:
                    out[i0:i1].real[:] = re
                    out[i0:i1].imag[:] = (self._bin_xy(obj_ds_im[sl], ups)
                                          if obj_ds_im is not None else 0)
                else:
                    out[i0:i1] = re
        return out

    @staticmethod
    def _bin_xy(a, ups):
        """Average-bin the last two axes of a [nz, m*ups, m*ups] block by `ups`."""
        if ups == 1:
            return a
        nz, m = a.shape[0], a.shape[1] // ups
        return a.reshape(nz, m, ups, m, ups).mean(axis=(2, 4), dtype='float32')

    def read_pos(self, out=None):
        """Read initial positions for this rank's theta-slice into out.

        Out layout is [ndist, local_ntheta, 2] (dist-major) to match Rec's
        per-dist slicing.
        """
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            raw = fid['/exchange/cshifts_final'][
                self.ids[self.st_theta:self.end_theta], :self.ndist
            ].astype('float32')      # [local_ntheta, ndist, 2]
        raw = np.ascontiguousarray(raw.transpose(1, 0, 2))    # [ndist, local_ntheta, 2]
        if out is None:
            out = raw
        else:
            out[:] = cp.array(raw) if isinstance(out, cp.ndarray) else raw

        # Positions and rotation_center_shift are offsets from the MIDDLE of
        # the detector, and the rotation axis sits at (n-1)/2 at every level
        # (see the gather kernel), so binning is a plain factor -- no
        # half-pixel term.
        scale = np.float32(1.0 / 2**self.bin)
        out *= scale
        out[..., 1] += np.float32(self.rotation_center_shift * scale)
        extra = self.read_correct3d_extra()
        if extra is not None:
            out += cp.array(extra) if isinstance(out, cp.ndarray) else extra
        return out

    def read_correct3d_extra(self):
        """Step 7's per-angle refinement as [ndist, local_ntheta, 2], or None.

        Peter's layout -- columns (horizontal, vertical), one row per scan
        angle plus the 180 deg repeat -- and his binned pixels, so the file
        scales by correct3d_bin exactly as correct_correct3D.txt does.  A
        path is resolved by config.py against the config's own directory.
        """
        if not self.correct3d_extra:
            return None
        path = self.correct3d_extra_file
        if not os.path.exists(path):
            if self.rank == 0:
                logger.info(f'correct3D extra: no {os.path.abspath(path)}, '
                            f'positions unchanged')
            return None
        raw = np.loadtxt(path)
        idx = self.ids[self.st_theta:self.end_theta]
        if raw.shape[0] <= idx.max():
            raise ValueError(f'{path}: {raw.shape[0]} rows, need more than '
                             f'{idx.max()} for this scan')
        scale = np.float32(self.correct3d_bin / 2**self.bin)
        s = raw[idx, ::-1].astype('float32') * scale          # (y, x), this bin
        if self.rank == 0:
            logger.info(f'correct3D extra: {os.path.abspath(path)}, '
                        f'{raw.shape[0]} rows, x{self.correct3d_bin} -> bin '
                        f'{self.bin} px:  y ptp {np.ptp(s[:, 0]):.4f}  '
                        f'x ptp {np.ptp(s[:, 1]):.4f}')
        return np.ascontiguousarray(np.broadcast_to(s, (self.ndist,) + s.shape))

    def read_shrink(self, out=None):
        """Read [ndist, local_ntheta, 2] shrink for this rank's theta-slice.

        Stored on disk as [ntheta, ndist, 2] with the trailing axis (y, x);
        transposed on read so per-dist slices are contiguous. A legacy 2-D
        [ntheta, ndist] dataset (written before shrinkage went per-axis) is
        broadcast across both axes, which reproduces what it used to mean.
        Falls back to zeros if /exchange/shrink is not present.
        """
        local_ntheta = self.end_theta - self.st_theta
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            if '/exchange/shrink' not in fid:
                data = cp.zeros((self.ndist, local_ntheta, 2), dtype='float32')
            else:
                ds  = fid['/exchange/shrink']
                idx = self.ids[self.st_theta:self.end_theta]
                if ds.ndim == 2:            # legacy [ntheta, ndist], axis-agnostic
                    raw = ds[idx, :self.ndist].astype('float32')
                    raw = np.repeat(raw[:, :, None], 2, axis=2)
                else:
                    raw = ds[idx, :self.ndist, :].astype('float32')
                # [local_ntheta, ndist, 2] -> [ndist, local_ntheta, 2]
                data = cp.array(np.ascontiguousarray(raw.transpose(1, 0, 2)))
        if out is not None:
            out[:] = data
        else:
            return data

    def read_prb(self, prb_file=None, out=None):
        """Initialise probe. Loads all ndist probes from prb_file if given, else ones."""
        if out is None:
            out = cp.empty([self.ndist, self.nz, self.n], dtype='complex64')
        if prb_file:
            with h5py.File(prb_file, 'r') as _f:
                for k in range(self.ndist):
                    _amp   = _f['prb_amp'][k]
                    _phase = _f['prb_phase'][k]
                    prb = (_amp * np.exp(1j * _phase)).astype('complex64')
                    nz0, n0 = prb.shape
                    if nz0 > self.nz or n0 > self.n:
                        bz = nz0 // self.nz
                        bn = n0 // self.n
                        prb = prb.reshape(self.nz, bz, self.n, bn).mean(axis=(1, 3))
                    out[k] = cp.array(prb) if isinstance(out, cp.ndarray) else prb
                if self.rank == 0:
                    logger.info(f'Probe read from {prb_file}, shape {tuple(_f["prb_amp"].shape)}')
        else:
            out[:] = 1
        return out

    def read_data(self, out=None):
        """Read projection data for this rank's theta-slice into out.

        Out layout is [ndist, local_ntheta, nz, n] (dist-major) so that per-dist
        slices `out[k]` are contiguous — matches how Rec consumes them in the
        outer-distance loop.

        The values are the measured INTENSITY, exactly as step 4 wrote them.
        This used to return the amplitude — it took the sqrt here — because
        Rec.F0 compared amplitudes; F0 is now an intensity misfit
        (1/N sum m (K|psi|^2 - d)^2), so the sqrt would only be undone.
        read_ref still returns an amplitude, but not because F1 is an amplitude
        misfit -- F1 is an intensity one too now.  PrbfitTerm squares `ref` at
        use, so the array stays an amplitude and every seeding path
        (read_ref, gen_sqrt_ref, disp_study.gen_ref) is left alone.
        """
        nz, n = self.nz, self.n
        local_ntheta = self.end_theta - self.st_theta
        if out is None:
            out = np.empty([self.ndist, local_ntheta, nz, n], dtype='float32')
        # Batch reads to stay under 2^31 bytes (MPI-IO uses int for transfer sizes)
        batch = max(1, (1 << 28) // (nz * n))
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            for k in range(self.ndist):
                nz0 = fid[f'/exchange/pdata{k}_{self.bin}'].shape[1]
                st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                ds = fid[f'/exchange/pdata{k}_{self.bin}']
                for i0 in range(0, local_ntheta, batch):
                    i1 = min(i0 + batch, local_ntheta)
                    out[k, i0:i1] = ds[self.ids[self.st_theta + i0:self.st_theta + i1], st:end]
        return out

    def read_ref(self, out=None):
        """Read reference (flat-field) on rank 0 and broadcast to all ranks."""
        nz = self.nz
        n = self.n
        # FIX: read once on rank 0, broadcast — avoids N redundant identical reads
        raw_np = np.empty((self.ndist, nz, n), dtype='float32')
        if self.rank == 0:
            with h5py.File(self.in_file, 'r') as fid:
                key_start = f'/exchange/pref_{self.bin}'
                key_end   = f'/exchange/pref_end_{self.bin}'
                nz0 = fid[key_start].shape[1]
                st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                raw_np[:] = fid[key_start][:self.ndist, st:end]
                if key_end in fid:
                    raw_np[:] = 0.5 * (raw_np + fid[key_end][:self.ndist, st:end])
        self.comm.Bcast(raw_np, root=0)
        raw = cp.array(raw_np)
        if out is None:
            out = cp.sqrt(raw)
        else:
            cp.sqrt(raw, out=out)
        return out

    def read_checkpoint(self, path, out_obj=None, out_prb=None, out_pos=None,
                        out_bd=None, out_tp=None):
        """Read a checkpoint saved at a coarser resolution and upsample.

        Scale is inferred automatically from checkpoint n vs self.n.

        prb  : upsampled in y and x by scale (repeat).
        obj  : upsampled in x and y by scale (repeat); z mapped by nearest-neighbour.
        pos  : multiplied by scale (pixel coords scale with resolution).
        tp   : (ndist, 2, 2) shrinkage parameters (A, B), NOT scaled by binning --
               shrink is a unitless ratio. A checkpoint written before shrinkage
               became a variable has no /tp, in which case out_tp is left alone
               and rank 0 logs a warning.

        NOTE the step-7 correction is applied ONCE, by read_pos, at the level
        that starts fresh (start_iter=0 -- find_latest_checkpoint returns None
        there whatever is on disk).  Levels that resume inherit it through the
        checkpoint's /pos, so it must not be added again here or it would
        double count.  The log line below records which case this is, because
        the failure mode it replaces -- a rerun that silently changed nothing
        -- is invisible otherwise.
        """
        if self.correct3d_extra and os.path.exists(self.correct3d_extra_file):
            if self.rank == 0:
                logger.info(
                    f'correct3D extra: {os.path.abspath(self.correct3d_extra_file)} '
                    f'not re-applied at this level -- resuming, so the '
                    f'checkpoint\'s positions already carry it IF the ladder '
                    f'was started fresh after step 7.  Only a start_iter=0 '
                    f'level reads it from cshifts_final.')

        # --- infer scale and probe on rank 0, broadcast ---
        prb_np = np.empty((self.ndist, self.nz, self.n), dtype='complex64')
        if self.rank == 0:
            with h5py.File(path, 'r') as f:
                scale = self.n // f['prb_abs'].shape[-1]
                prb_raw = (f['prb_abs'][:] * np.exp(1j * f['prb_phase'][:])).astype('complex64')
            for axis in [2, 1]:
                prb_raw = np.repeat(prb_raw, scale, axis=axis)
            prb_np[:] = prb_raw
            del prb_raw

        scale_arr = np.zeros(1, dtype='int32')
        if self.rank == 0:
            scale_arr[0] = scale
        self.comm.Bcast(scale_arr, root=0)
        scale = int(scale_arr[0])
        self.comm.Bcast(prb_np, root=0)

        if out_prb is None:
            out_prb = cp.array(prb_np)
        else:
            # vars['prb'] is a PINNED HOST buffer (see Rec.__init__), not a device
            # array, so only wrap in cp.array when the caller really passed one.
            out_prb[:] = cp.array(prb_np) if isinstance(out_prb, cp.ndarray) else prb_np
        del prb_np
        # --- obj: z-batched read to cap peak CPU RAM ---
        # Old code read all nz_src slices into obj_re + obj_im + block at once,
        # which can exceed tens of GB per rank for large objects.
        # Now we process one z-batch at a time: peak extra RAM ≈ 2 × batch × nobj0² × 8 B.
        with h5py.File(path, 'r', driver="mpio", comm=self.comm) as f:
            ds_re   = f['obj_re']
            ds_im   = f['obj_im']
            nzobj0  = ds_re.shape[0]
            nobj0   = ds_re.shape[1]
            # `scale` above is the DETECTOR scale, inferred from the probe, and
            # stays right for prb and pos.  The object's z and x/y scales are
            # read from the object dataset instead, because with tomo_upsample
            # they can differ from it and from each other: going bin 1 -> bin 0
            # at tomo_upsample 2 the probe doubles and the object's z doubles,
            # but its x/y grid is already at the target width and must not be
            # repeated.  On every ladder without tomo_upsample all three are
            # equal and this is exactly what the single `scale` did.
            scale_z  = max(1, self.nzobj // nzobj0)
            scale_xy = max(1, self.nobj  // nobj0)
            st_src  = self.st_obj  // scale_z
            end_src = self.end_obj // scale_z
            n0      = self.end_obj - self.st_obj
            nz_src  = max(1, end_src - st_src)
            if self.rank == 0 and (scale_z != scale or scale_xy != scale):
                logger.warning(
                    f"checkpoint {os.path.basename(path)}: obj "
                    f"[{nzobj0}, {nobj0}, {nobj0}] -> [{self.nzobj}, {self.nobj}, "
                    f"{self.nobj}] (z x{scale_z}, x/y x{scale_xy}); "
                    f"prb/pos x{scale}")

            if out_obj is None:
                out_obj = np.empty((n0, self.nobj, self.nobj), dtype='complex64')

            # Target ~256 MB per batch (complex64 = 8 B worst case)
            z_batch = max(1, (1 << 28) // (nobj0 * nobj0 * 8))

            for i0 in range(0, n0, z_batch):
                i1     = min(i0 + z_batch, n0)
                src_i0 = int(i0 * nz_src / n0)
                src_i1 = min(int((i1 - 1) * nz_src / n0) + 1, nz_src)
                nz_b   = src_i1 - src_i0

                # Read directly into a complex64 buffer via .real/.imag views
                blk = np.zeros((nz_b, nobj0, nobj0), dtype='complex64')
                _re = ds_re[st_src + src_i0:st_src + src_i1].astype('float32')
                blk.real[:] = _re; del _re
                if ds_im is not None:
                    _im = ds_im[st_src + src_i0:st_src + src_i1].astype('float32')
                    blk.imag[:] = _im; del _im

                if scale_xy > 1:
                    for axis in [2, 1]:
                        blk = np.repeat(blk, scale_xy, axis=axis)

                idx_local = np.clip(
                    (np.arange(i0, i1) * nz_src / n0).astype(np.intp), 0, nz_src - 1
                ) - src_i0

                if out_obj.dtype == np.complex64:
                    out_obj[i0:i1] = blk[idx_local]
                else:
                    out_obj[i0:i1] = blk[idx_local].real
                del blk

            # --- pos: scale pixel coordinates up ---
            # Stored on disk as [ntheta, ndist, 2]; transposed to [ndist, local_ntheta, 2].
            pos = f['pos'][self.st_theta:self.end_theta].astype('float32')

        pos_up = np.ascontiguousarray(pos.transpose(1, 0, 2)) * scale
        if out_pos is None:
            out_pos = cp.array(pos_up)
        else:
            out_pos[:] = cp.array(pos_up, dtype='float32') if isinstance(out_pos, cp.ndarray) else pos_up

        # Optional scalar attribute, broadcast across ranks. Skipped entirely
        # unless the caller asked for it: it costs a second rank-0 open of the
        # checkpoint plus a collective, and no solver in the package uses it.
        bd_arr = np.full(1, np.nan, dtype='float32')
        if out_bd is not None:
            if self.rank == 0:
                with h5py.File(path, 'r') as f:
                    if 'bd' in f.attrs:
                        bd_arr[0] = float(f.attrs['bd'])
            self.comm.Bcast(bd_arr, root=0)
            if not np.isnan(bd_arr[0]):
                out_bd[0] = float(bd_arr[0])

        tp_np = self._read_tp(path)
        if out_tp is not None and tp_np is not None:
            out_tp[:] = cp.asarray(tp_np) if isinstance(out_tp, cp.ndarray) else tp_np

        return {'obj': out_obj, 'prb': out_prb, 'pos': out_pos,
                'bd': float(bd_arr[0]), 'tp': tp_np}

    def _read_tp(self, path):
        """Rank-0 read of /tp from a checkpoint, broadcast to all ranks.

        Returns None (and warns on rank 0) when the checkpoint predates the
        shrinkage variable, so the caller can keep whatever tp it already has.
        """
        has = np.zeros(1, dtype='int32')
        tp  = np.zeros((self.ndist, 2, 2), dtype='float32')
        if self.rank == 0:
            with h5py.File(path, 'r') as f:
                if 'tp' in f:
                    has[0] = 1
                    tp[:] = f['tp'][:self.ndist].astype('float32')
                elif not getattr(self, '_warned_no_tp', False):
                    # once per reader: recompute_conv walks every checkpoint of
                    # a level, and they are all legacy or all not
                    self._warned_no_tp = True
                    logger.warning(f'read_checkpoint: {path} has no /tp dataset '
                                   f'(legacy checkpoint); leaving tp unchanged.')
        self.comm.Bcast(has, root=0)
        if not has[0]:
            return None
        self.comm.Bcast(tp, root=0)
        return tp

    def read_pos_checkpoint(self, path, out=None, out_tp=None):
        """Read positions from a checkpoint file and upsample to current resolution.

        Scale is inferred from the checkpoint probe size vs self.n. `out_tp`, if
        given, additionally picks up the shrinkage parameters (unscaled).
        """
        if self.rank == 0:
            with h5py.File(path, 'r') as f:
                scale = self.n / f['prb_abs'].shape[-1]
        scale_arr = np.zeros(1, dtype='float32')
        if self.rank == 0:
            scale_arr[0] = scale
        self.comm.Bcast(scale_arr, root=0)
        scale = float(scale_arr[0])

        with h5py.File(path, 'r', driver="mpio", comm=self.comm) as f:
            pos = f['pos'][self.ids[self.st_theta:self.end_theta]].astype('float32')

        # Stored on disk as [ntheta, ndist, 2]; transpose to [ndist, local_ntheta, 2].
        pos_up = np.ascontiguousarray(pos.transpose(1, 0, 2)) * scale
        if out is None:
            out = cp.array(pos_up)
        else:
            out[:] = cp.array(pos_up, dtype='float32') if isinstance(out, cp.ndarray) else pos_up

        if out_tp is not None:
            tp_np = self._read_tp(path)
            if tp_np is not None:
                out_tp[:] = (cp.asarray(tp_np) if isinstance(out_tp, cp.ndarray)
                             else tp_np)
        return out

    def read_obj_unbin(self, out):
        """Read initial object in one bulk I/O call and upsample by 2**bin."""
        st, end = self.st_obj, self.end_obj
        n0 = end - st
        scale = 2 ** (-self.bin)
        nz_src = max(1, n0 // scale)
        st_src = st // scale
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            ds = fid['/exchange/obj']
            batch = max(1, (1 << 28) // (ds.shape[1] * ds.shape[2] * ds.dtype.itemsize))
            block = np.empty((nz_src,) + ds.shape[1:], dtype=ds.dtype)
            for i0 in range(0, nz_src, batch):
                i1 = min(i0 + batch, nz_src)
                block[i0:i1] = ds[st_src + i0 : st_src + i1]
        # upsample spatial dimensions in memory
        block = np.repeat(np.repeat(block, scale, axis=1), scale, axis=2)
        # map source z-slices to output z-slices
        idx0 = np.clip(
            (np.arange(n0) * nz_src / n0).astype(np.intp),
            0, nz_src - 1,
        )
        out[:] = block[idx0].astype('complex64')
        return out

    def read_vol_obj(self, vol_path, out, scale=1.0, vol_dtype='float32'):
        """Read this rank's z-slice from a raw binary .vol file as object initial guess.

        Vol shape is nzobj*2^b x nobj*2^b x nobj*2^b where b is inferred from
        the file size. Block-averaging downsampling is applied when b > 0.
        Each rank reads independently (no MPI-IO needed for raw binary).
        """
        itemsize  = np.dtype(vol_dtype).itemsize
        file_size = os.path.getsize(vol_path)
        total_el  = file_size // itemsize

        # Infer power-of-2 bin level: total_el = nzobj * nobj^2 * 8^b
        base = self.nzobj * self.nobj * self.nobj
        if total_el % base != 0:
            raise ValueError(
                f"{vol_path}: file has {total_el} elements, "
                f"not a multiple of nzobj*nobj*nobj={base}"
            )
        ratio = total_el // base  # should be 8^b
        b = round(math.log2(ratio) / 3) if ratio > 1 else 0
        if 8 ** b != ratio:
            raise ValueError(
                f"{vol_path}: size ratio {ratio} is not a power of 8 "
                f"(expected nzobj*nobj^2 * 8^b)"
            )
        factor   = 2 ** b
        nobj_vol = self.nobj  * factor
        nz_vol   = self.nzobj * factor

        # Centre offsets — symmetric by construction when factor is a power of 2
        stz_vol = (nz_vol   - self.nzobj * factor) // 2  # always 0
        stx_vol = (nobj_vol - self.nobj  * factor) // 2  # always 0

        slice_pixels = nobj_vol * nobj_vol
        local_nz     = self.end_obj - self.st_obj

        logger.info(
            f"read_vol_obj: rank {self.rank} reading z=[{self.st_obj}:{self.end_obj}] "
            f"from vol [{nz_vol},{nobj_vol},{nobj_vol}]"
            + (f" (downsample 2^{b})" if b > 0 else "")
            + f" -> rec [{self.nzobj},{self.nobj},{self.nobj}]"
        )
        with open(vol_path, 'rb') as fh:
            for i in range(local_nz):
                acc = np.zeros([self.nobj * factor, self.nobj * factor], dtype='float32')
                for bz in range(factor):
                    z_vol = stz_vol + (self.st_obj + i) * factor + bz
                    if not (0 <= z_vol < nz_vol):
                        continue
                    fh.seek(z_vol * slice_pixels * itemsize)
                    row = np.frombuffer(fh.read(slice_pixels * itemsize), dtype=vol_dtype).astype('float32')
                    acc += row.reshape(nobj_vol, nobj_vol)[
                        stx_vol:stx_vol + self.nobj * factor,
                        stx_vol:stx_vol + self.nobj * factor,
                    ]
                acc /= factor
                if factor > 1:
                    acc = acc.reshape(self.nobj, factor, self.nobj, factor).mean(axis=(1, 3))
                if out.dtype == np.complex64:
                    out[i].real[:] = acc
                    out[i].imag[:] = 0
                else:
                    out[i][:] = acc

        if scale != 1.0:
            out /= np.float32(scale)
        logger.info(f"read_vol_obj: rank {self.rank} done (scale={scale})")
        return out

    def read_prb_unbin(self, out):
        """Read initial probe and upsample by 2**bin in spatial dimensions."""
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            prb = fid['/exchange/prb'][:]
        scale = 2 ** (-self.bin)
        for axis in [2, 1]:
            prb = np.repeat(prb, scale, axis=axis)
        out[:] = (cp.array(prb) if isinstance(out, cp.ndarray) else prb).astype('complex64')
        return out
