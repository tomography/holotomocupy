import os
import h5py
import numpy as np
import cupy as cp
import tifffile
from .logger_config import logger


class Writer:
    """MPI-aware HDF5 writer for reconstruction checkpoints.

    Uses parallel HDF5 (mpio driver). All ranks open the file collectively;
    obj and pos are written with collective I/O; prb is written by rank 0 only.

    File layout — {path_out}/checkpoints/checkpoint_{iter:04}.h5:
      /obj_re  (nzobj, nobj, nobj)  float32 — real part of obj (assembled from all ranks)
      /obj_im  (nzobj, nobj, nobj)  float32 — imag part of obj (complex64 dtype only)
      /prb_abs   (ndist, nz, n)     float32 — probe amplitude (from rank 0)
      /prb_phase (ndist, nz, n)     float32 — probe phase     (from rank 0)
      /pos     (ntheta, ndist, 2)   float32 — assembled from all ranks (theta-distributed)
      /tp      (ndist, 2, 2)        float32 — linear shrink params (A, B) per axis
                                              (from rank 0; tp is global)

    Attrs on the root group:
      iter

    Two preview TIFFs go alongside it, in {path_out}/checkpoints_tiff/:
      checkpoint_{iter:04}_obj_re.tiff       (nobj, nobj)   -- z = nzobj//2
      checkpoint_{iter:04}_obj_re_vert.tiff  (nzobj, nobj)  -- y = nobj//2,
                                             the vertical cut through the axis
    """

    def __init__(self, path_out, comm,
                 st_obj, end_obj, nzobj, nobj,
                 st_theta, end_theta, ntheta,
                 ndist, nz, n):
        self.path_out  = path_out
        self.comm      = comm
        self.rank      = comm.Get_rank()
        self.size      = comm.Get_size()
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

        self.h5_dir   = os.path.join(path_out, 'checkpoints')
        self.tiff_dir = os.path.join(path_out, 'checkpoints_tiff')
        if self.rank == 0:
            os.makedirs(self.h5_dir,   exist_ok=True)
            os.makedirs(self.tiff_dir, exist_ok=True)
        comm.Barrier()  # ensure directories exist before other ranks proceed

    @staticmethod
    def _cpu(x):
        """Move a CuPy or NumPy array to a contiguous CPU NumPy array."""
        if isinstance(x, cp.ndarray):
            return x.get()
        return np.asarray(x)

    def write_checkpoint(self, vars, i, norm_const, pos_init=None,
                         shrink=None, shrink_init=None, shrink_gt=None):
        """Save obj, prb, pos, tp for iteration i to an HDF5 checkpoint file.

        Parameters
        ----------
        vars : dict
            Reconstruction variables with keys 'obj', 'prb', 'pos' and,
            when shrinkage is a variable, 'tp' (ndist, 2, 2) — GLOBAL, i.e.
            identical on every rank, so rank 0 writes it.
            obj is expected to be scaled by 1/norm_const (as during iteration).
        i : int
            Iteration number, used in the filename.
        norm_const : float
            Normalisation constant — obj is multiplied by this before saving.
        pos_init : array, optional
            Initial positions; when provided, a PNG of per-(theta, dist) drift
            is also saved under {path_out}/pos_errors/.
        shrink, shrink_init : array, optional
            This rank's shrink slices, (local_ntheta, ndist, 2). When both are
            given, a PNG overlaying init vs current shrink per (dist, axis) is
            saved under {path_out}/shrink/.
        shrink_gt : array, optional
            Full (ntheta, ndist, 2) ground truth, identical on every rank
            (synthetic data only); drawn as a third curve on that PNG.
        """
        path = os.path.join(self.h5_dir, f"checkpoint_{i:04}.h5")

        pos = self._cpu(vars['pos'])
        prb = self._cpu(vars['prb'])
        # tp is small (ndist, 2, 2) and global — same on every rank.
        tp  = self._cpu(vars['tp']) if 'tp' in vars else None

        # mpio block: all ranks create datasets and write obj/pos collectively
        with h5py.File(path, 'w', driver="mpio", comm=self.comm) as f:
            f.attrs['iter']      = i
            # Save scalar variables (e.g. RecDelta's bd) as HDF5 attributes if present.
            if 'bd' in vars:
                f.attrs['bd'] = float(vars['bd'][0])

            obj_shape = (self.nzobj, self.nobj, self.nobj)
            ds_re = f.create_dataset('obj_re', shape=obj_shape, dtype='float32')
            ds_im = f.create_dataset('obj_im', shape=obj_shape, dtype='float32')
            ds_pos = f.create_dataset('pos', shape=(self.ntheta, self.ndist, 2), dtype='float32')
            prb_shape = (self.ndist, self.nz, self.n)
            f.create_dataset('prb_abs',   shape=prb_shape, dtype='float32')
            f.create_dataset('prb_phase', shape=prb_shape, dtype='float32')
            if tp is not None:
                # Created collectively (mpio requires it), filled by rank 0 below.
                f.create_dataset('tp', shape=tp.shape, dtype='float32')

            # Write obj in z-batches: avoids a full [local_nzobj, nobj, nobj] copy.
            # np.multiply(src, scalar, out=slab_buf) is zero-allocation per batch.
            local_nz = self.end_obj - self.st_obj
            z_batch  = max(1, (1 << 28) // (self.nobj * self.nobj * 4))  # ~256 MB slab
            slab_buf = np.empty((z_batch, self.nobj, self.nobj), dtype='float32')
            for i0 in range(0, local_nz, z_batch):
                i1  = min(i0 + z_batch, local_nz)
                nzb = i1 - i0
                obj_slab = vars['obj'][i0:i1]          # pinned view, no copy
                np.multiply(obj_slab.real, np.float32(norm_const), out=slab_buf[:nzb])
                ds_re[self.st_obj + i0:self.st_obj + i1] = slab_buf[:nzb]
                np.multiply(obj_slab.imag, np.float32(norm_const), out=slab_buf[:nzb])
                ds_im[self.st_obj + i0:self.st_obj + i1] = slab_buf[:nzb]
            del slab_buf

            # vars['pos'] is [ndist, local_ntheta, 2]; on-disk format is [ntheta, ndist, 2].
            ds_pos[self.st_theta:self.end_theta] = np.ascontiguousarray(pos.transpose(1, 0, 2))

        # Vertical (y = mid) slice, assembled from the ranks' own z slabs rather
        # than read back out of the file: the horizontal slice below is one
        # contiguous row of obj_re, but a vertical one is nzobj scattered reads
        # (3264 of them at bin 0), and every rank already holds its part of it.
        vmid    = self.nobj // 2
        v_local = self._cpu(vars['obj'][:, vmid, :].real).astype('float32')
        v_local *= np.float32(norm_const)
        v_parts = self.comm.gather(v_local, root=0)

        # prb + tp written by rank 0 only via serial driver after mpio block closes
        self.comm.Barrier()
        if self.rank == 0:
            with h5py.File(path, 'a') as f:
                f['prb_abs'][:]   = np.abs(prb).astype('float32')
                f['prb_phase'][:] = np.angle(prb).astype('float32')
                if tp is not None:
                    f['tp'][:] = tp.astype('float32')
        self.comm.Barrier()
        if self.rank == 0:
            logger.info(f"Writer: checkpoint saved → {path}")
            mid = self.nzobj // 2
            with h5py.File(path, 'r') as f:
                slice_re = f['obj_re'][mid]
            tiff_path = os.path.join(self.tiff_dir, f"checkpoint_{i:04}_obj_re.tiff")
            tifffile.imwrite(tiff_path, slice_re)
            logger.info(f"Writer: mid-slice TIFF saved → {tiff_path}")

            # ... and the vertical one, (nzobj, nobj), through the rotation axis
            vert_path = os.path.join(self.tiff_dir, f"checkpoint_{i:04}_obj_re_vert.tiff")
            tifffile.imwrite(vert_path, np.concatenate(v_parts, axis=0))
            logger.info(f"Writer: vertical-slice TIFF saved → {vert_path}")

        if pos_init is not None:
            self._save_pos_errors_plot(vars['pos'] - pos_init, i)

        if shrink is not None and shrink_init is not None:
            self._save_shrink_plot(shrink, shrink_init, i, shrink_gt=shrink_gt)

    def _save_pos_errors_plot(self, delta_local, i):
        """Gather per-(theta, dist) position drift across ranks; rank 0 logs stats and saves PNG."""
        if isinstance(delta_local, cp.ndarray):
            delta_local = delta_local.get()
        # delta_local is [ndist, local_ntheta, 2]; gather concatenates along the theta axis (axis=1).
        all_deltas = self.comm.gather(delta_local, root=0)
        if self.rank != 0:
            return

        all_delta = np.concatenate(all_deltas, axis=1)    # [ndist, ntheta, 2]
        abs_delta = np.abs(all_delta)
        mean_err  = abs_delta.mean(axis=1)                # [ndist, 2]
        std_err   = abs_delta.std(axis=1)
        max_err   = abs_delta.max(axis=1)
        parts = "  ".join(
            f"d{j}: y=({mean_err[j,0]:.4f}±{std_err[j,0]:.4f} max={max_err[j,0]:.4f})"
            f"  x=({mean_err[j,1]:.4f}±{std_err[j,1]:.4f} max={max_err[j,1]:.4f})"
            for j in range(self.ndist)
        )
        logger.warning(f"iter={i}: pos abs error [px]  {parts}")

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, self.ndist, figsize=(5 * self.ndist, 6))
        if self.ndist == 1:
            axes = axes[:, np.newaxis]
        theta_idx = np.arange(self.ntheta)
        for j in range(self.ndist):
            for d, label in enumerate(['y', 'x']):
                ax = axes[d, j]
                ax.plot(theta_idx, all_delta[j, :, d])
                ax.set_title(f"dist {j}, {label}")
                ax.set_xlabel("theta index")
                ax.set_ylabel("error [px]")
                ax.grid(True)
        fig.tight_layout()
        pos_err_dir = os.path.join(self.path_out, "pos_errors")
        os.makedirs(pos_err_dir, exist_ok=True)
        png_path = os.path.join(pos_err_dir, f"pos_error_{i:04}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info(f"pos error plot → {png_path}")

    def _save_shrink_plot(self, shrink_local, shrink_init_local, i, shrink_gt=None):
        """Gather per-(theta, dist) shrink across ranks and save a 2×ndist plot
        overlaying init (dashed), current (solid), and — when provided — ground
        truth (dotted). Row 0 = y axis, row 1 = x."""
        if isinstance(shrink_local, cp.ndarray):
            shrink_local = shrink_local.get()
        if isinstance(shrink_init_local, cp.ndarray):
            shrink_init_local = shrink_init_local.get()
        all_curr = self.comm.gather(np.asarray(shrink_local),      root=0)
        all_init = self.comm.gather(np.asarray(shrink_init_local), root=0)
        # shrink_gt is a full-ntheta array, identical on every rank, so no gather.
        if self.rank != 0:
            return

        # locals are [local_ntheta, ndist, 2]; concatenate along theta.
        curr = np.concatenate(all_curr, axis=0)           # [ntheta, ndist, 2]
        init = np.concatenate(all_init, axis=0)
        gt   = None if shrink_gt is None else (
            shrink_gt.get() if isinstance(shrink_gt, cp.ndarray) else np.asarray(shrink_gt))

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, self.ndist,
                                 figsize=(5 * self.ndist, 6), squeeze=False)
        theta_idx = np.arange(curr.shape[0])
        for j in range(self.ndist):
            for d, label in enumerate(['y', 'x']):
                ax = axes[d, j]
                ax.plot(theta_idx, init[:, j, d], label='init', linestyle='--', color='C1')
                ax.plot(theta_idx, curr[:, j, d], label='current',              color='C0')
                if gt is not None:
                    ax.plot(theta_idx, gt[:, j, d], label='ground truth',
                            linestyle=':', color='C2', linewidth=2)
                ax.set_title(f"dist {j}, {label}")
                ax.set_xlabel("theta index")
                ax.set_ylabel("shrink")
                ax.grid(True)
                ax.legend(fontsize=8)
        fig.tight_layout()
        shrink_dir = os.path.join(self.path_out, "shrink")
        os.makedirs(shrink_dir, exist_ok=True)
        png_path = os.path.join(shrink_dir, f"shrink_{i:04}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info(f"shrink plot → {png_path}")
