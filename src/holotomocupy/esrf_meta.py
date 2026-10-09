"""Read back what ESRF's own pipeline recorded about a scan.

Peter's drop for a scan is spread over two directories -- `<pfile>_1_/` holds
the EDFs and the beamline's own files, `<pfile>_/` holds his octave pipeline's
outputs and a `naburec/` -- and both of them state, in passing, numbers we
would otherwise have to retype: the rotation axis he reconstructed on, and the
binning of the grid his shift files are fitted on.

Retyping those is how they go stale.  `<pfile>_rec_.par`'s
ROTATION_AXIS_POSITION was 73.7 px out of date on AtomiumS1_HT and ~15 px out
on ctxl_FT; the `naburec/*.conf` next to it was right both times.  So every
function here reads, reports and returns -- **nothing here changes a
reconstruction**.  `steps15.py` logs what these return and warns when it
disagrees with the config, and the config value is what is actually used.  The
one exception is `find_drop_file`, which locates a file we then read.

Units: every `*_pixelsize` / `pixelsize_m` in this module is **metres**, to
match `voxelsize` in steps15.py.  ESRF's `.info` files are in microns and are
converted on read; the `.mat` files are already in metres.

Nothing here raises on a missing or malformed file: a drop that has not landed
yet, or has landed one level too deep, returns None and a note.  That is the
normal case for a scan ESRF has not processed.
"""

import glob
import os

import numpy as np

from .logger_config import logger
from .reader import load_octave_text_mat

# naburec/ accumulates a conf per attempt and they do not all agree -- on
# ctxl_FT, nabu_correct3D_v.conf reads 1031.121372 where the other six read
# 1031.235375.  Prefer the ones that produced a full volume, most-final first.
# Anything not listed is still read, reported as a candidate, and ranked last.
_NABU_CONF_PRIORITY = (
    'nabu_final_cm.conf',       # what the three full-recon slurm jobs ran
    'nabu_final_cm_even.conf',
    'nabu_final_cm_odd.conf',
    'nabu_final.conf',
    'nabu_correct3D.conf',
    'nabu.conf',
)


# ---------------------------------------------------------------------------
# drop layout
# ---------------------------------------------------------------------------

def drop_dirs(path, pfile):
    """The directories a drop's files can be in, in search order.

    `<pfile>_/` first (Peter's pipeline outputs), then `<pfile>_1_/` (EDFs and
    beamline files), then one level deeper in each -- a drop that was copied
    with its own parent directory lands in `<pfile>_/<pfile>_/` and reads as
    empty otherwise.
    """
    base = path.rstrip('/')
    outer = [f'{base}/{pfile}_', f'{base}/{pfile}_1_']
    return outer + [f'{d}/{os.path.basename(d)}' for d in outer]


def find_drop_file(path, pfile, name):
    """Locate `name` anywhere in the drop.  Returns (abspath or None, note).

    `note` is '' for the ordinary case, and otherwise says something the caller
    should log: found nested one level deeper, or not found at all.
    """
    cands = drop_dirs(path, pfile)
    for i, d in enumerate(cands):
        p = os.path.join(d, name)
        if os.path.exists(p):
            if i < 2:
                return p, ''
            return p, (f'{name} found NESTED at {p} -- the drop was copied one '
                       f'level too deep; everything else in it is probably '
                       f'nested too')
    return None, f'{name} not found in any of: ' + ', '.join(cands)


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------

def bin_from_pixelsize(pixelsize_m, voxelsize_m, tol=0.02):
    """Bin factor implied by a stated pixel size.  Returns (int or None, why).

    ESRF's shift files are fitted on whatever grid his pipeline ran on, and
    each file stamps that grid's pixel size.  Dividing by this scan's own voxel
    gives the factor those numbers have to be multiplied by to reach raw
    detector pixels -- measured from the file rather than guessed from the
    driver's `bin_factor`, which is often left unset.

    None when either input is missing or the ratio is not within `tol` of a
    positive integer; `why` always explains which.
    """
    if pixelsize_m is None or voxelsize_m is None:
        return None, 'pixel size or voxel size unknown'
    try:
        ratio = float(pixelsize_m) / float(voxelsize_m)
    except (TypeError, ValueError, ZeroDivisionError):
        return None, f'cannot divide {pixelsize_m!r} by {voxelsize_m!r}'
    if not np.isfinite(ratio) or ratio <= 0:
        return None, f'ratio {ratio} is not a positive number'
    n = int(round(ratio))
    if n < 1:
        return None, f'ratio {ratio:.4f} rounds below 1'
    if abs(ratio - n) > tol * n:
        return None, (f'ratio {ratio:.4f} is not within {tol:.0%} of an integer '
                      f'(nearest {n})')
    return n, f'{pixelsize_m:.6g} m / {voxelsize_m:.6g} m = {ratio:.4f} -> {n}'


# ---------------------------------------------------------------------------
# <pfile>_rec_.info -- the grid nabu reconstructed on
# ---------------------------------------------------------------------------

def rec_info(path, pfile):
    """Parse `<pfile>_rec_.info` into a dict, or None if there is none.

    Keys are lowercased as written (`dim_1`, `pixelsize`, `correct_shrink`,
    `delta_beta`, ...) with numeric values as floats, plus two derived ones:

      pixelsize_m   PixelSize converted from microns to metres
      width         Dim_1, as an int, when present

    `PixelSize` here is the grid `correct_correct3D.txt` is fitted on, which is
    the point of reading it: on ctxl_FT it is 0.015 um against a 0.0075 um
    voxel, so that file is in 2x2-binned pixels.
    """
    p, _ = find_drop_file(path, pfile, f'{pfile}_rec_.info')
    if p is None:
        return None
    out = {'source': p}
    try:
        with open(p, 'r') as f:
            for line in f:
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip().rstrip('=').strip().lower()
                v = v.strip()
                if not k:
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    except OSError as e:
        logger.warning(f'esrf_meta: cannot read {p}: {e}')
        return None
    if isinstance(out.get('pixelsize'), float):
        out['pixelsize_m'] = out['pixelsize'] * 1e-6      # .info is in microns
    else:
        out['pixelsize_m'] = None
    if isinstance(out.get('dim_1'), float):
        out['width'] = int(out['dim_1'])
    else:
        out['width'] = None
    return out


# ---------------------------------------------------------------------------
# Octave .mat scalars
# ---------------------------------------------------------------------------

def mat_pixelsize(matpath):
    """`pixelsize` out of an Octave ASCII .mat, in metres, or None.

    `rhapp.mat` and `reference_motion.mat` both carry one, and it is the grid
    their own numbers are on.  Missing file, missing variable and unparseable
    file all return None rather than raising -- at ndist=1 there is no
    rhapp.mat at all, which is normal.
    """
    if not matpath or not os.path.exists(matpath):
        return None
    try:
        v = load_octave_text_mat(matpath, 'pixelsize')
    except (KeyError, ValueError, OSError) as e:
        logger.debug(f'esrf_meta: no usable pixelsize in {matpath}: {e}')
        return None
    v = np.asarray(v, dtype='float64').ravel()
    return float(v[0]) if v.size else None


def reference_motion(path, pfile):
    """`reference_motion.mat`, or None.  Keys:

      ref_v, ref_h              [ntheta] drift of the reference plane, ESRF's
                                own, which validates our correct_motion read
      reference_plane_1based    as written in the file (1-based)
      ref_dist                  the same minus 1, i.e. our 0-based ref_dist
      pixelsize_m               the grid ref_v/ref_h are on

    The reference plane is not ours to pick: rhapp is differenced against it
    and `correct_motion.txt` is written for it, so a mismatch with the config's
    `ref_dist` offsets every distance by a constant.  Hence returning it.
    """
    p, _ = find_drop_file(path, pfile, 'reference_motion.mat')
    if p is None:
        return None
    out = {'source': p, 'ref_v': None, 'ref_h': None,
           'reference_plane_1based': None, 'ref_dist': None,
           'pixelsize_m': mat_pixelsize(p)}
    for key in ('ref_v', 'ref_h'):
        try:
            out[key] = np.asarray(load_octave_text_mat(p, key),
                                  dtype='float64').ravel()
        except (KeyError, ValueError, OSError) as e:
            logger.debug(f'esrf_meta: no {key} in {p}: {e}')
    try:
        rp = np.asarray(load_octave_text_mat(p, 'reference_plane'),
                        dtype='float64').ravel()
        if rp.size:
            out['reference_plane_1based'] = int(rp[0])
            out['ref_dist'] = int(rp[0]) - 1
    except (KeyError, ValueError, OSError) as e:
        logger.debug(f'esrf_meta: no reference_plane in {p}: {e}')
    return out


# ---------------------------------------------------------------------------
# rotation axis
# ---------------------------------------------------------------------------

def _rcs_from_axis(axis_pos, width, bin_factor, one_based):
    """Axis position on a binned grid -> rotation_center_shift in raw px.

    The centre of an N-wide grid is (N-1)/2 counting from 0.  A PyHST `.par`
    counts from 1, so its centre is (N+1)/2 in its own coordinates; passing
    `one_based=True` uses that instead.  Scaling by the grid's bin factor puts
    the result in raw detector pixels, which is what `rotation_center_shift`
    is.  Used AS PRINTED, no negation -- two ESRF truths settled that.
    """
    if axis_pos is None or not width or not bin_factor:
        return None
    centre = (width + 1) / 2.0 if one_based else (width - 1) / 2.0
    return (float(axis_pos) - centre) * float(bin_factor)


def _parse_nabu_conf(fpath):
    """Flatten a nabu .conf to {key: str}.  Section headers are dropped -- the
    keys we want are unique across sections."""
    out = {}
    try:
        with open(fpath, 'r') as f:
            for line in f:
                line = line.split('#', 1)[0].strip()
                if not line or line.startswith('[') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                out[k.strip()] = v.strip()
    except OSError as e:
        logger.debug(f'esrf_meta: cannot read {fpath}: {e}')
    return out


def _grid(path, pfile, voxelsize, extra_bin=1):
    """(width, bin_factor) of the grid ESRF reconstructed on, or (None, None).

    Read from `<pfile>_rec_.info`, never assumed: his grid can be pi/2-padded
    (ctxl_HT's is 3216, not 4096).  `extra_bin` is nabu's own `binning`, which
    shrinks the grid again on top of whatever the `.info` already describes.
    """
    info = rec_info(path, pfile)
    if info is None or not info.get('width'):
        return None, None
    b, _ = bin_from_pixelsize(info.get('pixelsize_m'), voxelsize)
    if b is None:
        return None, None
    return info['width'] // extra_bin, b * extra_bin


def nabu_axis(path, pfile, voxelsize):
    """Rotation axis from `<pfile>_/naburec/*.conf`, in raw detector px.

    Returns None when there is no naburec/, else a dict:

      rcs         rotation_center_shift implied by the preferred conf, or None
      source      that conf's full path
      note        how it was chosen, and whether the candidates disagree
      candidates  [{file, axis_pos, shifts, rcs}] for every conf that states an
                  axis, preferred first

    Never applied automatically.  The axis is degenerate with the x column of
    cshifts_final, so it has to be typed into `rotation_center_shift` in
    `config_steps15.conf` AND every `config_step6_*.conf` together; a switch
    that changed only one of them would silently desynchronise step 5 from
    step 6.  So this reports, and steps15.py warns on a mismatch.
    """
    nabudir, _ = find_drop_file(path, pfile, 'naburec')
    if nabudir is None or not os.path.isdir(nabudir):
        return None
    confs = sorted(glob.glob(os.path.join(nabudir, '*.conf')))
    if not confs:
        return None

    def rank(p):
        b = os.path.basename(p)
        return _NABU_CONF_PRIORITY.index(b) if b in _NABU_CONF_PRIORITY \
            else len(_NABU_CONF_PRIORITY)

    cands = []
    for c in sorted(confs, key=lambda p: (rank(p), os.path.basename(p))):
        cfg = _parse_nabu_conf(c)
        raw = cfg.get('rotation_axis_position', '')
        try:
            axis_pos = float(raw)
        except ValueError:
            continue                      # alignment_config.conf states none
        try:
            extra_bin = int(float(cfg.get('binning', 1) or 1))
        except ValueError:
            extra_bin = 1
        width, bin_factor = _grid(path, pfile, voxelsize, extra_bin)
        shifts = os.path.basename(cfg.get('translation_movements_file', '') or '')
        cands.append({'file': os.path.basename(c),
                      'axis_pos': axis_pos,
                      'shifts': shifts or None,
                      'nabu_binning': extra_bin,
                      'rcs': _rcs_from_axis(axis_pos, width, bin_factor, False)})
    if not cands:
        return None

    best = cands[0]
    spread = max(c['axis_pos'] for c in cands) - min(c['axis_pos'] for c in cands)
    note = f'from {best["file"]}'
    if best['shifts']:
        note += f' (shifts {best["shifts"]})'
    if spread > 1e-6:
        note += (f'; {len(cands)} confs span {spread:.6f} px on the binned grid '
                 f'-- preferred the most-final one')
    return {'rcs': best['rcs'], 'source': os.path.join(nabudir, best['file']),
            'note': note, 'candidates': cands}


def pyhst_axis(path, pfile, voxelsize):
    """Rotation axis from the PyHST `<pfile>_rec_.par`, in raw detector px.

    A cross-check only, and a weak one: `.par` files go stale.  It was 73.7 px
    out on AtomiumS1_HT and ~16 px out here, while the `naburec/*.conf` beside
    it was right both times.  Believe `nabu_axis` when they disagree.

    PyHST counts pixels from 1, so the centre used is (N+1)/2 -- one pixel
    left of nabu's.  `rcs` is therefore 2 x bin_factor lower than reading the
    same number 0-based; it makes no difference to the "is the .par stale"
    question, which is the only thing this is for.
    """
    p, _ = find_drop_file(path, pfile, f'{pfile}_rec_.par')
    if p is None:
        return None
    axis_pos = None
    try:
        with open(p, 'r') as f:
            for line in f:
                line = line.split('#', 1)[0]
                if 'ROTATION_AXIS_POSITION' in line and '=' in line:
                    try:
                        axis_pos = float(line.split('=', 1)[1].strip())
                    except ValueError:
                        pass
                    break
    except OSError as e:
        logger.debug(f'esrf_meta: cannot read {p}: {e}')
        return None
    if axis_pos is None:
        return None
    width, bin_factor = _grid(path, pfile, voxelsize)
    return {'rcs': _rcs_from_axis(axis_pos, width, bin_factor, True),
            'source': p, 'axis_pos': axis_pos,
            'note': f'PyHST .par, 1-based centre ({width}+1)/2'
                    if width else 'PyHST .par, grid unknown'}
