#!/usr/bin/env python
"""make_figure.py for the brain test: ground truth next to the reconstruction.

The brain test is a single case, not a sweep, so this is make_figure.py with
--amps / --prb-smooths pinned to that one case and the probe rows off.  Writes
slices_brain*.png (and a one-point nrmse_brain*.png) into this folder.

    python make_figure_brain.py --root /data3/vnikitin/brain_study
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = [
    '--root',        os.environ.get('BRAIN_STUDY_OUT', '/data3/vnikitin/brain_study'),
    '--ndist',       '1',
    '--amps',        '128',
    '--prb-smooths', '0',
    '--probe-row',   'off',
    '--slices',      'hv',
    '--tag',         'brain',
    '--crop',        '0.10',
    # no pinned --vmin/--vmax: Re(obj) = -delta scales with OBJ_SCALE (roughly
    # [-70, -20] at 15, [-5, -2.5] at 1), so a fixed range blanks the panels as
    # soon as the scale changes.  The range comes from the 99.8th percentile of
    # the ground truth instead; pass --vmin/--vmax to pin it.
]

sys.argv = [os.path.join(_HERE, 'make_figure.py')] + DEFAULTS + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
