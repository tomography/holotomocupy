#!/usr/bin/env python
"""gen_data.py with the brain-volume defaults.

Same code as the rest of the study -- this only pins the settings of the
brain test in one place, so run_brain.sh and a bare run agree:

  * the object is the real reconstructed volume in OBJ_VOL instead of the
    synthetic phantom, scaled by --obj-scale and rescaled onto the object grid
    so it never leaves the detector window at +-128 px displacement;
  * n = 2048, i.e. the unbinned detector, so the probe is used at its native
    2048^2 and nothing is cropped or resampled;
  * sigma = 0 -- the measured ID16A probe, no smoothing at all.

Any flag given on the command line overrides the default of the same name:

    mpirun -np 4 ./set_affinity_gpu.sh python gen_data_brain.py --out <dir>
    mpirun -np 4 ./set_affinity_gpu.sh python gen_data_brain.py --ndist 4 --ntheta 1800
"""

import os
import runpy
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
OBJ_VOL = os.environ.get('BRAIN_OBJ_VOL',
                         '/data3/vnikitin/mosaic_brain/init.h5::exchange/data')

DEFAULTS = [
    '--n',          '2048',      # unbinned detector: probe used as measured
    '--ntheta',     '1800',
    '--ndist',      '1',
    '--amp',        '128',       # displacement half-width [detector px]
    '--prb-smooth', '0',         # the measured probe, unsmoothed
    '--obj-vol',    OBJ_VOL,
    '--obj-scale',  '15',        # sets the projected phase; see the line the run prints
    '--photons',    '0',         # noiseless
    '--nchunk',     '4',
]

sys.argv = [os.path.join(_HERE, 'gen_data.py')] + DEFAULTS + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
