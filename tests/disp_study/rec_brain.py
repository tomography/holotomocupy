#!/usr/bin/env python
"""rec.py with the brain-volume defaults.

  * 1025 BH iterations, positions started 2 px off the truth;
  * the initial object is the ground truth blurred by 32 px -- 4x the n = 512
    sigma, matching the 4x bigger grid, so the starting point holds the same
    fraction of the spectrum (common.gaussian_blur3d does it on the GPU);
  * checkpoints every 512 iterations: one is a full 2458^3 complex volume,
    119 GB, so the n = 512 step of 32 would write 3.8 TB.

Any flag given on the command line overrides the default of the same name:

    mpirun -np 4 ./set_affinity_gpu.sh python rec_brain.py --in <dir>
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = [
    '--niter',           '1025',
    '--pos-err',         '0',     # initial position error half-width [px]
    '--obj-init',        'blur',
    '--obj-blur',        '32',
    '--checkpoint-step', '512',
    '--error-step',      '32',
    '--nchunk',          '4',
]

sys.argv = [os.path.join(_HERE, 'rec.py')] + DEFAULTS + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
