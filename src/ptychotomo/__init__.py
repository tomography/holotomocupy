from pkg_resources import get_distribution, DistributionNotFound

from ptychotomo.solver_admm import *
#from ptychotomo.gendata import *

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
