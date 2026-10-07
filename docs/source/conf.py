"""Sphinx configuration for the holotomocupy docs.

Read the Docs builds this in a clean container with no GPU, no MPI and no
CUDA, so every import that needs one is mocked below -- autodoc only reads
signatures and docstrings, it never runs a kernel.  `logger_config` calls
MPI.COMM_WORLD.Get_rank() at module scope; the mock answers that happily.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('../../src'))

project = 'holotomocupy'
author = 'Viktor Nikitin'
copyright = f'{date.today().year}, Argonne National Laboratory'
version = release = open(os.path.join(os.path.dirname(__file__), '..', '..', 'VERSION')).read().strip()

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_copybutton',
    'sphinxcontrib.bibtex',
]

# Carried over from the upstream docs: the papers to cite (credits.rst).
bibtex_bibfiles = ['bibtex/cite.bib', 'bibtex/ref.bib']
bibtex_default_style = 'unsrt'

# Everything that needs a GPU, an MPI runtime or a beamline file format.
autodoc_mock_imports = [
    'cupy', 'cupyx', 'mpi4py', 'h5py', 'nvtx', 'dxchange', 'tifffile',
    'pandas', 'psutil', 'matplotlib',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

myst_enable_extensions = ['colon_fence', 'deflist']
myst_heading_anchors = 3
suppress_warnings = ['myst.xref_missing', 'myst.header']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

html_theme = 'furo'
html_title = f'holotomocupy {version}'
html_static_path = []
