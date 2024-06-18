============
Installation
============


::

    (base) $ conda create -n holotomocupy -c conda-forge cupy dxchange xraylib matplotlib jupyter astropy olefile
    (base) $ conda activate holotomocupy
    (holotomocupy)$ $ git clone https://github.com/nikitinvv/holotomocupy
    (holotomocupy)$ $ cd holotomocupy
    (holotomocupy)$ $ pip install .


Update
======

**holotomocupy** is constantly updated to include new features. To update your locally installed version

::

    (holotomocupy)$ cd holotomocupy
    (holotomocupy)$ git pull
    (holotomocupy)$ pip install .
