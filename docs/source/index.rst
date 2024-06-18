============
Holotomocupy
============

Holotomography is a coherent imaging technique that provides three-dimensional reconstruction of a sample’s complex refractive index by integrating holography principles with tomographic methods. This approach is particularly suitable for micro- and nano-tomography instruments at the latest generation of synchrotron sources.

This software package presents a family of novel algorithms, encapsulated in an efficient implementation for X-ray holotomography reconstruction. 


Features
--------

* Based on Python, GPU acceleration with cuPy (GPU-accelerated numPy). Easy to install with pip, no C/C++ or NVCC compilers needed. 

* Regular operators (tomographic projection, Fresnel propagator, scaling, shifts, etc.) and processing methods are implemented and can be reused.

* Jupyter notebooks give examples of full pipelines for synthetic/experimental data reconstruction.

* New operators/processing methods can be added by users. Implemented Python decorator @gpu_batch splits data into chunks if data do not fit into GPU memory.

* Pipeline GPU data processing with CUDA streams within cuPy allows significantly reduced time for some CPU-GPU memory transfers.

* Demonstrated for:

    1. Holotomography reconstruction with illumination retrieval.
    2. Holotomography reconstruction with coded apertures.


Contribute
----------

* Documentation: https://github.com/tomography/holotomocupy/tree/master/doc
* Issue Tracker: https://github.com/tomography/holotomocupy/docs/issues
* Source Code: https://github.com/tomography/holotomocupy/

Content
-------

.. toctree::
   :maxdepth: 2

   install
   usage
   api
   credits
