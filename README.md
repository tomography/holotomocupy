# HolotomocuPy
## Overview

Holotomography is a coherent imaging technique that provides three-dimensional reconstruction of a sample’s complex refractive index by integrating holography principles with tomographic methods. This approach is particularly suitable for micro- and nano-tomography instruments at the latest generation of synchrotron sources.

This software package presents a family of novel algorithms, encapsulated in an efficient implementation for X-ray holotomography reconstruction. 

## Key features

### Based on Python, GPU acceleration with cuPy (GPU-accelerated numPy). Easy to install with pip, no C/C++ or NVCC compilers needed. 

### Regular operators (tomographic projection, Fresnel propagator, scaling, shifts, etc.) and processing methods are implemented and can be reused.

### Jupyter notebooks give examples of full pipelines for synthetic/experimental data reconstruction.

### New operators/processing methods can be added by users. Implemented Python decorator @gpu_batch splits data into chunks if data do not fit into GPU memory.

### Pipeline GPU data processing with CUDA streams within cuPy allows significantly reduced time for some CPU-GPU memory transfers.

### Demonstrated for: 1. Holotomography reconstruction with illumination retrieval, 2. Holotomography reconstruction with coded apertures



## Installation

```console
conda create -n holotomocupy -c conda-forge cupy dxchange xraylib matplotlib jupyter astropy olefile
```

```console
git clone https://github.com/nikitinvv/holotomocupy

cd holotomocupy

pip install .
```

## Jupyter notebook for synthetic data reconstruction

### 2d Siemens star 

*examples_synthetic/siemens_star/modeling.ipynb* - generate data for a siemens star

*examples_synthetic/siemens_star/rec_1step.ipynb - phase retreival by a 1-step method (CTF, MultiPaganin)

*examples_synthetic/siemens_star/rec_iterative.ipynb - iterative reconstruction with illumination retrieval


### 3d_ald

*examples_synthetic/3d_ald/modeling.ipynb* - generate data for a 3D ALD synthetic sample 

*examples_synthetic/3d_ald/rec_1step.ipynb - phase retreival by a 1-step method (CTF, MultiPaganin)

*examples_synthetic/3d_ald/rec_iterative.ipynb - iterative reconstruction with illumination retrieval


*3d_ald_syn/rec.ipynb* - reconstruction of the 3D ALD sample with probe retrieval


## Jupyter notebook for experimental data reconstruction

### 2D Siemens star

*examples_experimental/siemens_star/rec_1step.ipynb* - phase retreival by a 1-step method (CTF, MultiPaganin)

*examples_experimental/siemens_star/rec_iterative.ipynb* - iterative reconstruction with illumination retrieval

## Holotomography with coded apertures

### Synthetic 2D Siemens star 

*coded_apertures/siemens_star/modeling.ipynb* - generate data for a siemens star with coded apertures

*coded_apertures/siemens_star/rec_iterative.ipynb* - reconstruction with an iterative scheme

### Synthetic 3D ALD

*coded_apertures/3d_ald/modeling.ipynb* - generate data for a 3d ALD sample star with coded apertures

*coded_apertures/3d_ald/rec_reprojection.ipynb* - joint phase retrieval and tomography reconstruction using the reprojection method




