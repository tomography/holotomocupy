# holotomocupy

GPU-accelerated X-ray holotomography reconstruction with MPI multi-GPU support.

Holotomography reconstructs the 3-D complex refractive index of a sample by
combining holography with tomography. This package implements the forward
model and a bilinear-Hessian solver that refines **object, probe and sample
positions jointly**, on one GPU or on hundreds.

```{code-block} bash
:caption: From clone to a real reconstruction

git clone https://github.com/tomography/holotomocupy
cd holotomocupy
conda env create -f environment.yml
conda activate holotomocupy
pip install -e .

tests/unit/run.sh                       # 148 checks, ~1 min on one GPU
cd demo && ./run.sh 01_nfp_probe.py     # ~2 min
```

One NVIDIA GPU and a CUDA 12 driver is all you need. The demo builds its own
phantom and fetches the measured probe on first use, so there is no beamline
data to find.

```{toctree}
:maxdepth: 2
:caption: Guide

installation
quickstart
pipeline
testing
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
credits
```

## Where things live

| | |
|---|---|
| [`src/holotomocupy/`](api) | the package: operators, solvers, I/O |
| `demo/` | the whole chain on synthetic data, 1 or N GPUs |
| `experimental/<dataset>/` | one folder per beamtime: configs + `step0/steps15/step6/step7` |
| `tests/` | 17 folders, scripts not pytest; start at `tests/unit` |
