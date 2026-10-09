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

## Citation

If you use this software, please cite:

> Viktor Nikitin et al., "Scalable joint X-ray nano-holotomography
> reconstruction with the bilinear Hessian method", *Optica* **13**(9),
> 1814–1826 (2026). <https://doi.org/10.1364/OPTICA.603920>

The BibTeX entry is on the [credits](credits) page.

## License

3-clause BSD, with the UChicago Argonne / U.S. Department of Energy
government-rights notice. Copyright (c) 2024-2026, UChicago Argonne, LLC;
produced under U.S. Government contract DE-AC02-06CH11357 for Argonne
National Laboratory. The software is provided "as is", without warranty of
any kind. See [credits](credits) for the summary and
[`LICENSE`](https://github.com/tomography/holotomocupy/blob/master/LICENSE)
for the full text.
