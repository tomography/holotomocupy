# HolotomocuPy

GPU-accelerated X-ray holotomography reconstruction with MPI multi-GPU support.

Holotomography is a coherent imaging technique that reconstructs the 3-D complex refractive-index distribution of a sample by combining holography with tomography. This package provides iterative algorithms optimized for large datasets at modern synchrotron sources.

## Quick start

```bash
git clone https://github.com/tomography/holotomocupy
cd holotomocupy
conda env create -f environment.yml     # OpenMPI, MPI-enabled h5py, CuPy
conda activate holotomocupy
pip install -e .

tests/unit/run.sh                       # 148 checks, ~1 min on one GPU
cd demo && ./run.sh 01_nfp_probe.py     # a real reconstruction, ~2 min
```

You need one NVIDIA GPU and a CUDA 12 driver (`cupy-cuda13x` in
`environment.yml` for CUDA 13). Nothing else is required: the demo builds its
own phantom and downloads the measured probe on first use, so there is no
beamline data to find. If `tests/unit/run.sh` passes, the install is good.

Full documentation: **https://holotomocupy.readthedocs.io** (built from
[`docs/`](docs/)). See also [`demo/README.md`](demo/README.md) for what the
two demos do and [`tests/README.md`](tests/README.md) for the rest of the
tests.

---

## Key features

- **GPU acceleration** via CuPy (drop-in GPU NumPy)
- **MPI multi-GPU** — one rank per GPU, tested with 1–1024 GPUs — close to linear performance gain
- **`@gpu_batch` decorator** — automatically chunks data that exceeds GPU memory
- **Modular operators** — tomographic projection (R / RT / FBP), Fresnel propagator, B-spline shifts; all reusable independently
- **Bilinear-Hessian (BH) solver** — joint optimization of object, probe, and sample positions
- **Checkpointing** — automatically resumes from the latest saved iteration; each checkpoint also writes a mid-slice TIFF of `obj_re` for quick visual inspection
- **External initial guess** — load a pre-reconstructed `.vol` binary file (e.g. from a standard FBP pipeline) via `init_vol` in the config
- **Position error monitoring** — RMS position errors per distance are logged at every checkpoint
- **Accurate GPU memory reporting** — reports bytes visible to `nvidia-smi` (not just the CuPy pool)
- **Jupyter notebook pipeline** — steps 1–5 for data preparation, step 6 for iterative reconstruction

---

## Installation

### Requirements

| Dependency | Notes |
|---|---|
| CUDA ≥ 11 | One GPU per MPI rank |
| CuPy | GPU NumPy |
| mpi4py | MPI bindings |
| h5py | HDF5 I/O — must be built with parallel (OpenMPI) support |
| tifffile | TIFF I/O for checkpoint slices |
| nvtx | NVIDIA NVTX profiling markers (used in the BH solver loop) |
| dxchange | Tomography I/O utilities |
| matplotlib | Visualization in notebooks |
| NVIDIA mathDX *(optional)* | Enables the cuFFTDx-based fast Fresnel propagator |

### Optional: cuFFTDx fast propagator (NVIDIA mathDX)

The Fresnel propagator (`propagation.py`) has two backends:

| Backend | Speed | Requirement |
|---|---|---|
| cuPy (default) | baseline | none — works out of the box |
| cuFFTDx | faster | NVIDIA mathDX + nvcc |

The cuFFTDx backend is selected automatically at startup when mathDX is found. If it is unavailable, the package silently falls back to cuPy with no loss of correctness.

**1. Download and install mathDX**

Follow the installation guide at https://docs.nvidia.com/cuda/mathdx/installation.html to download and unpack the mathDX package:

```bash
tar -xzf nvidia-mathdx-*.tar.gz -C /opt/nvidia
```

**2. Set environment variables**

```bash
export MATHDX_ROOT=/opt/nvidia/nvidia-mathdx-25.12.1-cuda13/nvidia/mathdx/25.12
export NVCC=/usr/local/cuda/bin/nvcc   # or wherever nvcc lives
# Optional overrides:
# export CUFFTDX_SM=80          # target SM version (default: 80)
# export CUFFTDX_SO_DIR=/tmp    # where JIT-compiled .so files are cached
```

Add these lines to your `~/.bashrc` or the job script so they persist.

**3. Verify detection**

```python
from holotomocupy.propagation import Propagation
# Should print: "cuFFTDx (mathDX) available — using fast cuFFTDx propagator."
```

If mathDX is not found you will see a `UserWarning` explaining which path is missing.

**JIT compilation and MPI**

The first time a new grid size is used, the package JIT-compiles a small CUDA shared library with `nvcc` and caches it in `CUFFTDX_SO_DIR`. In an MPI run, **only rank 0 compiles**; all other ranks wait at a barrier and then load the pre-built library. Subsequent runs reuse the cached `.so` and skip compilation entirely.

### Environment

```bash
conda env create -f environment.yml
conda activate holotomocupy
pip install -e .
```

[`environment.yml`](environment.yml) pins the two things that are easy to get
wrong: **OpenMPI**, and an **MPI-enabled h5py** (`h5py=*=mpi_openmpi_*`) --
`Writer` opens its checkpoints with `driver='mpio'`, and the plain serial h5py
fails there. `mpi4py` is imported by the logger, so it is needed even for a
single-GPU run. [`requirements.txt`](requirements.txt) is the pip-only list,
but you still have to supply MPI and a parallel h5py yourself.

---

## Reconstruction pipeline

**Start with [`demo/`](demo/)** — the whole chain on synthetic data, two
notebooks for one GPU with MPI scripts beside them, no beamline data needed.
A complete example on real data is in `experimental/Y350a_dist1234/`. The
pipeline has two stages:

```bash
cd experimental/Y350a_dist1234

# Stage 1 — data preparation (single node, steps 0–5)
python step0.py config_step0.conf          # NFP probe calibration (optional)
python steps15.py config_steps15.conf      # steps 1–5: convert, preprocess, shifts, Paganin

# Stage 2 — iterative reconstruction (multi-node/multi-GPU)
mpirun -np 4 ./bind.sh python step6.py config_step6.conf
```

### Step 0 — NFP probe calibration (optional)

Near-field ptychography (NFP) reconstruction of the illumination probe. Writes a probe HDF5 file that step 6 uses as its starting probe instead of a flat-field estimate.

### Steps 1–5 — Data preparation (`steps15.py`)

`steps15.py` runs all data preparation with MPI across multiple nodes and GPUs:

- **Step 1** — reads raw EDF detector frames in parallel, writes a single HDF5 file with all distances, flat/dark fields, encoder shifts, and beam-monitor attributes
- **Step 2** — outlier removal (median-filter spike detection) and intensity normalisation per projection (GPU)
- **Step 3** — combines all shift sources into `cshifts_final`: encoder shifts from `correct.txt`, inter-plane alignment from Peter's RHAPP pipeline (`rhapp.mat`), slow-drift motion correction (`correct_motion.txt`), and optional 3-D tomographic correction (`correct_correct3D.txt`)
- **Step 4** — multi-distance back-projection onto the object plane at multiple bin levels; includes amplitude normalisation across distances
- **Step 5** — multi-distance Paganin phase retrieval followed by FBP reconstruction at all bin levels to produce the initial object guess for step 6

### Step 6 — Iterative MPI reconstruction

Joint iterative refinement of object, probe, and sample positions using the Bilinear-Hessian (BH) algorithm. Scales across multiple nodes and GPUs — one MPI rank per GPU:

```bash
mpirun -np <ngpus> ./bind.sh python step6.py config_step6.conf
```

**Startup order of priority for initial object:**

1. Latest checkpoint in `path_out` (automatic resume)
2. External `.vol` file specified by `init_vol` in the config
3. Paganin reconstruction written by step 5

### Step 7 — Per-angle drift refinement (optional)

`step7.py` re-projects a step-6 checkpoint and searches for the per-angle
shift that minimises the entropy of an FBP reconstruction — one GPU, no MPI.
It writes `correct_correct3D_extra.txt`, which step 6 adds to the positions it
reads (`correct3d_extra` in the config), so refining the alignment costs a
step-6 rerun and not a `steps15.py` one.

```bash
python step7.py config_step6_bin2.conf
mpirun -np <ngpus> ./bind.sh python step6.py config_step6_bin2.conf   # again
```

The algorithm is `holotomocupy.autofocus`; `tests/find_shifts_extra` tests it
and `tests/find_shifts_extra/doc/nelder_mead.pdf` writes it up. It currently
exists for the AtomiumS1, AtomiumS1_HT, ctxl_FT and ctxl_HT datasets.

---

## Running on Polaris (ALCF)

Polaris is an A100 cluster at Argonne Leadership Computing Facility (ALCF). Each node has **4 A100 GPUs**. Ready-to-use job scripts are in each `experimental/<dataset>/` folder, and they all source [`experimental/polaris_env.sh`](experimental/polaris_env.sh).

### Python environment

Polaris uses a shared conda base environment loaded via modules. Create a per-user virtual environment on top of it so you can install packages:

```bash
module use /soft/modulefiles; module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="$HOME/venvs/${CONDA_NAME}"
mkdir -p "${VENV_DIR}"
python -m venv "${VENV_DIR}" --system-site-packages
source "${VENV_DIR}/bin/activate"
pip install -e /path/to/holotomocupy
```

`experimental/polaris_env.sh` does exactly this inside the job, plus the Lustre / MPI-IO settings parallel HDF5 needs on `/eagle`. Override the venv it picks with `HTC_VENV`, and set `HTC_ENV_CHECK=1` to print what resolved.

See [ALCF Python docs](https://docs.alcf.anl.gov/polaris/data-science/python/) for more details.

### GPU affinity

`polaris/set_affinity_gpu_polaris.sh` assigns GPUs in reverse order to match the Polaris PCIe topology (see [ALCF machine overview](https://www.alcf.anl.gov/support/user-guides/polaris/hardware-overview/machine-overview/index.html)):

```bash
#!/bin/bash -l
num_gpus=4
gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
exec "$@"
```

### PBS job script

`polaris/prun.sh` — submit with `qsub polaris/prun.sh` from the experiment directory:

```bash
#!/bin/bash
#PBS -A <project_id>
#PBS -l select=256:system=polaris   # number of nodes (4 GPUs each)
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=00:40:00
#PBS -q prod

cd $PBS_O_WORKDIR

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=4          # 4 MPI ranks per node = 4 GPUs per node
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

# NCCL networking settings for Polaris HPE Slingshot interconnect
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn1

# Activate conda + venv
module use /soft/modulefiles; module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
source "$HOME/venvs/${CONDA_NAME}/bin/activate"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} \
    --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} \
    ./set_affinity_gpu_polaris.sh \
    python step6.py config_step6.conf
```

### Typical workflow on Polaris

```bash
# 1. Log in
ssh <user>@polaris.alcf.anl.gov

# 2. Create the virtual environment (one-time setup)
module use /soft/modulefiles; module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="$HOME/venvs/${CONDA_NAME}"
mkdir -p "${VENV_DIR}"
python -m venv "${VENV_DIR}" --system-site-packages
source "${VENV_DIR}/bin/activate"

# 3. Clone and install
git clone https://github.com/tomography/holotomocupy
cd holotomocupy
pip install -e .

# 4. Copy input data to eagle filesystem
cp /path/to/data.h5 /eagle/<project>/data.h5

# 5. Edit the config to point to eagle paths
cd experimental/Y350a_dist1234
vi config_step6.conf   # set in_file and path_out

# 6. Copy the Polaris scripts next to your step6 script
cp ../../polaris/set_affinity_gpu_polaris.sh .

# 7. Edit prun.sh: set project ID, node count, config path
vi ../../polaris/prun.sh

# 8. Submit
qsub ../../polaris/prun.sh

# 9. Monitor
qstat -u $USER
tail -f <jobid>.o
```

### Scaling

4 GPUs per node × N nodes = 4N total ranks. Typical settings:

| Nodes | Total GPUs | `nchunk` | Dataset size |
|-------|-----------|----------|--------------|
| 1 | 4 | 8 | small / debug |
| 16 | 64 | 8 | medium |
| 256 | 1024 | 8 | full 2500-angle 3D dataset |

---

## Configuration file (`config_step6.conf`)

All parameters as `key=value`; inline comments with `#`:

```ini
in_file=/data/dataset.h5       # input HDF5 file
path_out=/data/results         # output directory for checkpoints and results
prb_file=/data/nfp_results.h5  # probe initial guess from step 0 (optional)
init_vol=/data/rec.vol         # external binary .vol initial guess (optional)

# Data dimensions
ntheta=2500                    # number of projection angles
nz=1024                        # vertical detector size (pixels)
n=1024                         # horizontal detector size (pixels)
nzobj=1632                     # vertical object size (pixels)
nobj=1632                      # horizontal/lateral object size (pixels)
ndist=4                        # number of propagation distances
bin=1                          # binning level (0 = full resolution, 1 = 2×, ...)

# Solver
niter=257                      # number of BH iterations
nchunk=8                       # projection chunk size (tune to GPU memory)
start_iter=0                   # resume from this iteration (0 = fresh start)
err_step=8                     # compute/print error every N iterations (-1 = never)
vis_step=8                     # save checkpoint every N iterations (-1 = never)

# Physics
energy=17.23                   # X-ray energy (keV)
paganin=20                     # Paganin regularization constant
rotation_center_shift=-8.78    # rotation center offset from detector center (pixels)
mask=1.1                       # tomographic mask radius (fraction of detector half-width)

# Regularization
lam_prbfit=3.1e-3              # probe fit weight
lam_laplacian=0                # 3-D Laplacian regularization weight
rho=1,0.05,0.02                # step-size scaling for object, probe, positions

# Misc
start_theta=0                  # first angle index
log_level=WARNING              # DEBUG / INFO / WARNING / ERROR
pos_checkpoint=                # override positions from a checkpoint file (optional)
```

### `init_vol` — external initial object

When `init_vol` is set, the solver reads a raw binary file as the starting object instead of the Paganin reconstruction. The expected format is a C-order `float32` flat binary array of shape `nzobj·2^b × nobj·2^b × nobj·2^b` where `b ≥ 0` is inferred automatically from the file size. When `b > 0`, block-averaging downsampling is applied. Values are normalized by `nobj/4` to match the internal reconstruction scale.

---

## Checkpoint outputs

At each `vis_step` interval (when `vis_step != -1` and `i > start_iter`), the following files are written to `path_out`:

| File | Content |
|---|---|
| `checkpoint_{iter:04}.h5` | Full object (real + imag), probe, positions |
| `checkpoint_{iter:04}_obj_re.tiff` | Middle z-slice of `obj_re`, `(nobj, nobj)`, for a quick visual check |
| `checkpoint_{iter:04}_obj_re_vert.tiff` | Middle y-slice of `obj_re`, `(nzobj, nobj)` -- the vertical cut through the rotation axis |

The log also records the mean absolute position error per distance at each checkpoint:

```
iter=256: pos mean abs error [px]  d0:(0.023,0.019)  d1:(0.026,0.020)  d2:(0.030,0.024)  d3:(0.031,0.026)
```

---

## Running tests

Tests are scripts, one folder per subject; each prints its own verdict and
exits non-zero on failure. [`tests/README.md`](tests/README.md) lists all of
them with what they need and how long they take.

```bash
tests/unit/run.sh                                    # start here, ~1 min
tests/find_shifts_extra/run.sh                       # folders with a run.sh
PYTHONPATH=src python tests/adjoint/test_adjoint_f1.py
```

[`tests/unit/`](tests/unit/) is the one to run after any change to the solver:
imports, GPU, MPI and parallel HDF5; the adjointness of every linear operator;
and the gradients and Hessians of the data misfit by Taylor test — 148
checks in about a minute on one GPU.

---

## Package layout

```
src/holotomocupy/
    rec_mpi.py          # BH iterative solver (MPI-aware)
    rec_nfp_mpi.py      # near-field ptychography probe calibration solver (MPI-aware)
    tomo.py             # tomographic projection: R, RT, FBP (ramp/shepp/parzen), rec_tomo CG
    shift.py            # B-spline sub-pixel shift operators (S, S*, curlyS, derivatives)
    shift_fft.py        # the same interface via the Fourier shift theorem
    autofocus.py        # entropy autofocus for the per-angle drift (step 7)
    psf.py              # Gaussian detector PSF
    paganin.py          # multi-distance Paganin phase retrieval (step 5, demo)
    extra_terms.py      # optional regularisation terms (probe fit, Laplacian)
    esrf_meta.py        # ESRF drop-folder metadata: bin factors, file lookup
    propagation.py      # Fresnel propagator (cuFFTDx or cuPy backend)
    conv2d_cufftdx.py   # cuFFTDx JIT wrapper + availability flag
    cuda/conv2d.cu      # cuFFTDx 2-D convolution kernel source
    chunking.py         # @gpu_batch decorator — auto-chunks over GPU memory limit
    cuda_kernels.py     # raw CUDA kernels (spline interpolation, NUFFT gather/scatter)
    reader.py           # MPI-aware HDF5 reader, raw .vol binary reader, Octave mat loader
    writer.py           # MPI-aware HDF5 writer / checkpointing + TIFF slice output
    mpi_functions.py    # MPI collective helpers (allreduce, redistribute)
    config.py           # configuration file parser (step 6 and steps 1–5)
    utils.py            # GPU/CPU memory utilities, visualization helpers, timer decorator
    logger_config.py    # colored MPI-aware logger

demo/                   # the pipeline end to end on synthetic data, 1 or N GPUs

experimental/
    polaris_env.sh      # modules + venv for ALCF Polaris, sourced by every job
    Y350a_dist1234/     # brain dataset pipeline (steps 0–6, 4 distances)
    AtomiumS2/          # Atomium S2 dataset pipeline (steps 0–6, 4 distances)
    y350a_80um/         # y350a 80 µm dataset pipeline (steps 0–6, 4 distances)

docs/                   # Sphinx sources for readthedocs.io

tests/                  # 17 folders; see tests/README.md
    unit/               # run this one after any change to the solver
    performance/        # benchmarks, incl. mpi_scaling/ (moved from experimental/)
```

---

## Citation

If you use this software, please cite:

> Viktor Nikitin et al., "Scalable joint X-ray nano-holotomography
> reconstruction with the bilinear Hessian method", *Optica* **13**(9),
> 1814–1826 (2026).
