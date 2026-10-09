# Performance Tests

End-to-end timing benchmarks for the holotomography MPI reconstruction on one or more GPUs.

## Files

| File | Purpose |
|---|---|
| `rec_iterative_mpi.py` | Benchmark on real data (config1.conf) |
| `rec_iterative_mpi_syn.py` | Benchmark on synthetic data (config2.conf) |
| `create_h5.ipynb` | Generate a synthetic HDF5 dataset for `rec_iterative_mpi_syn.py` |
| `parse_output.py` | Parse timing output from log files |
| `config1.conf` | Configuration for real data benchmark |
| `config2.conf` | Configuration for synthetic data benchmark |
| `bind.sh` | GPU-to-rank binding for mpirun / SLURM |
| `set_affinity_gpu_polaris.sh` | GPU-to-rank binding for Polaris (ALCF) |
| `prun.sh` | PBS job script for Polaris |

## Setup

### Real data (config1.conf)

Set `in_file` and `path_out` in `config1.conf` to point to your dataset and output directory.

### Synthetic data (config2.conf)

Open `create_h5.ipynb` and run all cells to generate the synthetic dataset, then set `in_file` and `path_out` in `config2.conf`.

## Running

### Local (mpirun)

Real data:
```bash
cd tests/performance/mpi_scaling
mpirun -np <ngpus> ./bind.sh python rec_iterative_mpi.py config1.conf
```

Synthetic data:
```bash
mpirun -np <ngpus> ./bind.sh python rec_iterative_mpi_syn.py config2.conf
```

Example with 4 GPUs:
```bash
mpirun -np 4 ./bind.sh python rec_iterative_mpi.py config1.conf
```

### Polaris (PBS)

Edit `prun.sh` to set project ID, node count, and walltime, then submit:
```bash
qsub prun.sh
```

`prun.sh` runs `rec_iterative_mpi_syn.py config2.conf` by default.

## Configuration

Key parameters to tune for benchmarking:

```ini
nchunk=4       # chunk size — increase until GPU memory is exhausted
ntheta=4500    # number of projection angles (real: max 4500, synthetic: max 18000)
niter=2        # number of iterations (≥2; first iter may be slower due to JIT)
bin=0          # binning: 0 = full res, 1 = 2×, -1 = 2× upsample
err_step=1     # print error every N iters
vis_step=-1    # disable checkpoint saving during benchmarks
```

Recommended `nchunk` on A100:

| `bin` | `nchunk` |
|-------|----------|
| 2     | 32       |
| 1     | 16       |
| 0     | 2–4      |

Dataset sizes (real data, 4 distances):

| `bin` | `ntheta` | Shape |
|-------|----------|-------|
| 1     | 900      | (900, 4, 1024, 1024) |
| 0     | 4500     | (4500, 4, 2048, 2048) |
