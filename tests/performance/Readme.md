# Performance benchmark — multi-distance holotomography

Fully-synthetic perf test for `holotomocupy.rec_mpi.Rec` and a log parser
that summarises one BH iteration.

## Files

| file | purpose |
|------|---------|
| `test.py`              | the benchmark — generates positions + random data, runs `Rec.BH()`, writes a perf log |
| `test_mosaic.py`       | the same benchmark for a MOSAIC scan (default 1 × 5 tiles × 4 distances = 20 distances over one wide object; `--ntile-v 2` gives 40) |
| `run_mosaic.sh`        | runs `test_mosaic.py` at one binning level under `mpirun`, then parses the log |
| `run.sh`               | the same for `test.py`, at one detector size |
| `run_mosaic_polaris.sh` / `run_polaris.sh` | the same two runs as **PBS job scripts for Polaris** — `qsub` them; `mpiexec`/PALS launch and `nchunk` sized for a 40 GB card |
| `set_affinity_gpu.sh`  | per-rank GPU pinning (`CUDA_VISIBLE_DEVICES = local_rank % ngpus`; Open MPI / SLURM / PMI) |
| `set_affinity_gpu_polaris.sh` | the Polaris variant: `PMI_LOCAL_RANK`, GPUs numbered opposite the CPU NUMA nodes |
| `parse_perf_log.py`    | reads the perf log; reports per-iter timing breakdown + max process / GPU memory |

All four drivers carry their settings as plain assignments at the top of the
file — size, rank count, `nchunk` — and take no environment variables. **Edit
the script, then run (or `qsub`) it.** One size per run, so a sweep is a few
edits or a few copies of the file.

> **Running this on a cluster?** Jump to [Running on a cluster](#running-on-a-cluster)
> — both benchmarks, node counts, `nchunk` / `ndistchunk` per size, and
> the memory budget per rank.

## Quick start

Single rank:
```bash
python test.py --n 2048 --ntheta 1800 --nchunk 4
python parse_perf_log.py perf.log
```

Multi-rank with one GPU per local rank (round-robin pinning):
```bash
mpirun -np 8 ./set_affinity_gpu.sh \
    python test.py --n 2048 --ntheta 1800 --nchunk 4 --log log2048_8
python parse_perf_log.py log2048_8
```

Or use the driver, which does both and prints the breakdown — set `NP`, `N`,
`NCHUNK` at the top of the file first:
```bash
./run.sh --plan                # sizes + memory for those settings, no GPU, no run
./run.sh                       # run it
```

## `test.py` CLI

| flag       | default      | meaning |
|------------|--------------|---------|
| `--n`      | **required** | detector size (square: `nz = n`) |
| `--ntheta` | **required** | number of projection angles |
| `--nchunk` | `4`          | theta chunk size for the batched GPU ops — main perf knob |
| `--ndistchunk` | `0` (= all) | distances sharing one upload of a theta chunk of `proj`; `1` restores the old outer-distance loop |
| `--plan`   | off          | print the per-rank host/GPU budget and exit without allocating |
| `--nranks` | actual size  | rank count to assume in `--plan` |
| `--log`    | `perf.log`   | output log path on rank 0 (`''` disables file logging) |

Other knobs (object padding, `ndist`, `niter`, dtype, regularization, …)
are hardcoded at the top of `test.py` — edit there. Defaults are:
`ndist = 4`, `niter = 1`,
`nobj = 3264·n/2048` (scales linearly with `n`, matches the brain-Y350 config),
`lam_prbfit = lam_laplacian = 0`, `checkpoint_step = -1`, `error_step = 1`.

Nothing is forward-modelled, the same as `test_mosaic.py`: the object stays at
the zero `Rec` allocated, the probe is flat, and `data` is filled with one
random frame per distance modulated by a per-angle scalar. BH runs a fixed
number of iterations regardless of the values, so this times identically to
real data while skipping the object synthesis and the `gen_data` forward
pass — which at `n = 8192` is the difference between a long startup and none.

The synthesised inputs (positions, data) are deterministic across machines and
rank counts — seeded from a single `MASTER_SEED` via
`numpy.SeedSequence.spawn()` per global theta index.

## Mosaic benchmark — `test_mosaic.py`

Emulates a reconstruction of the mosaic scan implemented in
`tests/mosaic_brain` (`gen_data.py` → `steps15.py` → `step6.py`), without any
data on disk. The tiles are flattened onto the distance axis exactly as
`mosaic_reader.MosaicReader` presents them to the solver:

```
ndist = ntiles * ndist_tile,   flat index = tile*ndist_tile + k
z1    = tile(z1_tile, ntiles)
```

so the default 1 × 5 tiles × 4 distances give `ndist = 20` over one wide object
grid, and each tile's place on the mosaic lives in `vars['pos']`.
`--ntile-v 2` gives the two-row mosaic, `ndist = 40`; both shapes are worth
benchmarking and `run_mosaic.sh` has `NTILE_V` / `NTILE_H` at the top.

Nothing is forward-modelled — the point is the cost of one BH iteration at this
size, and BH runs a fixed number of iterations regardless of the values:

| input | value |
|-------|-------|
| `prb`  | `1` (flat probe, as a from-scratch `step6` starts) |
| `ref`  | `\|D·prb\|`, so the probe-fit regularizer has something to fit |
| `data` | random, positive (one random frame per distance × a per-angle scale) |
| `pos`  | tile offset + random per-angle encoder jitter (±30 detector px) |
| `obj`  | `0` — a from-scratch start, as a production `step6` run does |

Sizes follow the detector: at `ndet = 2048` the scan has 6000 angles (the
`ntheta` of `mosaic_brain/config_gen.conf`), and both scale with the binning
level, so `--bin b` gives

```
n = nz = 2048 >> b      ntheta = 6000 >> b      nobj, nzobj = mosaic >> b
```

The object grid is derived from the tile layout itself — outermost tile offset
+ jitter + the half-FOV of the least-magnified distance, rounded up — so it
always holds the whole mosaic. At bin 0 that is `4096 × 12288 × 12288` for the
default 1 × 5 layout and `6144 × 12288 × 12288` for 2 × 5 (a second tile row
makes the object taller, not wider). Acquisition geometry, tile pitch, jitter,
`mask`, `rho` and `lam_prbfit` are copied from
`tests/mosaic_brain/config_gen.conf` + `config_step6.conf`, so the two are
directly comparable.

### Plan first — host memory is the limit

```bash
python test_mosaic.py --bin 2 --nchunk 8 --nranks 8 --plan
```

prints the pinned-host and (approximate) GPU footprint per rank without
allocating anything or touching a GPU. Rough pinned-host totals over all ranks
(`obj` + `proj` + `proj_tmp` + `data`):

| bin | detector | angles | object (1×5) | host total | object (2×5) | host total |
|-----|----------|--------|--------------|------------|--------------|------------|
| 3   | 256      | 750    | 512 × 1536²   | 48 GB  | 768 × 1536²   | 74 GB  |
| 2   | 512      | 1500   | 1024 × 3072²  | 386 GB | 1536 × 3072²  | 594 GB |
| 1   | 1024     | 3000   | 2048 × 6144²  | 3.0 TB | 3072 × 6144²  | 4.6 TB |
| 0   | 2048     | 6000   | 4096 × 12288² | 24 TB  | 6144 × 12288² | 37 TB  |

`host total` is summed over all ranks, so it divides by the rank count — that
is what sets the minimum node count. GPU per rank does *not* divide; at
`nchunk = 4` it is 0.8 / 3.3 / 13.2 / 52.7 GB (1×5) and 0.9 / 3.5 / 13.9 /
55.7 GB (2×5), which is why bin 0 has to drop to `nchunk = 1`.

The GPU side is comfortable by comparison up to bin 1, so the rank count is set
by RAM, not by the devices. See
[Running on a cluster](#running-on-a-cluster) for concrete node counts.

### Running

Set `NP`, `BIN`, `NTILE_V` / `NTILE_H` and `NCHUNK` at the top of
`run_mosaic.sh`, then:

```bash
./run_mosaic.sh --plan       # sizes + memory for those settings, no GPU
./run_mosaic.sh              # run + parse
```

or directly:

```bash
mpirun -np 8 ./set_affinity_gpu.sh \
    python test_mosaic.py --bin 2 --nchunk 8 --log logmosaic2_8
python parse_perf_log.py logmosaic2_8
```

The log format is identical to `test.py`'s, so `parse_perf_log.py` works
unchanged. Before iterating, rank 0 logs where every tile window lands on the
object grid — the same placement check `mosaic_brain/step6.py` prints.

### Memory

`@timer` records GPU memory as it happens to stand at the end of each timed
call, and `parse_perf_log.py` reports the maximum it saw. That is enough to
compare runs; transient peaks inside a call (cuFFT work areas, the scratch in
`Shift`/`Tomo`) and the allocation phase are not covered. For an up-front
estimate, run with `--plan`, which prints the pinned-host and GPU budgets per
rank without allocating anything.

### `test_mosaic.py` CLI

| flag | default | meaning |
|------|---------|---------|
| `--bin`        | **required** | `n = nz = 2048>>bin`, `ntheta = 6000>>bin` |
| `--nchunk`     | `4`   | theta chunk size — main perf knob |
| `--ndistchunk` | `0` (= all) | distances sharing one upload of a theta chunk of `proj`; `1` restores the old outer-distance loop |
| `--ntile-h`    | `5`   | tiles per row |
| `--ntile-v`    | `1`   | tile rows |
| `--ndist-tile` | `4`   | distances per tile (max 4) |
| `--niter`      | `1`   | BH iterations; with 2+ the last one excludes the JIT compile |
| `--ntheta`     | auto  | override the `6000>>bin` angle count |
| `--nobj` / `--nzobj` | auto | override the object grid (binned px) |
| `--plan`       | off   | print sizes + memory and exit |
| `--nranks`     | MPI size | rank count to assume in `--plan` |
| `--log`        | `perf_mosaic.log` | log path on rank 0 |

## Running on a cluster

Written for a node with **8 GPUs (H100, 80 GB), 2 TB of host RAM**. Everything
below assumes **one MPI rank per GPU**, i.e. 8 ranks per node. For **Polaris**
(4 × A100 40 GB, 512 GB per node, PBS) the node counts and `nchunk` values are
different enough to have their own tables — see
[Polaris](#polaris-4--a100-40-gb-512-gb-per-node).

There are **two benchmarks and both are wanted**. They are independent; run
them in either order.

| | script | driver | geometry | what it stresses |
|---|---|---|---|---|
| **A. Mosaic** | `test_mosaic.py` | `run_mosaic.sh` | a row (or two rows) of tiles at 4 distances each, flattened onto one distance axis over a single wide object | the workload the package is being optimized for; wide object, many distances |
| **B. Single tile** | `test.py` | `run.sh` | one tile, 4 distances, square object, swept over detector sizes 512 → 8192 | how the solver scales with raw detector size |

Both run `Rec.BH()` — the Bilinear-Hessian conjugate-gradient solver — on
synthetic data and time every internal stage. Nothing is read from disk and no
result is written; probe, positions and data are generated in memory from a
fixed seed, so a run is reproducible and depends only on the size knobs. The
point is the timing breakdown, not the reconstruction.

### Setup

```bash
pip install -e .                      # from the repo root
```
Needs `cupy` (matching the CUDA toolkit), `mpi4py` built against the site MPI,
plus `numpy`, `h5py`, `psutil`, `nvtx`, `tifffile`, `matplotlib`, `pandas`.
The CUDA kernels ship as source and are JIT-compiled by CuPy on first use —
iteration 0 therefore absorbs that compile. Both benchmarks default to
`--niter 1`, so the one reported iteration *includes* the JIT cost. Append
`--niter 2` to a driver invocation (`./run.sh --niter 2`) — extra arguments go
straight to the benchmark — to get a clean steady-state number; the parser
always breaks down the *last* iteration in the log, so nothing else changes.
At the largest sizes one iteration is already expensive: keep `--niter 1`
there and note it in the report.

### Launching

`set_affinity_gpu.sh` pins each rank to one GPU via
`CUDA_VISIBLE_DEVICES = local_rank % ngpus`. It reads the Open MPI, the SLURM
or the PMI/PALS rank variables, so all three launchers work (and it aborts
rather than stacking every rank on GPU 0 if it recognises none of them):

```bash
# SLURM
srun -N 4 --ntasks-per-node 8 --gpus-per-task 1 \
     ./set_affinity_gpu.sh python test_mosaic.py --bin 1 --nchunk 4 --log logmosaic1x5_1_32

# Open MPI
mpirun -np 32 --map-by ppr:8:node ./set_affinity_gpu.sh \
     python test_mosaic.py --bin 1 --nchunk 4 --log logmosaic1x5_1_32
```

Each run writes its log on rank 0 and **appends its own parsed summary to the
end of that log**, so the log file alone is the deliverable — send those back.

`run.sh` / `run_mosaic.sh` call `mpirun -np $NP` directly. If your site starts
ranks another way, edit that one line — `srun -n $NP`, `mpiexec -n $NP`, … The
Polaris pair (`run_polaris.sh`, `run_mosaic_polaris.sh`) already use
`mpiexec … --cpu-bind depth` with the PALS affinity wrapper.

### Copy-paste: complete job scripts (SLURM)

Everything below is for **8 × H100 80 GB + 2 TB RAM per node, 1 rank per GPU**.
Adjust only the `module load` / `activate` lines, which are site-specific.

The drivers carry one configuration each, so a multi-size sweep is either a few
sequential edits or a few copies of the driver. In a batch script it is simpler
to call the benchmark directly with the numbers from the tables below — that is
what the jobs here do; `run.sh` / `run_mosaic.sh` remain the interactive path.

**Step 0 — sanity check on a login node.** Allocates nothing and touches no
device (CuPy still has to import, so the CUDA libraries must be visible), and
catches a bad `nchunk` before it costs you an allocation:

```bash
./run_mosaic.sh --plan          # the settings currently in run_mosaic.sh
./run.sh --plan                 # the settings currently in run.sh

# or, without editing anything, straight from the tables:
python test_mosaic.py --bin 1 --ntile-v 2 --nchunk 4 --nranks 64 --plan
python test.py --n 4096 --ntheta 3600 --nchunk 4 --nranks 64 --plan
```

Compare the `TOTAL` lines against the tables below. Remember the GPU total is a
lower bound — budget **3×** it.

**Job 1 — mosaic, both tile shapes, bins 3 and 2 (1 node).**

```bash
#!/bin/bash
#SBATCH --job-name=htc-mosaic-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

module load cuda openmpi                    # site-specific
source /path/to/venv/bin/activate           # site-specific
cd "$SLURM_SUBMIT_DIR"

run () {                                    # bin, ntile_v, nchunk
    log="logmosaic${2}x5_${1}_$SLURM_NTASKS"
    srun -n "$SLURM_NTASKS" ./set_affinity_gpu.sh \
        python test_mosaic.py --bin "$1" --ntile-v "$2" --ntile-h 5 \
                              --nchunk "$3" --niter 2 --log "$log"
    python parse_perf_log.py "$log"
}

for v in 1 2; do
    run 3 $v 16
    run 2 $v 16
done
```

**Job 2 — mosaic, bin 1 (host-memory bound; 4 nodes minimum, 8 comfortable).**
Same header with `--nodes=8`, same `run ()`, then:

```bash
for v in 1 2; do run 1 $v 4; done
```

**Job 3 — mosaic, bin 0, the full production size (`--nodes=32` minimum for
both shapes; 64 halves the per-node RAM).** This one only tolerates
`nchunk = 1`, so treat it as a feasibility run, not a throughput number, and
leave it at one iteration:

```bash
for v in 1 2; do
    log="logmosaic${v}x5_0_$SLURM_NTASKS"
    srun -n "$SLURM_NTASKS" ./set_affinity_gpu.sh \
        python test_mosaic.py --bin 0 --ntile-v $v --ntile-h 5 \
                              --nchunk 1 --log "$log"
    python parse_perf_log.py "$log"
done
```

At 32 nodes that is 793 GB/node (1×5) and 1227 GB/node (2×5) — both fit 2 TB,
the second with less room to spare.

**Job 4 — single tile, 512 / 1024 / 2048 (1 node).**

```bash
#!/bin/bash
#SBATCH --job-name=htc-single-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

module load cuda openmpi
source /path/to/venv/bin/activate
cd "$SLURM_SUBMIT_DIR"

run () {                                    # n, nchunk, [extra flags]
    n=$1; nc=$2; shift 2
    log="log${n}_${nc}"
    srun -n "$SLURM_NTASKS" ./set_affinity_gpu.sh \
        python test.py --n "$n" --ntheta "$(( 1800 * n / 2048 ))" --nchunk "$nc" \
                       --niter 2 --log "$log" "$@"
    python parse_perf_log.py "$log"
}

run 512   8
run 1024 16
run 2048 16
```

n = 2048 sits at 1464 GB of 2 TB on one node — if the queue allows, give it
`--nodes=2` and it drops to 735 GB (`nchunk` stays 16; GPU memory does not
divide by node count).

**Job 5 — single tile, 4096 (8 nodes) and 8192 (64–128 nodes).** 8192 is the
awkward one: it is GPU-bound, not host-bound, and needs `--ndistchunk 2` to fit
an 80 GB card at all.

```bash
run 4096 4                                  # --nodes=8
run 8192 1 --ndistchunk 2                   # --nodes=64, 1563 GB/node
```

If even 64 nodes is out of reach, run 8192 with fewer angles —
`python test.py --n 8192 --ntheta 1800 --nchunk 1 --ndistchunk 2 ...` — and say
so in the report. The breakdown is still meaningful; the absolute time is just
not comparable to the rest of the sweep.

**Collecting the results.** Every run appends its own parsed summary to its own
log, so:

```bash
tar czf perf-logs-$(hostname -s).tar.gz log* 
```

### The two knobs

|  | drives | scales with |
|---|---|---|
| `--nchunk` | **GPU** memory | `nchunk × nzobj × nobj` — the *global* object grid |
| `--ndistchunk` | how many distances share one upload of a `proj` chunk | detector plane, `ndistchunk × nz × n` |

The important and slightly counter-intuitive part: **adding nodes buys host
memory, not GPU memory.** Host arrays are split over ranks, so pinned RAM per
rank falls as `1/nranks`. The GPU chunking pool is sized from `nzobj` and
`nobj`, which are global and do not shrink. So `nchunk` is essentially a
per-size constant — if you add nodes to fit a bigger problem in RAM, do *not*
expect to raise `nchunk` too.

**`--ndistchunk` matters for benchmark B and not for benchmark A**, and the
reason is worth stating because it is not obvious. The chunking pool holds
`max(3 × proj_chunk + ndistchunk × dist_chunk, 3 × obj_chunk)`. Distances only
add to the *first* term, so as long as that term stays under `3 × obj_chunk`
the distances are free. Setting the two equal gives the break-even:

```
ndist* = 6 · nobj · (nobj − nzobj) / (nz · n)          (independent of nchunk and of bin)
```

* **Mosaic**: the object is wider than it is tall (`nzobj < nobj`), so there is
  real headroom — `ndist*` is **144** at 1×5 and **108** at 2×5, against an
  actual `ndist` of 20 and 40. Nothing to tune: **leave `--ndistchunk` at 0.**
* **Single tile**: `nzobj == nobj`, so `ndist* = 0` — there is no headroom at
  all and every resident distance costs from the first one. At small `n` that
  is irrelevant, but at `n = 8192` dropping `--ndistchunk` from 4 to 2 is the
  difference between ~78 GB (does not fit) and ~66 GB (fits).

`--ndistchunk 1` restores the old outer-distance loop. It is 3–4× slower on the
cascades and exists to reproduce the pre-optimization numbers — use it for an
A/B, or as a last-resort memory lever at the largest single-tile size.

### Recommended settings — A. Mosaic

`--bin` sets the size: `n = nz = 2048>>bin` detector pixels and
`ntheta = 6000>>bin` angles. **bin 0 is the full production problem**; bin 3 is
1/8 of it in every dimension and runs on a single node. `lth` is
`local_ntheta` = angles per rank. `plan GPU` is what `--plan` reports; `~actual`
is the realistic figure — see the calibration note below.

`nchunk` is always a power of two here: it is the batch size of every chunked
GPU op, and rounding it to 2^k keeps the chunk count (and the tail chunk)
predictable across sizes. Each entry is the largest power of two that stays
inside both the card and `lth/4`.

Two mosaic shapes are wanted — `NTILE_V=1` and `NTILE_V=2` in `run_mosaic.sh`.
Two rows per bin: the **minimum** node count and one comfortable step up.

#### 1 × 5 tiles — ndist = 20, object 4096 x 12288 x 12288 at bin 0  (ndistchunk crossover 144)
| bin | nodes | ranks | lth | `--nchunk` | `--ndistchunk` | host/rank | host/node | plan GPU | ~actual GPU |
|-----|-------|-------|-----|-----------|----------------|-----------|-----------|----------|-------------|
| 3 | 1 | 8 | 94 | 16 | 20 | 6.1 GB | 49 GB | 3.2 GB | ~10 GB |
|   | 2 | 16 | 47 | 8 | 20 | 3.0 GB | 24 GB | 1.6 GB | ~5 GB |
| 2 | 1 | 8 | 188 | 16 | 20 | 48.4 GB | 387 GB | 12.8 GB | ~38 GB |
|   | 2 | 16 | 94 | 16 | 20 | 24.3 GB | 194 GB | 12.8 GB | ~38 GB |
| 1 | 4 | 32 | 94 | 4 | 20 | 97.0 GB | 776 GB | 13.2 GB | ~40 GB |
|   | 8 | 64 | 47 | 4 | 20 | 48.8 GB | 390 GB | 13.2 GB | ~40 GB |
| 0 | 32 | 256 | 24 | 1 | 20 | 99.2 GB | 793 GB | 14.8 GB | ~44 GB |
|   | 64 | 512 | 12 | 1 | 20 | 50.5 GB | 404 GB | 14.8 GB | ~44 GB |

#### 2 × 5 tiles — ndist = 40, object 6144 x 12288 x 12288 at bin 0  (ndistchunk crossover 108)
| bin | nodes | ranks | lth | `--nchunk` | `--ndistchunk` | host/rank | host/node | plan GPU | ~actual GPU |
|-----|-------|-------|-----|-----------|----------------|-----------|-----------|----------|-------------|
| 3 | 1 | 8 | 94 | 16 | 40 | 9.3 GB | 75 GB | 3.3 GB | ~10 GB |
|   | 2 | 16 | 47 | 8 | 40 | 4.7 GB | 38 GB | 1.7 GB | ~5 GB |
| 2 | 1 | 8 | 188 | 16 | 40 | 74.5 GB | 596 GB | 13.1 GB | ~39 GB |
|   | 2 | 16 | 94 | 16 | 40 | 37.4 GB | 299 GB | 13.1 GB | ~39 GB |
| 1 | 4 | 32 | 94 | 4 | 40 | 149.5 GB | 1196 GB | 13.9 GB | ~42 GB |
|   | 8 | 64 | 47 | 4 | 40 | 75.2 GB | 602 GB | 13.9 GB | ~42 GB |
| 0 | 32 | 256 | 24 | 1 | 40 | 153.4 GB | 1227 GB | 17.2 GB | ~52 GB |
|   | 64 | 512 | 12 | 1 | 40 | 78.6 GB | 629 GB | 17.2 GB | ~52 GB |

Bins 1 and 0 are **host-memory bound**, which is what sets the minimum node
count. Bin 0 only tolerates `nchunk = 1`, so it will be inefficient — treat it
as a feasibility run, not a throughput measurement, and prefer bin 1 for
scaling studies. If you have fewer nodes than the table asks for, cut
`--ntheta` (host memory is close to linear in it) rather than squeezing
`nchunk`: e.g. `--bin 1 --ntheta 750 --nchunk 4` on 1 node.

### Recommended settings — B. Single tile

`n` is the detector size; the object is `nobj = nzobj = 1.59 n` cubed and
`ntheta = 1800 n / 2048` — 1800 angles at n = 2048, halving with n — both set
by `run.sh`. `ndist` is always 4.

| n | nobj | ntheta | nodes | ranks | lth | `--nchunk` | `--ndistchunk` | host/rank | host/node | plan GPU | ~actual GPU |
|---|------|--------|-------|-------|-----|-----------|----------------|-----------|-----------|----------|-------------|
| 512 | 816 | 450 | 1 | 8 | 57 | 8 | 4 | 2.9 GB | 23 GB | 0.6 GB | ~2 GB |
|   |   |   | 2 | 16 | 29 | 4 | 4 | 1.5 GB | 12 GB | 0.3 GB | ~1 GB |
| 1024 | 1632 | 900 | 1 | 8 | 113 | 16 | 4 | 23.0 GB | 184 GB | 4.9 GB | ~15 GB |
|   |   |   | 2 | 16 | 57 | 8 | 4 | 11.6 GB | 93 GB | 2.5 GB | ~8 GB |
| 2048 | 3264 | 1800 | 1 | 8 | 225 | 16 | 4 | 183.0 GB | 1464 GB | 19.6 GB | ~59 GB |
|   |   |   | 2 | 16 | 113 | 16 | 4 | 91.9 GB | 735 GB | 19.6 GB | ~59 GB |
| 4096 | 6528 | 3600 | 8 | 64 | 57 | 4 | 4 | 185.1 GB | 1480 GB | 20.9 GB | ~63 GB |
|   |   |   | 16 | 128 | 29 | 4 | 4 | 93.9 GB | 751 GB | 20.9 GB | ~63 GB |
| 8192 | 13056 | 7200 | 64 | 512 | 15 | 1 | 2 | 195.4 GB | 1563 GB | 22.1 GB | ~66 GB |
|   |   |   | 128 | 1024 | 8 | 1 | 2 | 103.1 GB | 825 GB | 22.1 GB | ~66 GB |

Notes on this sweep:

* 512 / 1024 / 2048 all fit on **one node**. 2048 is the tight one at 1 node
  (1464 GB of 2 TB) — use 2 nodes if the allocation allows.
* **4096 needs 8 nodes**, host-bound.
* **8192 is the hard one** and is GPU-bound, not host-bound: even at
  `nchunk = 1` the object-plane buffers (`nobj = 13056`) dominate. It needs
  `--ndistchunk 2` to fit an 80 GB card at all. 64 nodes is the practical
  minimum (1563 GB/node); 225 nodes is the most that is allowed, since past
  1800 ranks `lth` falls to 3 and rule 2 below rejects it. If 8192 is out of
  reach for your allocation, **run it with a reduced `--ntheta`** and say so in the
  report — the timing breakdown is still meaningful, the absolute numbers just
  are not comparable to the others.

### Polaris (4 × A100 40 GB, 512 GB per node)

Everything above is sized for the 8 × H100 node. Polaris differs in both of the
dimensions that bind here, and the two changes do not cancel:

| | H100 node | Polaris node |
|---|---|---|
| GPUs = ranks per node | 8 × 80 GB | **4 × 40 GB** |
| GPU budget per rank | ~71 GB usable | **~36 GB usable** |
| host RAM per node | 2 TB | **512 GB** |
| host RAM per rank | 256 GB | **128 GB** |
| launcher | SLURM `srun` / `mpirun` | **PBS Pro + `mpiexec` (PALS)** |

Half the GPU memory halves `nchunk`; a quarter of the RAM spread over half the
ranks means the same problem wants roughly **4× the nodes**. The tables below
are the same `--plan` arithmetic re-run against those limits — `nchunk` is the
largest power of two with `plan GPU ≤ 11.9 GB` (≈ 36 GB actual at the 3×
calibration, the same 89 % of the card the H100 tables target) that also stays
inside `lth/4`.

#### The two Polaris job scripts

`run_polaris.sh` (single tile) and `run_mosaic_polaris.sh` (mosaic) are **PBS
job scripts** — `qsub` them. They are the same measurement as `run.sh` /
`run_mosaic.sh`, with the Polaris launch line and Polaris-sized `nchunk`
defaults:

* PBS header at the top: `-l select=… -q … -A …`, `place=scatter`,
  `filesystems`, `walltime`;
* node count from `$PBS_NODEFILE`, `NP = nodes × 4`;
* `mpiexec -n $NP -ppn 4 --cpu-bind depth -d 8 ./set_affinity_gpu_polaris.sh …`
  — `-d 8` gives each of the 4 ranks 8 of the node's 32 cores;
* the size (`N`, or `BIN` + `NTILE_V` / `NTILE_H`) and `NCHUNK` as plain
  assignments below the header, with the recommended values in a comment.

Both scripts source `./env_polaris.sh` if you put one next to them, so the
ranks do not depend on `~/.bashrc`; without that file they run in whatever
environment `qsub` inherits. Either way `MATHDX_ROOT` has to point at a mathDx
install — without it cuFFTDx is missing and propagation falls back to cuPy FFT,
which makes the timings incomparable with the other machines.

To run one, edit two things — `#PBS -l select=` and the size block — and submit
(`#PBS -A` is already the 17445 allocation):

```bash
qsub run_polaris.sh
qsub run_mosaic_polaris.sh
```

`select=` and `NCHUNK` belong together: the tables below give the node count
each size needs. Both scripts also work on a login node for the memory check,
where `NODES=` in the script stands in for the missing job:

```bash
./run_polaris.sh --plan
./run_mosaic_polaris.sh --plan
```

Nothing is allocated and no device is touched (CuPy still imports, so the CUDA
libraries have to be visible).

#### Launching

Polaris runs Cray MPICH under PALS, so neither `OMPI_*` nor `SLURM_*` rank
variables exist — `set_affinity_gpu_polaris.sh` reads `PMI_LOCAL_RANK`, and
because Polaris numbers its GPUs opposite to the CPU NUMA nodes it maps local
rank 0 to GPU 3:

```bash
gpu=$(( 4 - 1 - PMI_LOCAL_RANK % 4 ))
```

It echoes its rank→device choice, so check that line first if timings look
halved: all four ranks on GPU 0 is the failure this wrapper exists to prevent.
(The generic `set_affinity_gpu.sh` also understands `PMI_*`, but not the
reversed numbering, and it aborts if it recognises no rank variable at all.)

Sanity-check the environment before spending a job on the benchmark:

```bash
mpiexec -n 4 --ppn 4 python -c \
  "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.Get_library_version().splitlines()[0])"
```

One more trap: `set_affinity_gpu_polaris.sh` runs under `#!/bin/bash` with **no
`-l`**. A login shell would re-source `~/.bashrc` in every rank, re-run its
module loads and can swap `PrgEnv` — and with it the cray-mpich lib dir — out
from under the environment the job script just built. PALS forwards that
environment to the ranks on its own.

#### Recommended settings — A. Mosaic, 4 ranks/node

Two rows per bin: the **minimum** node count and one comfortable step up.
`lth` = `local_ntheta`; `~actual GPU` is `plan × 3`, against 40 GB.

1 × 5 tiles — ndist = 20:

| bin | nodes (`select=`) | ranks | lth | `NCHUNK` | host/rank | host/node | plan GPU | ~actual GPU |
|-----|-------|-------|-----|-----------|-----------|-----------|----------|-------------|
| 3 | 1 | 4 | 188 | 32 | 12.1 GB | 48 GB | 6.4 GB | ~19 GB |
|   | 2 | 8 | 94 | 16 | 6.1 GB | 24 GB | 3.2 GB | ~10 GB |
| 2 | 1 | 4 | 375 | 8 | 96.6 GB | 386 GB | 6.5 GB | ~19 GB |
|   | 2 | 8 | 188 | 8 | 48.4 GB | 194 GB | 6.5 GB | ~19 GB |
| 1 | 8 | 32 | 94 | 2 | 97.0 GB | 388 GB | 6.9 GB | ~21 GB |
|   | 16 | 64 | 47 | 2 | 48.8 GB | 195 GB | 6.9 GB | ~21 GB |
| 0 | — | — | — | — | — | — | 14.8 GB | **does not fit** |

2 × 5 tiles — ndist = 40:

| bin | nodes (`select=`) | ranks | lth | `NCHUNK` | host/rank | host/node | plan GPU | ~actual GPU |
|-----|-------|-------|-----|-----------|-----------|-----------|----------|-------------|
| 3 | 1 | 4 | 188 | 32 | 18.6 GB | 74 GB | 6.5 GB | ~19 GB |
|   | 2 | 8 | 94 | 16 | 9.3 GB | 37 GB | 3.3 GB | ~10 GB |
| 2 | 2 | 8 | 188 | 8 | 74.5 GB | 298 GB | 6.7 GB | ~20 GB |
|   | 4 | 16 | 94 | 8 | 37.4 GB | 149 GB | 6.7 GB | ~20 GB |
| 1 | 16 | 64 | 47 | 2 | 75.2 GB | 301 GB | 7.5 GB | ~23 GB |
|   | 32 | 128 | 24 | 2 | 38.4 GB | 153 GB | 7.5 GB | ~23 GB |
| 0 | — | — | — | — | — | — | 17.2 GB | **does not fit** |

The 2 × 5 shape needs one more doubling of the node count than 1 × 5 at bins 2
and 1: on one node it would want 594 GB and on eight 598 GB, both over the
512 GB the node has. Leave `--ndistchunk` at 0 as on the H100 — the crossover
(144 at 1×5, 108 at 2×5) is a property of the object aspect ratio, not of the
card.

#### Recommended settings — B. Single tile, 4 ranks/node

| n | nobj | ntheta | nodes (`select=`) | ranks | lth | `NCHUNK` | `--ndistchunk` | host/rank | host/node | plan GPU | ~actual GPU |
|---|------|--------|-------|-------|-----|-----------|----------------|-----------|-----------|----------|-------------|
| 512 | 816 | 450 | 1 | 4 | 113 | 16 | 4 | 5.7 GB | 23 GB | 1.2 GB | ~4 GB |
|   |   |   | 2 | 8 | 57 | 8 | 4 | 2.9 GB | 12 GB | 0.6 GB | ~2 GB |
| 1024 | 1632 | 900 | 1 | 4 | 225 | 32 | 4 | 45.8 GB | 183 GB | 9.7 GB | ~29 GB |
|   |   |   | 2 | 8 | 113 | 16 | 4 | 23.0 GB | 92 GB | 4.9 GB | ~15 GB |
| 2048 | 3264 | 1800 | 4 | 16 | 113 | 8 | 4 | 91.9 GB | 367 GB | 10.0 GB | ~30 GB |
|   |   |   | 8 | 32 | 57 | 8 | 4 | 46.3 GB | 185 GB | 10.0 GB | ~30 GB |
| 4096 | 6528 | 3600 | 32 | 128 | 29 | 2 | 4 | 93.9 GB | 376 GB | 11.3 GB | ~34 GB |
|   |   |   | 64 | 256 | 15 | 2 | 4 | 48.9 GB | 195 GB | 11.3 GB | ~34 GB |
| 8192 | 13056 | 7200 | — | — | — | — | — | — | — | 20.1 GB | **does not fit** |

The single-node `n = 1024` row (`nchunk 32`, ~29 GB of 40) is the tightest
entry in either table; drop to 16 if it OOMs. n = 2048 does not fit on fewer
than 4 nodes (2 nodes would be 732 GB against 512).

#### What does not fit a 40 GB card

Two entries of the H100 sweep have no Polaris equivalent. Both are **GPU**-bound
and adding nodes does not help — the chunking pool is sized from the *global*
object grid.

* **Mosaic bin 0** — the full production size, object `4096 × 12288²`. At
  `nchunk = 1`, already the floor, the plan is 14.8 GB (1×5) / 17.2 GB (2×5),
  i.e. ~44 / ~52 GB. `--ndistchunk 1` only brings 1×5 down to 13.0 GB (~39 GB),
  because on this geometry distances are nearly free (crossover 144) — what is
  left is irreducible: the pool's `3 × obj_chunk` floor (7.1 GB) plus the
  `[1, 24576, 24576]` tomo FDE buffer (4.5 GB). Both scale with `nobj`, which no
  flag shrinks.
* **Single tile n = 8192** — object `13056³`. The floor at
  `nchunk = 1 --ndistchunk 1` is 20.1 GB, ~60 GB actual. Unlike on the H100,
  cutting `--ntheta` does not rescue it: only the `tomo sino` buffer depends on
  `ntheta`, and it is well under a gigabyte.

So on Polaris **bin 1 is the largest mosaic and 4096 the largest single tile**.
Bin 1 is the better scaling-study size anyway — it runs at `nchunk = 2` instead
of bin 0's forced `nchunk = 1`.

If a bin-0 *voxel size* matters more than the full mosaic width, a narrower
mosaic does fit, at the cost of covering less of the sample:

| shape | ndist | object | nodes | ranks | lth | `NCHUNK` | host/node | plan GPU | ~actual GPU |
|-------|-------|--------|-------|-------|-----|-----------|-----------|----------|-------------|
| `--ntile-h 2` | 8 | 4096 × 6144² | 32 | 128 | 47 | 2 | 275 GB | 7.8 GB | ~24 GB |
| `--ntile-h 3` | 12 | 4096 × 8192² | 32 | 128 | 47 | 1 | 420 GB | 7.2 GB | ~22 GB |
|               |    |              | 64 | 256 | 24 | 1 | 214 GB | 7.2 GB | ~22 GB |
| `--ntile-v 2 --ntile-h 3` | 24 | 6144 × 8192² | 128 | 512 | 12 | 1 | 171 GB | 8.6 GB | ~26 GB |

The 2 × 3 shape skips the otherwise natural 64-node point on purpose: there
`lth` would be 24, equal to its `ndist`, which rule 3 below rejects.

#### Copy-paste: complete job scripts (PBS)

Each job is one `qsub` of one of the two scripts, with the header and the size
block edited to match a row of the tables above. `-A 17445` is already set;
`-l filesystems` and the queue are the only other site-specific bits.

**Job 1 — mosaic bin 3, 1 × 5 (`select=1`, `debug`).** The shipped defaults:

```bash
#PBS -l select=1:system=polaris
#PBS -q debug
...
BIN=3
NTILE_V=1
NCHUNK=32
```

```bash
qsub run_mosaic_polaris.sh
```

**Job 2 — mosaic bin 2, 1 × 5 (`select=1`, `debug`).** Same file, three lines
different: `BIN=2`, `NCHUNK=8`. For 2 × 5 use `select=2` and `NTILE_V=2`.

**Job 3 — mosaic bin 1, 1 × 5 (`select=8`, `-q debug-scaling`).**

```bash
#PBS -l select=8:system=polaris
#PBS -q debug-scaling
...
BIN=1
NTILE_V=1
NCHUNK=2
```

**Job 4 — mosaic bin 1, 2 × 5 (`select=16`, `-q prod`, `walltime=03:00:00`).**
Same as job 3 with `NTILE_V=2`.

**Job 5 — single tile 512 / 1024 (`select=1`, `debug`).** In
`run_polaris.sh`: `N=512, NCHUNK=16`, then `N=1024, NCHUNK=32`. Two
submissions, or one job with the `mpiexec` line repeated for both sizes.

**Job 6 — single tile 2048 (`select=4`, `-q debug-scaling`).** `N=2048`,
`NCHUNK=8`.

**Job 7 — single tile 4096 (`select=32`, `-q prod`, `walltime=06:00:00`).**
`N=4096`, `NCHUNK=2`.

Each script appends its parsed summary to its own log, so the log files are the
deliverable. Queue fit, for reference (verify with `qstat -Qf <queue>` —
policies change):

| runs | nodes | queue | max walltime |
|------|-------|-------|--------------|
| mosaic bin 3 / 2 (1×5), single tile 512 / 1024 | 1–2 | `debug` | 1 h |
| mosaic bin 1 (1×5), single tile 2048 | 4–8 | `debug-scaling` | 1 h |
| mosaic bin 1 (2×5) | 16 | `prod` → small | 3 h |
| single tile 4096 | 32 | `prod` → medium | 6 h |

#### If you have fewer nodes than the table asks

The H100 advice — cut `--ntheta`, host memory is close to linear in it — is
weaker at bin 1, and it is worth knowing before you spend a job on it. The
object triple (`3 × lnz × nobj²`) does not depend on `ntheta` at all, so it
becomes the floor. For 1×5 bin 1 on 8 nodes:

| `--ntheta` | host/node |
|------------|-----------|
| 3000 (full) | 388 GB |
| 1500 | 303 GB |
| 750 | 261 GB |

A 4× cut in angles buys 33 %. To get bin 1 onto fewer nodes, drop tiles
(`--ntile-h 3`) instead — that shrinks `nobj`, which is the term that dominates.

### Three rules the code enforces

1. **`nchunk ≤ lth`.** A chunk larger than the rank's own slab of angles just
   sizes the pool up for nothing. (The tables above cap `nchunk` at `lth/4` so
   there are ≥ 2 chunks in flight for the H2D/compute/D2H overlap, and round
   down to a power of two; the memory maximum is higher.)
2. **`lth ≥ 4`.** `rec_mpi` asserts `local_ntheta != 3`. Since
   `lth ≈ ntheta/nranks`, that caps useful ranks at `ntheta/4`: 187 at bin 3,
   375 at bin 2, 750 at bin 1, 1500 at bin 0 for the mosaic; with
   `ntheta = 1800 n / 2048`, 112 / 225 / 450 / 900 / 1800 for single-tile
   n = 512 … 8192.
3. **`ndistchunk ≠ lth`.** The chunking layer tells theta-chunked arrays from
   broadcast ones by comparing `shape[0]` to the chunk size, so a distance
   group of exactly `lth` would be misread. It aborts loudly, not silently, if
   you hit it with a custom `--ntheta` or rank count.

### Check before you burn allocation

`--plan` prints the per-rank host and GPU budget and exits without allocating
anything. Both scripts have it:

```bash
python test_mosaic.py --bin 1 --nchunk 4 --nranks 32 --plan
python test.py --n 8192 --ntheta 7200 --nchunk 1 --ndistchunk 2 --nranks 1024 --plan
./run_mosaic.sh --plan          # whatever BIN / NCHUNK the script says
./run.sh --plan                 # whatever N / NCHUNK the script says
./run_mosaic_polaris.sh --plan  # same, on a login node
./run_polaris.sh --plan
```

The four drivers plan the one configuration written into them, so to sweep a
range edit `BIN` / `N` and `NCHUNK` between calls, or call `test*.py --plan`
directly as in the first two lines.

**Calibration:** the host figure is accurate — measured RSS came in within 4%
of prediction. The GPU figure is a **lower bound**: it counts the chunking pool
and the persistent buffers but not cuFFT work areas or transient kernel
scratch. Across five recorded runs the CuPy pool measured **2.5–3.6× the
prediction**, so budget `plan × 3` and keep headroom — that is what the
`~actual GPU` column is. (Those runs predate a change that removed ~600 MB of
per-chunk temporaries, so the ratio should now be a little better, but plan on
3× until you have a measurement of your own.)

### If it OOMs

* **GPU OOM** → halve `--nchunk` and rerun; if already at 1, lower
  `--ndistchunk` (single tile only — on the mosaic it will not help, see
  above). Node count does not help GPU memory.
* **Host OOM / pinned allocation failure** → double the node count, or cut
  `--ntheta`.
* Confirm one rank per GPU: `set_affinity_gpu.sh` echoes
  `<rank> uses <dev> of <ngpus> <host>` for every rank at startup, and the log
  header records `gpus_used=` and the rank→device map. Two ranks sharing a
  device roughly halves effective PCIe bandwidth and quietly distorts every
  timing.

### What to report back

The log files — `logmosaic<VxH>_<bin>_<ranks>` from benchmark A and
`log<n>_<nchunk>` from benchmark B (`log<n>_<nchunk>_<nodes>n` from
`run_polaris.sh`). Each already ends with the
`parse_perf_log.py` breakdown: per-function timings and the grouped categories
(`gradient` / `hessian` / `redist` / `linear_batch` / `min` / `allreduce`).
Also useful: node count, ranks, the exact command line, and anything you had to
change to make it fit.

## `parse_perf_log.py`

```bash
python parse_perf_log.py [LOG] [--iter N]
```

Every driver calls this on its own log when it finishes, so **every log
already ends with its own summary**. `--iter` defaults to the **last iteration
present in the log**, so a two-iteration run is summarised after the JIT warmup
without anyone having to say so. Running it by hand is only needed for an
earlier iteration or for an older log.

Reads the perf log and prints:

1. **Max process memory and max GPU memory** seen across the whole run
   (parsed from every `@timer`-decorated function entry).
2. **Per-function summary for the requested iter** (default: the last iter in
   the log — with `--niter 2` that is iter 1, the one after JIT warmup).
   Columns: calls, total seconds, mean ms, percentage of iter total.
3. **Grouped categories** — totals for the BH building blocks of iter `N`:

   | category   | members                                                    |
   |------------|------------------------------------------------------------|
   | `gradient` | `gradients_cascade + adj_tomo + fwd_tomo + gradient` (the last being the extra_terms regularizer) |
   | `hessian`  | `hessian_cascade[3] + hessian[3]` — the cascade sweeps plus their extra_terms regularizers |
   | `redist`   | both MPI `redist` calls inside the iter                    |
   | `linear_batch` | `linear_batch + linear_redot_batch`                    |
   | `min`      | the line-search `Rec.min()`                                |
   | `allreduce`    | `allreduce + allreduce2 + allreduce_scalars`           |
   | `other`        | any remaining timed functions (empty on a healthy log) |

   With `fused_hessian=false` the solver runs three separate `Rec.hessian()`
   calls per iter (β-num, β-den, α-den) and the `hessian` row is followed by a
   `#1 / #2 / #3` breakdown, one per call. `fused_hessian=true` (the default)
   derives all three forms from one `hessian_cascade3` sweep, so there is a
   single instance and no breakdown.

   Iter boundaries are detected from `error_debug`'s `iter=N: <t>sec ...` lines,
   so the perf log needs `error_step ≥ 1` (default in `test.py`). The `<t>sec`
   anchor matters: `rec_mpi` also logs `iter=N: coeff_cache hits=…` right after
   that line, and matching those too used to reopen the bucket and leave the
   last iteration empty ("no timer calls recorded").

The same report is available as `report(path, iter_no, out=..., show_info=...,
show_header=...)` — that is what the benchmarks call to append
their own summary.

## What goes into the log

`test.py` writes (rank 0 only, plain text, no ANSI codes):

* **Machine info** — CPU model + cores, total RAM, RAM modules (best-effort
  via `dmidecode`/`/sys/firmware/dmi`, "unavailable" if neither works), GPU
  model + memory + visible count.
* **Job layout** — `ranks`, `hosts`, `gpus_used`; per-host summary; one line
  per rank with `(host, dev)` mapping. The physical device id is derived from
  `CUDA_VISIBLE_DEVICES` so it's correct even when one GPU is pinned per rank.
* **Run config** — the `perf-test: n= nz= nobj= ntheta= ndist= nchunk= …` line.
* **`@timer` lines** — one per call, e.g.
  `gradients_cascade: 0.1234 sec, process memory 1.23 GB, GPU memory 4.56 GB`.
* **`iter=N: …sec err=…`** — emitted by `error_debug` at the end of each
  BH iter, used as iter-boundary markers by the parser.
* **Final summary** — `BH done: niter iters in T s (ms/iter) [nranks, n, nobj, ntheta, ndist, nchunk]`.

## Tuning notes

* **`nchunk`** dominates memory vs throughput. Keep it a power of two
  (1, 2, 4, 8, 16, 32 …) — start at 4 and double until the GPU OOMs, then step
  back one. The tables above are exactly that search, capped at `lth/4`.
* **GPU memory** for `Rec` scales with `nobj·max(nobj, nzobj, ntheta)·complex_item`
  per chunk (the formula in `Rec.__init__`); cross-check with the
  `Allocate […]` lines at startup.
* **`niter`** defaults to 1, so the single reported iteration still contains
  the CuPy JIT compile. Append `--niter 2` to a driver invocation
  (`./run.sh --niter 2`) when you want the steady state — the extra flags are
  passed straight through to the python, and the parser always breaks down the
  last iteration, so it picks up the warm one on its own.
* **Reproducibility**: the synthetic data depends only on global slice /
  theta / distance index, so the same `--n / --ntheta` is bit-identical
  whether you run with 1 rank or 32.
