#!/usr/bin/env bash
# Shared multi-node MPI settings for the mosaic_brain scripts.
# Sourced by run_mn.sh, run_steps15_mn.sh, run_step6_mn.sh -- not run directly.
#
# Knobs (all overridable from the environment):
#   NODES     "tomo2 tomo3 tomo4 tomo5"   nodes to use
#   NP        <total GPUs over NODES>     number of ranks
#   HOSTFILE  hosts.tomo                  regenerated unless KEEP_HOSTFILE=1
#   MAP_BY    <unset>                     set to "node" to spread ranks round-robin
#   SKIP_MPI_ENV_CLEAN  0                 1 keeps the hpcx module env as-is
#   PY        .../envs/holotomocupy/bin/python
#   MPIRUN    <PY's env>/bin/mpirun
#
# Rank placement when NP is less than the total GPU count: Open MPI fills the
# node you launched from first, then the remaining nodes in hostfile order, one
# rank per GPU (slots stop it oversubscribing). MAP_BY=node spreads instead --
# rank 0 on the first node, rank 1 on the second, and so on.

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

NODES=${NODES:-"tomo2 tomo3 tomo4 tomo5"}
PY=${PY:-/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python}

# mpi4py in the holotomocupy env is linked against conda's Open MPI 5.0.8.
# The hpcx module on PATH is Open MPI 4.1.9 -- launching with it against this
# mpi4py mixes two MPI runtimes. Default to the env's own mpirun.
MPIRUN=${MPIRUN:-"$(dirname "$PY")/mpirun"}

# ~/.bashrc does `module add nvhpc-hpcx-cuda13`, which exports OPAL_PREFIX and
# PMIX_INSTALL_PREFIX pointing at hpcx's Open MPI 4.1.9 and puts its libraries
# on LD_LIBRARY_PATH. Conda's mpirun (5.0.8) obeys OPAL_PREFIX, then fails to
# find its own plugins -- it surfaces as "prterun-exec-failed" complaining about
# a missing help-mpirun.txt under the hpcx tree. Ranks on this node would also
# load hpcx's libmpi.so under a mpi4py built against 5.0.8.
#
# Remote ranks are unaffected (non-interactive ssh never defines `module`), so
# only the launching shell needs cleaning. SKIP_MPI_ENV_CLEAN=1 disables this.
SDK_ROOT=${SDK_ROOT:-/local/vnikitin/hpc_sdk_multi}

_strip_path() {   # drop every $2-containing entry from the path-list $1
    local out='' p
    local IFS=:
    for p in $1; do
        case "$p" in *"$2"*) continue ;; esac
        [ -n "$p" ] && out="${out:+$out:}$p"
    done
    printf '%s' "$out"
}

if [ "${SKIP_MPI_ENV_CLEAN:-0}" != 1 ]; then
    PATH=$(_strip_path "$PATH" "$SDK_ROOT")
    LD_LIBRARY_PATH=$(_strip_path "${LD_LIBRARY_PATH:-}" "$SDK_ROOT")
    export PATH LD_LIBRARY_PATH
    _leaked=$(env | awk -F= '/^(HPCX_|OMPI_|PMIX_|PRTE_|OPAL_)/ {print $1}')
    [ -n "$_leaked" ] && unset $_leaked
    unset MPI_HOME OMPI_HOME _leaked
    # conda's mpirun lives at <env>/bin/mpirun, so the env root is its prefix
    export OPAL_PREFIX="$(dirname "$(dirname "$MPIRUN")")"
fi

# tomo4 and tomo5 both carry the same link-local BMC address 169.254.3.1/24.
# Without pinning the interface, Open MPI's out-of-band channel hangs forever
# when the job is launched from either of them. InfiniBand (UCX, ~12 GB/s) is
# still selected on its own; these flags only constrain the TCP fallback.
IB_SUBNET=${IB_SUBNET:-10.54.113.0/24}
MCA_FLAGS=${MCA_FLAGS:-"--mca oob_tcp_if_include $IB_SUBNET --mca btl_tcp_if_include $IB_SUBNET"}

HOSTFILE=${HOSTFILE:-"$HERE/hosts.tomo"}

# Build the hostfile by asking each node how many GPUs it has, so a node that
# gains or loses a card does not silently get the wrong rank count.
build_hostfile() {
    : > "$HOSTFILE.tmp"
    local total=0
    for h in $NODES; do
        local ng
        ng=$(timeout 20 ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$h" \
                 'nvidia-smi -L 2>/dev/null | wc -l' 2>/dev/null | tr -dc '0-9')
        if [ -z "$ng" ] || [ "$ng" -eq 0 ]; then
            echo "warn: $h unreachable or reports no GPU -- skipping" >&2
            continue
        fi
        echo "$h slots=$ng" >> "$HOSTFILE.tmp"
        total=$((total + ng))
    done
    [ "$total" -gt 0 ] || { echo "error: no usable nodes in NODES='$NODES'" >&2; exit 1; }
    mv "$HOSTFILE.tmp" "$HOSTFILE"
    TOTAL_SLOTS=$total
}

if [ "${KEEP_HOSTFILE:-0}" = 1 ] && [ -s "$HOSTFILE" ]; then
    TOTAL_SLOTS=$(awk -F'slots=' '/slots=/{s+=$2} END{print s+0}' "$HOSTFILE")
else
    build_hostfile
fi

NP=${NP:-$TOTAL_SLOTS}

# --bind-to none: each rank runs OMP_NUM_THREADS=4 alongside its GPU, so the
# default core binding would pin all four threads onto one core.
MPI_ARGS=${MPI_ARGS:-"--hostfile $HOSTFILE --bind-to none ${MAP_BY:+--map-by $MAP_BY} $MCA_FLAGS"}

mn_banner() {
    echo "--- multi-node: $NP rank(s) over $(wc -l < "$HOSTFILE") node(s), $TOTAL_SLOTS GPU(s) total"
    sed 's/^/      /' "$HOSTFILE"
    echo "      mpirun: $MPIRUN"
}
