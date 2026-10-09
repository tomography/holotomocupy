#!/bin/bash
# Pre-flight GPU probe for Polaris.  Writes a hostfile containing only the
# nodes on which every GPU can actually create a CUDA context, so a node that
# comes up with
#     cudaErrorDevicesUnavailable: CUDA-capable device(s) is/are busy or unavailable
#     cudaErrorNoDevice:           no CUDA-capable device is detected
# is dropped before the real run starts instead of killing it 30 s in.
#
#   Usage:  gpu_healthcheck.sh <out_hostfile> [gpus_per_node] [min_nodes]
#   Exit 0 -> <out_hostfile> written with >= min_nodes healthy nodes.
#   Exit 1 -> too few healthy nodes (or probe could not run); caller decides.
#
# PBS has no equivalent of Slurm's --exclude: `-l select=...` can pin hosts
# (host=/vnode=) but cannot negate them, so a bad node can only be filtered
# at run time, from inside the job.  A static blacklist is also the wrong
# tool here -- in the failures seen so far no single node recurs, the bad GPU
# moves around, so the check has to be per-job.
set -u

OUT="${1:?usage: gpu_healthcheck.sh <out_hostfile> [gpus_per_node] [min_nodes]}"
GPN="${2:-4}"
MIN="${3:-1}"

: "${PBS_NODEFILE:?not inside a PBS job}"
NN=$(wc -l < "${PBS_NODEFILE}")
RAW="${OUT}.probe"

# Each probe rank pins itself to one GPU exactly the way set_affinity_gpu_polaris.sh
# does, then forces a real context + allocation -- nvidia-smi is not enough,
# it happily lists a GPU that is already held by another process.
read -r -d '' PROBE <<'PY' || true
import os, socket
tag = "%s %s" % (socket.gethostname(), os.environ.get("CUDA_VISIBLE_DEVICES", "?"))
try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    cp.zeros(1) + 1
    cp.cuda.Device(0).synchronize()
    print("HC_OK", tag, flush=True)
except Exception as e:
    print("HC_BAD", tag, type(e).__name__, e, flush=True)
PY

echo "[healthcheck] probing ${NN} nodes x ${GPN} GPUs ..." >&2
timeout 300 mpiexec -n "$(( NN * GPN ))" --ppn "${GPN}" --hostfile "${PBS_NODEFILE}" \
    bash -c 'export CUDA_VISIBLE_DEVICES=$(( '"${GPN}"' - 1 - PMI_LOCAL_RANK % '"${GPN}"' )); exec python -c "$0"' "${PROBE}" \
    > "${RAW}" 2>&1
rc=$?
[ ${rc} -ne 0 ] && echo "[healthcheck] WARNING: probe mpiexec exited ${rc} (a node may have failed to launch)" >&2

grep '^HC_BAD' "${RAW}" >&2 || true

# A node passes only if all GPN of its GPUs answered.  Short hostnames are
# matched back to the full .hsn.* nodefile entries so MPI still routes over
# the high-speed network.
grep '^HC_OK' "${RAW}" | awk -v g="${GPN}" '{c[$2]++} END{for (h in c) if (c[h] >= g) print h}' | sort -u > "${OUT}.good"
awk 'NR==FNR{ok[$1]=1; next} {split($1, p, "."); if (p[1] in ok) print}' "${OUT}.good" "${PBS_NODEFILE}" > "${OUT}"

NG=$(wc -l < "${OUT}")
echo "[healthcheck] healthy: ${NG}/${NN} nodes" >&2
if [ "${NG}" -lt "${MIN}" ]; then
    echo "[healthcheck] ERROR: only ${NG} healthy nodes, need ${MIN}" >&2
    exit 1
fi
comm -23 <(awk '{split($1,p,"."); print p[1]}' "${PBS_NODEFILE}" | sort -u) "${OUT}.good" \
    | sed 's/^/[healthcheck] EXCLUDED: /' >&2
exit 0
