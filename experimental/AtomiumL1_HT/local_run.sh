#!/bin/bash
# AtomiumL1 HT -- local (single-node, multi-GPU) driver.
#
# The raw data sits on tomo5 at
#     /data3/vnikitin/ESRF/atomium/20250607/AtomiumL1
# (/data3 is not mounted on handyn), so this is the primary way to run it;
# polaris_run.sh is kept for the case where the 59 GB of EDF is staged to
# Polaris first.
#
#     ./local_run.sh                                          # steps 1-5
#     NP=4 ./local_run.sh                                     # on 4 GPUs
#     SCRIPT=step6.py CONFIG=config_step6_bin2.conf ./local_run.sh
#     # to re-run only part of steps 1-5, edit start_step= in config_steps15.conf
#
# Pick NP to match the GPUs on the box you are on:  nvidia-smi -L
# (tomo5 has 4x A100-SXM4-40GB, so NP=4).
#
# GPU-memory guide, from the sizes in config_step6_bin*.conf: bin 2 (n=512,
# nobj=816 -> a 4.3 GB object) is comfortable on tomo5's 4 x 40 GB, bin 1
# (n=1024, nobj=1632 -> 35 GB) fits across the four cards, and bin 0 (n=2048,
# nobj=3264 -> a 278 GB object) exceeds the 160 GB aggregate and needs the
# multi-node Polaris run.  Lower nchunk before anything else if a level OOMs.
set -e
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

NP=${NP:-$(nvidia-smi -L | wc -l)}
SCRIPT=${SCRIPT:-steps15.py}
CONFIG=${CONFIG:-config_steps15.conf}
# Conda env with cupy + mpi4py + h5py + fabio.
HTC_ENV=${HTC_ENV:-}
[ -n "$HTC_ENV" ] && source "$HTC_ENV"

echo "Sample dir: ${SCRIPT_DIR}"
echo "Host:       $(hostname)   GPUs: ${NP}"
echo "Running:    ${SCRIPT} ${CONFIG}"
python -c "import holotomocupy, sys; print('holotomocupy:', holotomocupy.__file__)"

# cd one level up, as polaris_run.sh does, so relative paths behave the same
cd "$(dirname "${SCRIPT_DIR}")"
# mpirun -n "${NP}" "${SCRIPT_DIR}/set_affinity_gpu.sh" \
#     python "${SCRIPT_DIR}/${SCRIPT}" "${SCRIPT_DIR}/${CONFIG}" \
#     2>&1 | tee "${SCRIPT_DIR}/log_$(basename ${CONFIG} .conf).txt"


# mpirun -n "${NP}" "${SCRIPT_DIR}/set_affinity_gpu.sh" \
#     python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2.conf" \
#     2>&1 | tee "${SCRIPT_DIR}/log_$(basename ${CONFIG} .conf).txt"

mpirun -n "${NP}" "${SCRIPT_DIR}/set_affinity_gpu.sh" \
    python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1.conf" \
    2>&1 | tee "${SCRIPT_DIR}/log_$(basename config_step6_bin1.conf .conf).txt"

mpirun -n "${NP}" "${SCRIPT_DIR}/set_affinity_gpu.sh" \
    python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin0.conf" \
    2>&1 | tee "${SCRIPT_DIR}/log_$(basename config_step6_bin0.conf .conf).txt"

