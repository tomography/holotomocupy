#!/bin/bash
# No -l on purpose: a login shell re-sources ~/.bashrc on every rank, which
# re-runs its module loads and can swap PrgEnv (and with it the cray-mpich
# lib dir) out from under the environment the job script set up.  PALS
# forwards that environment to the ranks already.
num_gpus=4
gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
echo "RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}"
exec "$@"
