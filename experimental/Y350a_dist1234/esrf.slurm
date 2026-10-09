#!/bin/bash
#SBATCH --job-name holo
#SBATCH --partition=gpuhop-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mem=0

##SBATCH --nodelist=gpbm18-[01-21]
##SBATCH --nodelist=frcrg-gpu5h200-01
##SBATCH --nodelist=vgpu5h200x4-01
##SBATCH --nodelist gpu4a100-37

# --- user configuration ---
CONFIG=config_step6.conf
SCRIPT=step6.py
# --------------------------

SCRIPT_DIR="$(pwd)"
rec_dir="$(dirname "${SCRIPT_DIR}")"

# snapshot only py and conf files into a dated folder inside rec_dir
scripts_dir="${rec_dir}/scripts$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${scripts_dir}"
cp "${SCRIPT_DIR}"/*.py   "${scripts_dir}/" 2>/dev/null || true
cp "${SCRIPT_DIR}"/*.conf "${scripts_dir}/" 2>/dev/null || true

cd "${rec_dir}"
exec > >(tee "${scripts_dir}/slurm-${SLURM_JOB_ID}.out" "${SCRIPT_DIR}/slurm-${SLURM_JOB_ID}.out") 2>&1

echo "Sample dir:  ${SCRIPT_DIR}"
echo "Rec dir:     ${rec_dir}"
echo "Snapshot:    ${scripts_dir}"
hostname

module add mamba cuda
mamba activate /home/esrf/viktor17051990/.conda/holotomocupy
export MATHDX_ROOT=/home/esrf/viktor17051990/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12

# srun --mpi=pmix_v3 "${SCRIPT_DIR}/set_affinity_gpu.sh" python "${SCRIPT_DIR}/step0.py" "${SCRIPT_DIR}/config_step0.conf"
srun --mpi=pmix_v3 "${SCRIPT_DIR}/set_affinity_gpu.sh" python "${SCRIPT_DIR}/${SCRIPT}" "${SCRIPT_DIR}/${CONFIG}"
# srun --mpi=pmix_v3 "${SCRIPT_DIR}/set_affinity_gpu.sh" python "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15.conf"
