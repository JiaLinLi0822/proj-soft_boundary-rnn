#!/bin/bash
# Submit from the repository root:
#   sbatch fit_tinyrnns/cluster/train_nested_cv.sbatch

#SBATCH --job-name=tinyrnn_ncv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16               # models trained simultaneously per task
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-6%7                   # one task per hidden size
#SBATCH --output=fit_tinyrnns/cluster/%x_%A_%a.out
#SBATCH --error=fit_tinyrnns/cluster/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"

# EDIT for your cluster environment.
module purge
module load python/3.11

# Each worker is a process, so prevent BLAS from spawning extra threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Array index -> hidden size. Keep #SBATCH --array consistent with this list.
HIDDEN_SIZES=(1 2 3 4 5 10 20)
HIDDEN="${HIDDEN_SIZES[${SLURM_ARRAY_TASK_ID}]}"

# EDIT training settings here, or override with sbatch --export=ALL,NAME=value.
MONKEY_ID="${MONKEY_ID:-1}"
RECURRENT_L1="${RECURRENT_L1:-1e-1 1e-2 1e-3 1e-4 1e-5}"
INIT_SEEDS="${INIT_SEEDS:-0 1 2}"
OUTER_FOLDS="${OUTER_FOLDS:-10}"
INNER_EPOCHS="${INNER_EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-3}"
PATIENCE="${PATIENCE:-200}"
FOLD_SEED="${FOLD_SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}"
OUTDIR="${OUTDIR:-fit_tinyrnns/runs/nested_cv/slurm_${SLURM_ARRAY_JOB_ID}/h${HIDDEN}}"

echo "hidden=${HIDDEN} workers=${NUM_WORKERS} l1=${RECURRENT_L1} init_seeds=${INIT_SEEDS}"

srun python -u fit_tinyrnns/train_nested_cv.py \
    --monkey-id "${MONKEY_ID}" \
    --hidden "${HIDDEN}" \
    --recurrent-l1-lambda ${RECURRENT_L1} \
    --init-seeds ${INIT_SEEDS} \
    --outer-folds "${OUTER_FOLDS}" \
    --inner-epochs "${INNER_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --early-stopping-patience "${PATIENCE}" \
    --device cpu \
    --num-workers "${NUM_WORKERS}" \
    --seed "${FOLD_SEED}" \
    --outdir "${OUTDIR}"
