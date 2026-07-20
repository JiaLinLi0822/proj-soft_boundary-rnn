#!/bin/bash
#SBATCH --job-name=opt_taskrnns
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/%x-%A_%a.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULTS_ROOT="${SCRIPT_DIR}/results"
LOG_ROOT="${SCRIPT_DIR}/slurm-logs"
CACHE_ROOT="${RESULTS_ROOT}/.cache"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}" "${CACHE_ROOT}/matplotlib" "${CACHE_ROOT}/fontconfig"

export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export PYTHONUNBUFFERED=1

STIMULI_LOGLR="-0.9,-0.7,-0.5,-0.3,0.3,0.5,0.7,0.9"
NUM_STIMULI=8
REWARD=1.0
URGENCY_COST=0.00
NUM_EPISODES=1000000

JOBS=(
  "time_constrained,0.01,10,10"
  "time_constrained,0.02,10,10"
  "time_constrained,0.03,10,10"
  "time_constrained,0.04,10,10"
  "time_constrained,0.05,10,10"
  "no_time_constraint,0.01,10000,10000"
  "no_time_constraint,0.02,10000,10000"
  "no_time_constraint,0.03,10000,10000"
  "no_time_constraint,0.04,10000,10000"
  "no_time_constraint,0.05,10000,10000"
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set."
  exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#JOBS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range for ${#JOBS[@]} jobs."
  exit 1
fi

IFS=',' read -r regime sample_cost max_samples max_steps <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

JOB_TAG="${regime}_cost=${sample_cost}_reward=${REWARD}_urgency_cost=${URGENCY_COST}_logLR=[${STIMULI_LOGLR}]_max_samples=${max_samples}_max_steps=${max_steps}_epNum=${NUM_EPISODES}"
SAVE_ROOT="${RESULTS_ROOT}/${regime}"

mkdir -p "${SAVE_ROOT}"

echo "Node: $(hostname)"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Regime: ${regime}"
echo "Sample cost: ${sample_cost}"
echo "Max samples: ${max_samples}"
echo "Max steps: ${max_steps}"
echo "Results: ${SAVE_ROOT}/exp_${JOB_TAG}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda run -n deeprl python "${SCRIPT_DIR}/training.py" \
    --jobid "${JOB_TAG}" \
    --path "${SAVE_ROOT}" \
    --hidden_size 64 \
    --num_trials 1 \
    --max_samples "${max_samples}" \
    --max_steps "${max_steps}" \
    --num_stimuli "${NUM_STIMULI}" \
    --stimuli_loglr="${STIMULI_LOGLR}" \
    --sampling_cost "${sample_cost}" \
    --urgency_cost "${URGENCY_COST}" \
    --reward "${REWARD}" \
    --num_episodes "${NUM_EPISODES}" \
    --lr 1e-3 \
    --batch_size 256 \
    --gamma 1.0 \
    --lamda 1.0 \
    --beta_v 0.05 \
    --beta_e 0.05 \
    --max_grad_norm 1.0
else
  echo "conda command not found. Load Conda on the cluster or edit this script for your environment."
  exit 1
fi
