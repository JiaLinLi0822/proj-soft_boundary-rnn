#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_SCRIPT="${SCRIPT_DIR}/training.py"
RESULTS_ROOT="${SCRIPT_DIR}/results"
CACHE_ROOT="${RESULTS_ROOT}/.cache"
MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
XDG_CACHE_HOME="${CACHE_ROOT}"

# 8-stimulus regime used for all training jobs.
STIMULI_LOGLR="-0.9,-0.7,-0.5,-0.3,0.3,0.5,0.7,0.9"
NUM_STIMULI=8
REWARD=1.0
URGENCY_COST=0.00
NUM_EPISODES=1000000
MAX_JOBS=4

# Sample-cost sweep to train in parallel.
COST_LEVELS=(0.01 0.02 0.03 0.04 0.05)

# regime_name,max_samples,max_steps
SETTINGS=(
  "time_constrained,10,10"
  "no_time_constraint,10000,10000"
)

mkdir -p "${RESULTS_ROOT}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}/fontconfig"

run_one() {
  local regime="$1"
  local cost="$2"
  local max_samples="$3"
  local max_steps="$4"

  local regime_path="${RESULTS_ROOT}/${regime}"
  local cost_tag="${cost/./p}"
  local jobid="${regime}_cost=${cost}_reward=${REWARD}_urgency_cost=${URGENCY_COST}_logLR=[${STIMULI_LOGLR}]_max_samples=${max_samples}_max_steps=${max_steps}_epNum=${NUM_EPISODES}"

  mkdir -p "${regime_path}"

  echo "[START] ${regime} | cost=${cost} | max_samples=${max_samples} | max_steps=${max_steps}"
  echo "  Saving to: ${regime_path}/exp_${jobid}"
  echo "  Stimuli: ${NUM_STIMULI}"
  echo "  logLR: [${STIMULI_LOGLR}]"

  conda run -n deeprl python "${PYTHON_SCRIPT}" \
    --jobid "${jobid}" \
    --path "${regime_path}" \
    --hidden_size 64 \
    --num_trials 1 \
    --max_samples "${max_samples}" \
    --max_steps "${max_steps}" \
    --num_stimuli "${NUM_STIMULI}" \
    --stimuli_loglr="${STIMULI_LOGLR}" \
    --sampling_cost "${cost}" \
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

  echo "[DONE] ${regime} | cost=${cost} | tag=${cost_tag}"
}

export -f run_one
export PYTHON_SCRIPT RESULTS_ROOT STIMULI_LOGLR NUM_STIMULI REWARD URGENCY_COST NUM_EPISODES
export MPLCONFIGDIR XDG_CACHE_HOME

jobs=()
for setting in "${SETTINGS[@]}"; do
  IFS=',' read -r regime max_samples max_steps <<< "${setting}"
  for cost in "${COST_LEVELS[@]}"; do
    jobs+=("${regime},${cost},${max_samples},${max_steps}")
  done
done

printf "%s\n" "${jobs[@]}" \
  | xargs -P "${MAX_JOBS}" -I {} bash -c 'IFS="," read -r regime cost max_samples max_steps <<< "$1"; run_one "$regime" "$cost" "$max_samples" "$max_steps"' _ {}

echo "All jobs finished."
