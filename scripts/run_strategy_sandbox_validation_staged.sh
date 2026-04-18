#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_ID [JUDGE_MODEL]" >&2
  exit 2
fi

RUN_ID="$1"
JUDGE_MODEL="${2:-gpt-4o-2024-11-20}"
JUDGE_TEMPLATE="${JUDGE_TEMPLATE:-markov_policy}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$ROOT/data/strategy_sandbox_runs/$RUN_ID"
LOG_DIR="$RUN_DIR"
GPU_BATCH1_A="${GPU_BATCH1_A:-1}"
GPU_BATCH1_B="${GPU_BATCH1_B:-2}"
GPU_BATCH2_A="${GPU_BATCH2_A:-$GPU_BATCH1_A}"
GPU_BATCH2_B="${GPU_BATCH2_B:-$GPU_BATCH1_B}"
SERIAL_GPU="${SERIAL_GPU:-}"

mkdir -p "$RUN_DIR"

launch_worker() {
  local worker_index="$1"
  local gpu_id="$2"
  local log_path="$LOG_DIR/worker_$(printf '%02d' "$worker_index").log"
  local pid_path="$LOG_DIR/worker_$(printf '%02d' "$worker_index").pid"
  rm -f \
    "$LOG_DIR/worker_$(printf '%02d' "$worker_index").judge_error.txt" \
    "$LOG_DIR/worker_$(printf '%02d' "$worker_index").pid"
  bash -lc \
    "cd $ROOT && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu_id conda run --no-capture-output -n llmbase python -u scripts/run_strategy_sandbox_validation.py --run-id $RUN_ID --judge-model $JUDGE_MODEL --judge-template $JUDGE_TEMPLATE --num-workers 4 --worker-index $worker_index > $log_path 2>&1" &
  local launcher_pid="$!"
  echo "$launcher_pid" > "$pid_path"
  echo "[launch] worker=$worker_index gpu=$gpu_id pid=$launcher_pid log=$log_path"
}

stop_worker() {
  local worker_index="$1"
  local pid_path="$LOG_DIR/worker_$(printf '%02d' "$worker_index").pid"
  if [[ -f "$pid_path" ]]; then
    local pid
    pid="$(cat "$pid_path")"
    pkill -P "$pid" >/dev/null 2>&1 || true
    kill "$pid" >/dev/null 2>&1 || true
  fi
}

worker_done_path() {
  local worker_index="$1"
  echo "$RUN_DIR/worker_$(printf '%02d' "$worker_index").json"
}

worker_err_path() {
  local worker_index="$1"
  echo "$RUN_DIR/worker_$(printf '%02d' "$worker_index").judge_error.txt"
}

worker_pid_path() {
  local worker_index="$1"
  echo "$RUN_DIR/worker_$(printf '%02d' "$worker_index").pid"
}

collect_pending_workers() {
  local pending=()
  local worker_index
  for worker_index in 0 1 2 3; do
    if [[ ! -f "$(worker_done_path "$worker_index")" ]]; then
      pending+=("$worker_index")
    fi
  done
  echo "${pending[*]}"
}

wait_for_workers() {
  local workers=("$@")
  while true; do
    local all_done=1
    local worker_index
    for worker_index in "${workers[@]}"; do
      local done_path
      local err_path
      local pid_path
      done_path="$(worker_done_path "$worker_index")"
      err_path="$(worker_err_path "$worker_index")"
      pid_path="$(worker_pid_path "$worker_index")"
      if [[ -f "$err_path" ]]; then
        echo "[abort] detected judge failure for worker=$worker_index"
        echo "[abort] $(cat "$err_path")"
        stop_worker "$worker_index"
        exit 1
      fi
      if [[ -f "$pid_path" ]] && ! kill -0 "$(cat "$pid_path")" >/dev/null 2>&1 && [[ ! -f "$done_path" ]]; then
        echo "[abort] worker $worker_index exited without completion"
        exit 1
      fi
      if [[ ! -f "$done_path" ]]; then
        all_done=0
      fi
    done
    if [[ "$all_done" -eq 1 ]]; then
      echo "[batch-complete] workers=[${workers[*]}]"
      return
    fi
    sleep 30
  done
}

echo "[run] staged validation run_id=$RUN_ID judge_model=$JUDGE_MODEL judge_template=$JUDGE_TEMPLATE batch1=[$GPU_BATCH1_A,$GPU_BATCH1_B] batch2=[$GPU_BATCH2_A,$GPU_BATCH2_B]"

if [[ -n "$SERIAL_GPU" ]]; then
  echo "[mode] serial single-gpu=$SERIAL_GPU"
  read -r -a pending_workers <<< "$(collect_pending_workers)"
  local_worker=""
  for local_worker in "${pending_workers[@]}"; do
    [[ -z "$local_worker" ]] && continue
    launch_worker "$local_worker" "$SERIAL_GPU"
    wait_for_workers "$local_worker"
  done

  cd "$ROOT"
  conda run --no-capture-output -n llmbase python -u scripts/run_strategy_sandbox_validation.py --run-id "$RUN_ID" --merge-only
  echo "[done] merged run_id=$RUN_ID"
  exit 0
fi

read -r -a pending_workers <<< "$(collect_pending_workers)"
if [[ "${#pending_workers[@]}" -ge 1 && -n "${pending_workers[0]}" ]]; then
  launch_worker "${pending_workers[0]}" "$GPU_BATCH1_A"
fi
if [[ "${#pending_workers[@]}" -ge 2 && -n "${pending_workers[1]}" ]]; then
  launch_worker "${pending_workers[1]}" "$GPU_BATCH1_B"
fi
if [[ "${#pending_workers[@]}" -ge 1 && -n "${pending_workers[0]}" ]]; then
  wait_for_workers "${pending_workers[@]:0:2}"
fi

if [[ "${#pending_workers[@]}" -ge 3 && -n "${pending_workers[2]}" ]]; then
  launch_worker "${pending_workers[2]}" "$GPU_BATCH2_A"
fi
if [[ "${#pending_workers[@]}" -ge 4 && -n "${pending_workers[3]}" ]]; then
  launch_worker "${pending_workers[3]}" "$GPU_BATCH2_B"
fi
if [[ "${#pending_workers[@]}" -ge 3 && -n "${pending_workers[2]}" ]]; then
  wait_for_workers "${pending_workers[@]:2:2}"
fi

cd "$ROOT"
conda run --no-capture-output -n llmbase python -u scripts/run_strategy_sandbox_validation.py --run-id "$RUN_ID" --merge-only
echo "[done] merged run_id=$RUN_ID"
