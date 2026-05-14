#!/usr/bin/env bash
set -u

LOG=/home/erol/lora_training/logs/r01.cpu.resume.log
OUT=/home/erol/lora_training/output/rag_synthesis
STATE=/home/erol/lora_training/logs/r01.monitor.state
ALERT=/home/erol/lora_training/logs/r01.monitor.alerts.log
INTERVAL=60
STALL_WARN_SAMPLES=3
STALL_FAIL_SAMPLES=10
TRAIN_PATTERN='train_lora_rocm.py .*--adapter r01'

mkdir -p "$(dirname "$STATE")" "$(dirname "$ALERT")"
touch "$ALERT" "$STATE"

samples=()
last_issue=""
log_baseline_bytes=0

if [ -f "$LOG" ]; then
  log_baseline_bytes=$(wc -c < "$LOG" 2>/dev/null | tr -d ' ')
fi

alert() {
  printf '%s %s\n' "$(date -Is)" "$1" | tee -a "$ALERT"
}

while true; do
  issue="ok"
  issue_detail=""

  if [ ! -f "$LOG" ]; then
    issue="log_missing"
    issue_detail="training log missing: $LOG"
  elif [ ! -d "$OUT" ]; then
    issue="output_missing"
    issue_detail="output directory missing: $OUT"
  else
    size=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    files=$(find "$OUT" -type f 2>/dev/null | wc -l | tr -d ' ')
    checkpoint_count=$(find "$OUT" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | wc -l | tr -d ' ')
    log_size=$(wc -c < "$LOG" | tr -d ' ')
    stamp="$(date -Is) size=$size files=$files checkpoints=$checkpoint_count log_bytes=$log_size"

    if [ "$log_size" -lt "$log_baseline_bytes" ]; then
      log_baseline_bytes=$log_size
    fi

    printf '%s\n' "$stamp" >> "$STATE"

    samples+=("$size|$files|$checkpoint_count|$log_size")
    if [ "${#samples[@]}" -gt "$STALL_FAIL_SAMPLES" ]; then
      samples=("${samples[@]: -$STALL_FAIL_SAMPLES}")
    fi

    if ! pgrep -af "$TRAIN_PATTERN" >/dev/null; then
      issue="process_missing"
      issue_detail="training process is not running"
    else
      recent_log=""
      if [ "$log_size" -gt "$log_baseline_bytes" ]; then
        recent_log="$(tail -c +"$((log_baseline_bytes + 1))" "$LOG" 2>/dev/null || true)"
      fi
      if printf '%s' "$recent_log" | grep -Ei "(GPU Hang|RuntimeError|Traceback|OOM|out of memory|killed process|CUDA error|error:)" >/dev/null; then
        issue="log_error"
        issue_detail="error pattern found in training log since monitor baseline"
      fi

      if [ "$issue" != "ok" ]; then
        :
      elif [ "${#samples[@]}" -ge "$STALL_FAIL_SAMPLES" ]; then
        same=1
        for ((i=1; i<STALL_FAIL_SAMPLES; i++)); do
          if [ "${samples[-1]}" != "${samples[-$((i + 1))]}" ]; then
            same=0
            break
          fi
        done
        if [ "$same" -eq 1 ]; then
          issue="stalled_fail"
          issue_detail="artifact has not grown across ${STALL_FAIL_SAMPLES} consecutive samples"
        fi
      fi

      if [ "$issue" = "ok" ] && [ "${#samples[@]}" -ge "$STALL_WARN_SAMPLES" ]; then
        same=1
        for ((i=1; i<STALL_WARN_SAMPLES; i++)); do
          if [ "${samples[-1]}" != "${samples[-$((i + 1))]}" ]; then
            same=0
            break
          fi
        done
        if [ "$same" -eq 1 ]; then
          issue="stalled_warn"
          issue_detail="artifact has not grown across ${STALL_WARN_SAMPLES} consecutive samples"
        fi
      fi

      if [ "$issue" = "ok" ] && [ "${#samples[@]}" -ge 2 ]; then
        last="${samples[-1]}"
        prev="${samples[-2]}"
        if [ "$last" = "$prev" ]; then
          issue="stalled_pair"
          issue_detail="artifact appears stalled between 2 consecutive samples"
        fi
      fi
    fi
  fi

  if [ "$issue" != "ok" ] && [ "$issue" != "$last_issue" ]; then
    alert "FAIL: $issue_detail"
    if [ -f "$LOG" ]; then
      tail -n 40 "$LOG" | tee -a "$ALERT"
    fi
  fi
  last_issue="$issue"

  sleep "$INTERVAL"
done
