#!/usr/bin/env bash
set -euo pipefail

# Run scenarios repeatedly until log file reaches target line count.
# Usage: run_until_lines.sh <log_file> <target_lines> [interval]

LOG_FILE="${1:-LogRAIL/output/logcat_raw_v2.txt}"
TARGET="${2:-300000}"
INTERVAL="${3:-2}"

if [ ! -f "$LOG_FILE" ]; then
  echo "[x] Log file not found: $LOG_FILE"
  exit 1
fi

count_lines() {
  wc -l < "$LOG_FILE" | tr -d ' '
}

cur=$(count_lines)
echo "[i] Start lines: $cur (target: $TARGET)"

while [ "$cur" -lt "$TARGET" ]; do
  bash LogRAIL/tools/android/run_scenarios.sh 1 "$INTERVAL"
  cur=$(count_lines)
  echo "[i] lines: $cur / $TARGET"
done

echo "[✓] Target reached: $cur"
