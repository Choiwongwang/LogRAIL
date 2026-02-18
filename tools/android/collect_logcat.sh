#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-output/logcat_raw.txt}"
mkdir -p "$(dirname "$OUT")"

adb wait-for-device
adb logcat -c
adb logcat -v threadtime >> "$OUT"
