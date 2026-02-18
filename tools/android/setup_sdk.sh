#!/usr/bin/env bash
set -euo pipefail

# Android SDK install (CLI) for Linux
# Adjust ANDROID_HOME if you use a different location.

ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
CMDLINE_TOOLS_ZIP="commandlinetools-linux-11076708_latest.zip"
CMDLINE_TOOLS_URL="https://dl.google.com/android/repository/${CMDLINE_TOOLS_ZIP}"

mkdir -p "$ANDROID_HOME/cmdline-tools"
cd /tmp

if [ ! -f "$CMDLINE_TOOLS_ZIP" ]; then
  echo "[+] Downloading command line tools..."
  curl -L -o "$CMDLINE_TOOLS_ZIP" "$CMDLINE_TOOLS_URL"
fi

mkdir -p "$ANDROID_HOME/cmdline-tools/latest"
unzip -o "$CMDLINE_TOOLS_ZIP" -d "$ANDROID_HOME/cmdline-tools/latest"

export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"

yes | sdkmanager --licenses
sdkmanager "platform-tools" "emulator" "platforms;android-33" "system-images;android-33;google_apis;x86_64"

echo "[✓] SDK installed at $ANDROID_HOME"
