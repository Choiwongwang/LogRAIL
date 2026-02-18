#!/usr/bin/env bash
set -euo pipefail

ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"

AVD_NAME="${1:-lograil_avd}"

emulator -avd "$AVD_NAME" -writable-system -no-snapshot -no-boot-anim -netdelay none -netspeed full -no-window -gpu swiftshader_indirect
