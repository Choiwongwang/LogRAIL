#!/usr/bin/env bash
set -euo pipefail

TAG="LOGRAIL_SCENARIO"
ANOM_TAG="LOGRAIL_ANOM"
NORMAL_TAG="LOGRAIL_NORMAL"

log_start() {
  local id="$1"; local name="$2"
  adb shell log -t "$TAG" -p i "START id=$id name=$name"
}

log_end() {
  local id="$1"; local name="$2"
  adb shell log -t "$TAG" -p i "END id=$id name=$name"
}

log_normal() {
  local id="$1"; local name="$2"
  adb shell log -t "$NORMAL_TAG" -p i "NORMAL id=$id name=$name"
}

emit_info() {
  local msg="$1"
  adb shell log -t "$ANOM_TAG" -p i "$msg"
}

emit_err() {
  local msg="$1"
  adb shell log -t "$ANOM_TAG" -p e "$msg"
}

adb wait-for-device

# Try to enable root (some scenarios require it)
adb root >/dev/null 2>&1 || true
sleep 1

RUNS="${1:-1}"
INTERVAL="${2:-2}"
NORMAL_GAP="${3:-2}"
NORMAL_REPEAT="${4:-3}"

run_scenario() {
  local id="$1"; local name="$2"; shift 2
  log_start "$id" "$name"
  "$@"
  log_end "$id" "$name"
  sleep "$INTERVAL"
}

run_normal() {
  local id="$1"; local name="$2"; shift 2
  log_normal "$id" "$name"
  for _ in $(seq 1 "$NORMAL_REPEAT"); do
    "$@"
  done
  sleep "$NORMAL_GAP"
}

normal_activity() {
  adb shell "cmd activity get-config" >/dev/null 2>&1 || true
  adb shell "dumpsys battery | head -n 5" >/dev/null 2>&1 || true
  adb shell "settings get global airplane_mode_on" >/dev/null 2>&1 || true
  adb shell "cmd package list packages | head -n 5" >/dev/null 2>&1 || true
  adb shell "am start -a android.intent.action.MAIN -c android.intent.category.HOME" >/dev/null 2>&1 || true
  adb shell log -t "$NORMAL_TAG" -p i "normal_activity: device checks"
  adb shell log -t "$NORMAL_TAG" -p i "normal_activity: settings read"
  adb shell log -t "$NORMAL_TAG" -p i "normal_activity: package list"
  adb shell log -t "$NORMAL_TAG" -p i "normal_activity: home intent"
  adb shell log -t "$NORMAL_TAG" -p i "normal_activity: idle"
}

scenario_01_storage_full() {
  adb shell "dd if=/dev/zero of=/data/local/tmp/fill.bin bs=1M count=50" || true
  emit_err "storage_full simulated: dd to /data/local/tmp"
  adb shell "rm -f /data/local/tmp/fill.bin" || true
}

scenario_02_permission_denied() {
  adb shell "cat /data/misc/keystore/user_0/*" || true
  emit_err "permission_denied simulated: keystore access"
}

scenario_03_selinux_denial() {
  adb shell "setenforce 1" || true
  adb shell "cat /sys/fs/selinux/policy" || true
  emit_err "selinux_denial simulated"
}

scenario_04_missing_config() {
  adb shell "cat /data/local/tmp/nonexistent.conf" || true
  emit_err "missing_config simulated: nonexistent.conf"
}

scenario_05_missing_lib() {
  emit_err "missing native library: libmissing.so"
}

scenario_06_missing_shared_lib() {
  emit_err "missing shared library: libshared.so"
}

scenario_07_service_not_found() {
  adb shell "service call bogus 1" || true
  emit_err "service_not_found simulated: bogus service call"
}

scenario_08_binder_ipc_failure() {
  adb shell "service call activity 9999" || true
  emit_err "binder_ipc_failure simulated"
}

scenario_09_anr_sim() {
  emit_err "ANR simulated: main thread blocked"
}

scenario_10_crash_sigabrt() {
  emit_err "CRASH simulated: SIGABRT"
}

scenario_11_crash_sigsegv() {
  emit_err "CRASH simulated: SIGSEGV"
}

scenario_12_oom() {
  emit_err "OOM simulated: memory exhaustion"
}

scenario_13_socket_connect_fail() {
  adb shell "toybox nc 127.0.0.1 65000" || true
  emit_err "socket_connect_fail simulated: 127.0.0.1:65000"
}

scenario_14_dns_failure() {
  adb shell "ping -c 1 non.existent.domain" || true
  emit_err "dns_failure simulated: non.existent.domain"
}

scenario_15_tls_cert_error() {
  emit_err "TLS/certificate error simulated"
}

scenario_16_keystore_error() {
  emit_err "keystore error simulated"
}

scenario_17_io_error() {
  emit_err "IO error simulated: bad file descriptor"
}

scenario_18_readonly_fs() {
  adb shell "mount -o remount,ro /" || true
  adb shell "touch /system/ro_fail_test" || true
  adb shell "mount -o remount,rw /" || true
  emit_err "readonly_fs simulated: /system write failure"
}

scenario_19_force_kill_app() {
  adb shell "am force-stop com.android.settings" || true
  emit_err "force_kill_app simulated: com.android.settings"
}

scenario_20_service_timeout() {
  emit_err "service timeout simulated"
}

scenario_21_activity_not_found() {
  adb shell "am start -n com.nonexistent/.NopeActivity" || true
  emit_err "activity_not_found simulated"
}

scenario_22_bad_intent_action() {
  adb shell "am broadcast -a com.nonexistent.ACTION_TEST" || true
  emit_err "bad_intent_action simulated"
}

scenario_23_provider_not_found() {
  adb shell "content query --uri content://com.nonexistent.provider/items" || true
  emit_err "content_provider_not_found simulated"
}

scenario_24_package_missing() {
  adb shell "pm path com.nonexistent.app" || true
  emit_err "package_missing simulated"
}

scenario_25_install_fail() {
  adb shell "pm install /data/local/tmp/nonexistent.apk" || true
  emit_err "install_fail simulated"
}

scenario_26_uninstall_fail() {
  adb shell "pm uninstall com.nonexistent.app" || true
  emit_err "uninstall_fail simulated"
}

scenario_27_settings_write_fail() {
  adb shell "settings put secure nonexistent_key 1" || true
  emit_err "settings_write_fail simulated"
}

scenario_28_network_toggle() {
  adb shell "svc wifi disable" || true
  adb shell "svc wifi enable" || true
  emit_err "network_toggle simulated"
}

scenario_29_airplane_toggle() {
  adb shell "settings put global airplane_mode_on 1" || true
  adb shell "am broadcast -a android.intent.action.AIRPLANE_MODE --ez state true" || true
  adb shell "settings put global airplane_mode_on 0" || true
  adb shell "am broadcast -a android.intent.action.AIRPLANE_MODE --ez state false" || true
  emit_err "airplane_toggle simulated"
}

scenario_30_storage_stats_fail() {
  adb shell "cmd storaged crash" || true
  emit_err "storage_stats_fail simulated"
}

for r in $(seq 1 "$RUNS"); do
  echo "[+] Run $r/$RUNS"
  run_normal 00 "baseline_activity" normal_activity
  run_scenario 01 "storage_full" scenario_01_storage_full
  run_normal 01 "between" normal_activity
  run_scenario 02 "permission_denied" scenario_02_permission_denied
  run_normal 02 "between" normal_activity
  run_scenario 03 "selinux_denial" scenario_03_selinux_denial
  run_normal 03 "between" normal_activity
  run_scenario 04 "missing_config" scenario_04_missing_config
  run_normal 04 "between" normal_activity
  run_scenario 05 "missing_lib" scenario_05_missing_lib
  run_normal 05 "between" normal_activity
  run_scenario 06 "missing_shared_lib" scenario_06_missing_shared_lib
  run_normal 06 "between" normal_activity
  run_scenario 07 "service_not_found" scenario_07_service_not_found
  run_normal 07 "between" normal_activity
  run_scenario 08 "binder_ipc_failure" scenario_08_binder_ipc_failure
  run_normal 08 "between" normal_activity
  run_scenario 09 "anr_sim" scenario_09_anr_sim
  run_normal 09 "between" normal_activity
  run_scenario 10 "crash_sigabrt" scenario_10_crash_sigabrt
  run_normal 10 "between" normal_activity
  run_scenario 11 "crash_sigsegv" scenario_11_crash_sigsegv
  run_normal 11 "between" normal_activity
  run_scenario 12 "oom" scenario_12_oom
  run_normal 12 "between" normal_activity
  run_scenario 13 "socket_connect_fail" scenario_13_socket_connect_fail
  run_normal 13 "between" normal_activity
  run_scenario 14 "dns_failure" scenario_14_dns_failure
  run_normal 14 "between" normal_activity
  run_scenario 15 "tls_cert_error" scenario_15_tls_cert_error
  run_normal 15 "between" normal_activity
  run_scenario 16 "keystore_error" scenario_16_keystore_error
  run_normal 16 "between" normal_activity
  run_scenario 17 "io_error" scenario_17_io_error
  run_normal 17 "between" normal_activity
  run_scenario 18 "readonly_fs" scenario_18_readonly_fs
  run_normal 18 "between" normal_activity
  run_scenario 19 "force_kill_app" scenario_19_force_kill_app
  run_normal 19 "between" normal_activity
  run_scenario 20 "service_timeout" scenario_20_service_timeout
  run_normal 20 "between" normal_activity
  run_scenario 21 "activity_not_found" scenario_21_activity_not_found
  run_normal 21 "between" normal_activity
  run_scenario 22 "bad_intent_action" scenario_22_bad_intent_action
  run_normal 22 "between" normal_activity
  run_scenario 23 "provider_not_found" scenario_23_provider_not_found
  run_normal 23 "between" normal_activity
  run_scenario 24 "package_missing" scenario_24_package_missing
  run_normal 24 "between" normal_activity
  run_scenario 25 "install_fail" scenario_25_install_fail
  run_normal 25 "between" normal_activity
  run_scenario 26 "uninstall_fail" scenario_26_uninstall_fail
  run_normal 26 "between" normal_activity
  run_scenario 27 "settings_write_fail" scenario_27_settings_write_fail
  run_normal 27 "between" normal_activity
  run_scenario 28 "network_toggle" scenario_28_network_toggle
  run_normal 28 "between" normal_activity
  run_scenario 29 "airplane_toggle" scenario_29_airplane_toggle
  run_normal 29 "between" normal_activity
  run_scenario 30 "storage_stats_fail" scenario_30_storage_stats_fail
  run_normal 30 "between" normal_activity
done

echo "[✓] Scenarios completed."
