#!/usr/bin/env bash
# power_recorder.sh — flight recorder for sudden power-offs (DGX Spark / GX10).
#
# When the firmware cuts power for heat or an overloaded power supply, the
# system log just stops: there is no shutdown sequence and no error to read.
# This logs GPU temperature and power draw, the hottest thermal zone, free
# memory, load and the busiest process every few seconds, and fsyncs each
# line, so the last seconds before a cut survive on disk.
#
#   systemd-run --user --unit=power-recorder --collect ./scripts/power_recorder.sh
#                                         # start; runs until stopped or reboot
#   systemctl --user stop power-recorder  # stop
#   tail -30 ~/power-recorder.csv         # after a power-off: what led up to it
#
# Arguments: [csv path, default ~/power-recorder.csv] [interval s, default 2]
set -u
f="${1:-$HOME/power-recorder.csv}"
every="${2:-2}"
[[ -s "$f" ]] || echo "time,gpu_temp_c,gpu_power_w,hottest_zone_c,mem_avail_gb,load1,top_process:cpu%" > "$f"
echo "# start $(date -Is) boot $(cat /proc/sys/kernel/random/boot_id)" >> "$f"
while :; do
  gpu="$(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  hottest=0
  for t in /sys/class/thermal/thermal_zone*/temp; do
    v="$(cat "$t" 2>/dev/null || echo 0)"
    (( v > hottest )) && hottest=$v
  done
  mem="$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)"
  load="$(cut -d' ' -f1 /proc/loadavg)"
  top="$(ps -eo comm,pcpu --sort=-pcpu --no-headers | head -1 | awk '{print $1":"$2}')"
  echo "$(date +%T),${gpu:-,},$((hottest / 1000)),$mem,$load,$top" >> "$f"
  sync "$f" 2>/dev/null || sync
  sleep "$every"
done
